# Mobilenetv3Small
# ——————MobileNetV3——————
import torch
import torch.nn as nn

# 定义一个Hard Sigmoid函数，用于SELayer中
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

# 定义一个Hard Swish函数，用于SELayer中
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# 定义Squeeze-and-Excitation（SE）模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        # Squeeze操作：全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation操作(FC+ReLU+FC+Sigmoid)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # 全连接层，将通道数降低为channel // reduction
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),  # 全连接层，恢复到原始通道数
            h_sigmoid()  # 使用Hard Sigmoid激活函数
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)  # 对输入进行全局平均池化
        y = y.view(b, c)  # 将池化后的结果展平为二维张量
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算每个通道的权重，并将其变成与输入相同的形状
        return x * y  # 将输入与权重相乘以实现通道注意力机制

# 定义卷积-批归一化-激活函数模块
class conv_bn_hswish(nn.Module):
    def __init__(self, c1, c2, stride):
        super(conv_bn_hswish, self).__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride, 1, bias=False)  # 3x3卷积层
        self.bn = nn.BatchNorm2d(c2)  # 批归一化层
        self.act = h_swish()  # 使用Hard Swish激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # 卷积 - 批归一化 - 激活函数

    def fuseforward(self, x):
        return self.act(self.conv(x))  # 融合版本的前向传播，省略了批归一化

# 定义MobileNetV3的基本模块
class MobileNetV3(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileNetV3, self).__init__()
        assert stride in [1, 2]  # 断言，要求stride必须是1或2
        self.identity = stride == 1 and inp == oup  # 如果stride为1且输入通道数等于输出通道数，则为恒等映射

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),  # 深度可分离卷积层
                nn.BatchNorm2d(hidden_dim),  # 批归一化层
                h_swish() if use_hs else nn.ReLU(inplace=True),  # 使用Hard Swish或ReLU激活函数
                SELayer(hidden_dim) if use_se else nn.Sequential(),  # 使用SELayer或空的Sequential
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  # 1x1卷积层
                nn.BatchNorm2d(oup)  # 批归一化层
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),  # 1x1卷积层，用于通道扩张
                nn.BatchNorm2d(hidden_dim),  # 批归一化层
                h_swish() if use_hs else nn.ReLU(inplace=True),  # 使用Hard Swish或ReLU激活函数
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),  # 深度可分离卷积层
                nn.BatchNorm2d(hidden_dim),  # 批归一化层
                SELayer(hidden_dim) if use_se else nn.Sequential(),  # 使用SELayer或空的Sequential
                h_swish() if use_hs else nn.ReLU(inplace=True),  # 使用Hard Swish或ReLU激活函数
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  # 1x1卷积层
                nn.BatchNorm2d(oup)  # 批归一化层
            )

    def forward(self, x):
        y = self.conv(x)  # 通过卷积层
        if self.identity:
            return x + y  # 恒等映射
        else:
            return y  # 非恒等映射


if __name__ == '__main__':
    from thop import profile  ## 导入thop模块

    model = MobileNetV3(16,16,16,3,2,1,0)
    input = torch.randn(1, 16, 640, 640)
    flops, params = profile(model, inputs=(input,))
    outpus = model.forward(input)
    print('flops', flops)  ## 打印计算量
    print('params', params)  ## 打印参数量