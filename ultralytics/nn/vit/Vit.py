import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Union, Sequence
from .mobilevit_v2_block import MobileViTBlockv2 as MbViTBkV2

class MbViTV3(MbViTBkV2):
    def __init__(
            self,
            in_channels: int,
            attn_unit_dim: int,
            patch_h: Optional[int] = 2,
            patch_w: Optional[int] = 2,
            ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
            n_attn_blocks: Optional[int] = 2,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            conv_ksize: Optional[int] = 3,
            attn_norm_layer: Optional[str] = "layer_norm_2d",
            enable_coreml_compatible_fn: Optional[bool] = False,
    ) -> None:
        super(MbViTV3, self).__init__(in_channels, attn_unit_dim)
        self.enable_coreml_compatible_fn = enable_coreml_compatible_fn
        if self.enable_coreml_compatible_fn:
            # we set persistent to false so that these weights are not part of model's state_dict
            self.register_buffer(
                name="unfolding_weights",
                tensor=self._compute_unfolding_weights(),
                persistent=False,
            )
        cnn_out_dim = attn_unit_dim
        self.conv_proj = nn.Conv2d(2 * cnn_out_dim, in_channels, 1, 1)

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)
        fm_conv1 = self.local_rep(x)
        fm_conv2 = self.local_rep(x)
        # convert feature map to patches
        patches1, output_size1 = self.unfolding_pytorch(fm_conv1)
        patches2, output_size2 = self.unfolding_pytorch(fm_conv2)
        # learn global representations on all patches
        patches1 = self.global_rep(patches1)
        patches2 = self.global_rep(patches2)
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm1 = self.folding_pytorch(patches=patches1, output_size=output_size1)
        fm2 = self.folding_pytorch(patches=patches2, output_size=output_size2)
        # MobileViTv3: local+global instead of only global
        fm1 = self.conv_proj(torch.cat((fm1, fm_conv1), dim=1))
        fm2 = self.conv_proj(torch.cat((fm2, fm_conv2), dim=1))
        # MobileViTv3: skip connection
        fm = fm1 + x + fm2

        return fm


if __name__ == '__main__':
    from thop import profile  ## 导入thop模块

    model = MbViTV3(1024, 1024, enable_coreml_compatible_fn=False)
    input = torch.randn(1, 1024, 640, 640)
    flops, params = profile(model, inputs=(input,))
    outpus = model.forward_spatial(input)
    print('flops', flops)  ## 打印计算量
    print('params', params)  ## 打印参数量