import torch
import torch.nn as nn
from ultralytics.utils.ops import make_divisible

# AMSA (unchanged)
class AMSA(nn.Module):
    def __init__(self, c, scales=None):
        super().__init__()
        if scales is None or not isinstance(scales, (list, tuple)):
            scales = [3, 5, 7]
        self.scales = scales
        self.num_scales = len(scales)
        self.multi_convs = nn.ModuleList([
            nn.Conv2d(c, c, kernel_size=k, padding=k // 2, groups=1)
            for k in scales
        ])
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * self.num_scales, c, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Conv2d(c, 1, 7, padding=3)
        self.weight_conv = nn.Conv2d(c, self.num_scales, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        multi_features = [conv(x) for conv in self.multi_convs]
        concat_features = torch.cat(multi_features, dim=1)
        weights = self.weight_conv(x)
        weights = self.sigmoid(weights.mean(dim=(2, 3), keepdim=True))
        weighted_features = [feature * weights[:, i:i+1, :, :] for i, feature in enumerate(multi_features)]
        fused_features = sum(weighted_features)
        ca = self.channel_att(concat_features)
        fused_features = fused_features * ca
        sa = self.sigmoid(self.spatial_att(fused_features))
        output = fused_features * sa
        return output

# Merged DSFB with inline MultiScaleFusion logic
class DSFB(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.n = n
        self.shortcut = shortcut and c1 == c2  # Only shortcut if dimensions match
        c2 = c2 if c2 is not None else c1  # Default to c1 if c2 not specified
        self.c_ = make_divisible(max(int(c2 * e), 8), 8)  # Hidden channels
        self.cv1 = nn.Conv2d(c1, 2 * self.c_, 1, 1)
        self.cv2 = nn.Conv2d((self.n + 1) * self.c_, c2, 1, 1)
        
        # Inline MultiScaleFusion init: Create scales and gate for each repeat
        self.scales_list = nn.ModuleList()
        self.gate_list = nn.ModuleList()
        for _ in range(n):
            branch_channels = max(self.c_ // 3, 1)
            scales = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(branch_channels, branch_channels, k, 1, k//2, groups=branch_channels),
                    nn.Conv2d(branch_channels, branch_channels, 1)
                ) for k in [3, 5, 7]
            ])
            gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.c_, 3, 1),
                nn.Softmax(dim=1)
            )
            self.scales_list.append(scales)
            self.gate_list.append(gate)

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = y.chunk(2, dim=1)  # Split into two parts
        ys = [y1]  # Start with first part
        
        last_output = y1  # Initialize with first part for sequential processing
        for i in range(self.n):
            # Inline MultiScaleFusion forward
            branch_channels = last_output.shape[1] // 3
            y_splits = torch.split(last_output, branch_channels, dim=1)
            branches = [scale(split) for scale, split in zip(self.scales_list[i], y_splits)]
            fused = torch.cat(branches, dim=1)
            weights = self.gate_list[i](fused)
            weighted_branches = [branch * weight for branch, weight in zip(branches, torch.split(weights, 1, dim=1))]
            final_fused = torch.cat(weighted_branches, dim=1)
            if self.shortcut:
                final_fused = final_fused + last_output
            ys.append(final_fused)
            last_output = final_fused  # Update for next iteration
        
        y = self.cv2(torch.cat(ys, dim=1))
        return x + y if self.shortcut else y