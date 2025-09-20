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

# MultiScaleFusion (unchanged)
class MultiScaleFusion(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        branch_channels = max(c // 3, 1)
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(branch_channels, branch_channels, k, 1, k//2, groups=branch_channels),
                nn.Conv2d(branch_channels, branch_channels, 1)
            ) for k in [3, 5, 7]
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, y):
        branch_channels = y.shape[1] // 3
        y_splits = torch.split(y, branch_channels, dim=1)
        branches = [scale(split) for scale, split in zip(self.scales, y_splits)]
        fused = torch.cat(branches, dim=1)
        weights = self.gate(fused)
        weighted_branches = [branch * weight for branch, weight in zip(branches, torch.split(weights, 1, dim=1))]
        final_fused = torch.cat(weighted_branches, dim=1)
        return final_fused + y if self.shortcut else final_fused

# Fixed DSFB
class DSFB(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.n = n
        self.shortcut = shortcut and c1 == c2  # Only shortcut if dimensions match
        c2 = c2 if c2 is not None else c1  # Default to c1 if c2 not specified
        c_ = max(int(c2 * e), 8)  # Hidden channels, ensure min 8
        c_ = make_divisible(c_, 8)  # Ensure divisibility for efficiency
        
        # Fix: cv1 should split input into two parts for processing
        self.cv1 = nn.Conv2d(c1, 2 * c_, 1, 1)
        
        # Fix: Calculate correct input channels for cv2
        # We have: c_ (from first split) + n * c_ (from n MultiScaleFusion outputs) + c_ (from second split)
        total_channels = c_ + n * c_ + c_  # This equals (n + 2) * c_
        self.cv2 = nn.Conv2d(total_channels, c2, 1, 1)
        
        self.m = nn.ModuleList(MultiScaleFusion(c_, shortcut=True) for _ in range(n))

    def forward(self, x):
        # Split cv1 output into two parts
        y = self.cv1(x)
        y1, y2 = y.chunk(2, dim=1)  # Split into two equal parts
        
        # Start with the first part
        ys = [y1]
        
        # Apply MultiScaleFusion modules sequentially
        last_output = y2  # Start with second part
        for m in self.m:
            last_output = m(last_output)
            ys.append(last_output)
        
        # Add the second part at the end
        ys.append(y2)
        
        # Concatenate all outputs
        y = self.cv2(torch.cat(ys, dim=1))
        
        # Apply shortcut connection if dimensions match
        return x + y if self.shortcut else y