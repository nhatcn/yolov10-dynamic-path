import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.ops import make_divisible

class AMSA(nn.Module):
    """Adaptive Multi-Scale Attention - Simplified and efficient"""
    def __init__(self, c, reduction=16):
        super().__init__()
        self.c = c
        
        # Multi-scale convolutions - reduced to 2 scales for efficiency
        self.conv3 = nn.Conv2d(c, c//2, 3, padding=1, groups=c//8)
        self.conv5 = nn.Conv2d(c, c//2, 5, padding=2, groups=c//8)
        
        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c//reduction, c, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(c, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # Fusion
        self.fusion = nn.Conv2d(c, c, 1)

    def forward(self, x):
        # Multi-scale features
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        multi_scale = torch.cat([f3, f5], dim=1)
        
        # Channel attention
        ca_weight = self.ca(multi_scale)
        multi_scale = multi_scale * ca_weight
        
        # Spatial attention  
        sa_weight = self.sa(multi_scale)
        multi_scale = multi_scale * sa_weight
        
        # Final fusion
        out = self.fusion(multi_scale)
        return x + out

class DSFB(nn.Module):
    """Dynamic Scale Fusion Block - Fixed version"""
    def __init__(self, c1, c2=None, n=1, shortcut=True, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.shortcut = shortcut and c1 == c2
        
        # Hidden channels
        c_hidden = make_divisible(int(c1 * e), 8)
        
        # Input projection
        self.cv1 = nn.Conv2d(c1, 2 * c_hidden, 1)
        
        # Multi-scale blocks
        self.blocks = nn.ModuleList()
        for i in range(n):
            self.blocks.append(MSBlock(c_hidden))
        
        # Output projection
        self.cv2 = nn.Conv2d((n + 1) * c_hidden, c2, 1)
        
    def forward(self, x):
        # Input split
        y = self.cv1(x)
        y1, y2 = y.chunk(2, dim=1)
        
        # Process through blocks
        outputs = [y1]
        current = y1
        
        for block in self.blocks:
            current = block(current)
            outputs.append(current)
        
        # Concatenate and project
        y = self.cv2(torch.cat(outputs, dim=1))
        
        return x + y if self.shortcut else y

class MSBlock(nn.Module):
    """Multi-Scale Block - Core component of DSFB"""
    def __init__(self, c):
        super().__init__()
        c_branch = max(c // 3, 8)  # Ensure minimum channels
        
        # Three scale branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(c_branch, c_branch, 3, 1, 1, groups=c_branch),
            nn.Conv2d(c_branch, c_branch, 1),
            nn.BatchNorm2d(c_branch),
            nn.SiLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(c_branch, c_branch, 5, 1, 2, groups=c_branch),  
            nn.Conv2d(c_branch, c_branch, 1),
            nn.BatchNorm2d(c_branch),
            nn.SiLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(c_branch, c_branch, 7, 1, 3, groups=c_branch),
            nn.Conv2d(c_branch, c_branch, 1), 
            nn.BatchNorm2d(c_branch),
            nn.SiLU(inplace=True)
        )
        
        # Adaptive weights
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # Adjust channels if needed
        self.c_branch = c_branch
        self.adjust_channels = c != 3 * c_branch
        if self.adjust_channels:
            self.channel_adjust = nn.Conv2d(c, 3 * c_branch, 1)
            self.output_adjust = nn.Conv2d(3 * c_branch, c, 1)
    
    def forward(self, x):
        if self.adjust_channels:
            x_adjusted = self.channel_adjust(x)
        else:
            x_adjusted = x
            
        # Split into branches
        x1, x2, x3 = torch.split(x_adjusted, self.c_branch, dim=1)
        
        # Process each branch
        out1 = self.branch1(x1)
        out2 = self.branch2(x2) 
        out3 = self.branch3(x3)
        
        # Generate adaptive weights
        weights = self.weight_gen(x_adjusted)
        w1, w2, w3 = torch.split(weights, 1, dim=1)
        
        # Weighted fusion
        fused = out1 * w1 + out2 * w2 + out3 * w3
        
        if self.adjust_channels:
            fused = self.output_adjust(torch.cat([out1, out2, out3], dim=1))
        else:
            fused = torch.cat([out1, out2, out3], dim=1)
        
        return fused