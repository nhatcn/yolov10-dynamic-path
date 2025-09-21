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
    """Dynamic Scale Fusion Block - Completely fixed version for YOLOv10"""
    def __init__(self, c1, c2=None, n=1, shortcut=True, e=0.5):
        super().__init__()
        c2 = c2 or c1
        self.shortcut = shortcut and c1 == c2
        
        # Use n as shortcut parameter if it's boolean
        if isinstance(n, bool):
            self.shortcut = n and c1 == c2
            n = 1  # Default to 1 block
        
        # Calculate hidden channels safely
        # Ensure c_hidden is divisible by 24 (so each branch gets 8 channels minimum)
        c_hidden = max(int(c1 * e), 24)
        c_hidden = ((c_hidden + 23) // 24) * 24  # Round up to nearest multiple of 24
        
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
    """Multi-Scale Block - Completely fixed channel handling"""
    def __init__(self, c):
        super().__init__()
        
        # Ensure c is divisible by 3 first, then by 8 for each branch
        c = max(c, 24)  # Minimum 24 channels (8*3)
        c = ((c + 23) // 24) * 24  # Make divisible by 24
        self.c = c
        
        # Split into 3 equal branches, each divisible by 8
        c_branch = c // 3
        c_branch = max(c_branch, 8)  # Minimum 8 channels per branch
        c_branch = ((c_branch + 7) // 8) * 8  # Make divisible by 8
        self.c_branch = c_branch
        
        # Calculate groups properly
        def safe_groups(channels, max_groups=8):
            """Calculate safe number of groups that divides channels evenly"""
            for groups in [max_groups, 4, 2, 1]:
                if channels % groups == 0:
                    return groups
            return 1
        
        groups1 = safe_groups(c_branch)
        groups2 = safe_groups(c_branch) 
        groups3 = safe_groups(c_branch)
        
        # Three scale branches with guaranteed valid groups
        self.branch1 = nn.Sequential(
            nn.Conv2d(c_branch, c_branch, 3, 1, 1, groups=groups1),
            nn.Conv2d(c_branch, c_branch, 1),
            nn.BatchNorm2d(c_branch),
            nn.SiLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(c_branch, c_branch, 5, 1, 2, groups=groups2),  
            nn.Conv2d(c_branch, c_branch, 1),
            nn.BatchNorm2d(c_branch),
            nn.SiLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(c_branch, c_branch, 7, 1, 3, groups=groups3),
            nn.Conv2d(c_branch, c_branch, 1), 
            nn.BatchNorm2d(c_branch),
            nn.SiLU(inplace=True)
        )
        
        # Input channel adjustment
        actual_input_needed = 3 * c_branch
        self.input_adjust = nn.Conv2d(c, actual_input_needed, 1) if c != actual_input_needed else None
        
        # Output channel adjustment  
        actual_output_channels = 3 * c_branch
        self.output_adjust = nn.Conv2d(actual_output_channels, c, 1) if actual_output_channels != c else None
        
        # Adaptive weights
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(actual_output_channels, 3, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Adjust input channels if needed
        if self.input_adjust is not None:
            x = self.input_adjust(x)
            
        # Split into branches
        x1, x2, x3 = torch.split(x, self.c_branch, dim=1)
        
        # Process each branch
        out1 = self.branch1(x1)
        out2 = self.branch2(x2) 
        out3 = self.branch3(x3)
        
        # Concatenate outputs
        concat_out = torch.cat([out1, out2, out3], dim=1)
        
        # Generate adaptive weights
        weights = self.weight_gen(concat_out)
        w1, w2, w3 = torch.split(weights, 1, dim=1)
        
        # Weighted fusion (optional - using concat for simplicity)
        # fused = out1 * w1 + out2 * w2 + out3 * w3
        
        # Adjust output channels if needed
        if self.output_adjust is not None:
            output = self.output_adjust(concat_out)
        else:
            output = concat_out
        
        return output