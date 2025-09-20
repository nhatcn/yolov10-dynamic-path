import torch
import torch.nn as nn

class ComplexityPredictor(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1//4),
            nn.ReLU(inplace=True),  # inplace=True để tiết kiệm memory
            nn.Linear(c1//4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Loại bỏ các check không cần thiết, chỉ giữ check cơ bản
        if x.dim() != 4:
            return torch.tensor(0.5, device=x.device, dtype=x.dtype)
            
        b, c, h, w = x.shape
        x = self.pool(x).view(b, c)
        complexity = self.fc(x).squeeze(-1)  # squeeze(-1) thay vì squeeze()
        return complexity

class DynamicPath(nn.Module):
    def __init__(self, c1, c2=None, k=3, s=2):  # Thêm k, s như Conv module
        super().__init__()
        if c2 is None:
            c2 = c1 * 2  # Default cho downsampling
            
        # Simple path - hiệu quả
        self.simple = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k//2, bias=False),  # Dùng k, s từ arguments
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        
        # Complex path - cải thiện feature extraction  
        self.complex = nn.Sequential(
            nn.Conv2d(c1, c1, k, 1, k//2, bias=False),  # Dùng k từ arguments
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, k, 1, k//2, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, k, s, k//2, bias=False),  # Dùng k, s từ arguments
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        
        self.predictor = ComplexityPredictor(c1)
        # Learnable threshold thay vì hard-coded 0.5
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # Đơn giản hóa input validation
        if x.dim() != 4:
            return self.simple(x) if hasattr(self, 'simple') else x
        
        complexity = self.predictor(x)
        
        if self.training:
            # Training: Progressive hard selection thay vì soft mixing
            simple_out = self.simple(x)
            complex_out = self.complex(x)
            
            # Hard selection với temperature annealing
            comp_val = complexity.mean() if complexity.dim() > 0 else complexity
            
            # Progressive threshold: start với mostly simple, gradually increase complex usage
            if comp_val > (self.threshold + 0.2):  # Bias toward simple path initially
                return complex_out
            else:
                return simple_out
        else:
            # Inference: Hard selection với learnable threshold
            comp_val = complexity.mean() if complexity.dim() > 0 else complexity
            
            if comp_val > self.threshold:
                return self.complex(x)
            else:
                return self.simple(x)