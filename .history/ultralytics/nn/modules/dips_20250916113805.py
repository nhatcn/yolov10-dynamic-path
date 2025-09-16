import torch
import torch.nn as nn

class ComplexityPredictor(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1//4),
            nn.ReLU(),
            nn.Linear(c1//4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pool(x).view(b, c)
        return self.fc(x).squeeze()

class DynamicPath(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # Simple path
        self.simple = nn.Conv2d(c1, c2, 3, 2, 1)
        # Complex path  
        self.complex = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.Conv2d(c1, c2, 3, 2, 1)
        )
        self.predictor = ComplexityPredictor(c1)
        
    def forward(self, x):
        complexity = self.predictor(x)
        
        if self.training:
            # Training: use both paths
            simple_out = self.simple(x)
            complex_out = self.complex(x)
            # Weighted combination based on complexity
            return complexity.unsqueeze(-1).unsqueeze(-1) * complex_out + \
                   (1-complexity.unsqueeze(-1).unsqueeze(-1)) * simple_out
        else:
            # Inference: choose path
            if complexity.mean() > 0.5:
                return self.complex(x)
            else:
                return self.simple(x)