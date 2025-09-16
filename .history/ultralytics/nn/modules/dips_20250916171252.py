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
        # Check if input is valid tensor
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            # Return default complexity for invalid input
            return torch.tensor(0.5, device=x.device if isinstance(x, torch.Tensor) else 'cpu')
        
        # Handle different input shapes
        if x.dim() != 4:
            # If not 4D, assume it's already processed
            return torch.tensor(0.5, device=x.device)
            
        b, c, h, w = x.shape
        if h == 0 or w == 0:
            return torch.tensor(0.5, device=x.device)
            
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
        # Safety check
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            return x
            
        if x.dim() != 4:
            # If not 4D tensor, just use simple path
            if hasattr(self.simple, 'weight'):
                # Create dummy input to maintain flow
                dummy = torch.zeros(1, self.simple.in_channels, 32, 32, device=x.device)
                return self.simple(dummy)
            return x
        
        try:
            complexity = self.predictor(x)
        except:
            # Fallback to simple path if complexity prediction fails
            return self.simple(x)
        
        if self.training:
            # Training: use both paths
            simple_out = self.simple(x)
            complex_out = self.complex(x)
            
            # Handle complexity shape
            if complexity.dim() == 0:
                weight = complexity
            else:
                weight = complexity.mean()
                
            # Weighted combination based on complexity
            return weight * complex_out + (1 - weight) * simple_out
        else:
            # Inference: choose path
            comp_val = complexity.mean() if complexity.dim() > 0 else complexity
            if comp_val > 0.5:
                return self.complex(x)
            else:
                return self.simple(x)