import torch.nn as nn
import torch

class KEENProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, dropout=0.1):
        super().__init__()
        
        # Enhanced architecture with multiple layers and skip connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Skip connection from input to final layer
        self.skip = nn.Linear(input_dim, hidden_dim // 4)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Main path
        encoded = self.encoder(x)
        
        # Skip connection
        skip = self.skip(x)
        
        # Combine paths
        combined = encoded + skip
        
        # Final classification
        return self.classifier(combined)

