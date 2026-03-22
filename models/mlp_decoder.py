import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, vocab_size=300, max_len=10, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, max_len * vocab_size))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x, labels=None):
        # x: [batch, k, hidden_dim]
        thought_4 = x[:, -1, :] 
        
        logits = self.mlp(thought_4).view(-1, self.max_len, self.vocab_size)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=1)
        return type("Output", (), {"logits": logits, "loss": loss})
