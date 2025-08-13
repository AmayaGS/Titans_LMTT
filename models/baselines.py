# models/baseline.py

import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    """Simple transformer baseline for comparison with Titans"""

    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.vocab_size = config['model']['vocab_size']
        self.max_seq_len = config['model']['max_seq_len']

        # Token embedding + positional encoding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)

        # Simple attention layer
        self.attention = nn.MultiheadAttention(
            self.d_model,
            config['model']['n_heads'],
            batch_first=True
        )

        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.ReLU(),
            nn.Linear(4 * self.d_model, self.d_model)
        )

        # Output projections for different tasks
        if config['data']['dataset'] == "needle_haystack":
            # For position prediction, we need a different output head
            self.position_proj = nn.Linear(self.d_model, self.max_seq_len)
        else:
            self.output_proj = nn.Linear(self.d_model, self.vocab_size)

        # Layer norm
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x, task_type="copy_task"):
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)

        # Attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        if task_type == "needle_haystack":
            return self.position_proj(x)  # [batch_size, seq_len]
        else:
            return self.output_proj(x)  # vocab_size classes