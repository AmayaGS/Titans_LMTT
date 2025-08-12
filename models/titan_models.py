import torch
import torch.nn as nn

from models.memory_module import NeuralMemory


class TitansMAC(nn.Module):
    """Memory as Context (MAC) architecture - simplified version."""

    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.vocab_size = config['model']['vocab_size']
        self.max_seq_len = config['model']['max_seq_len']
        self.segment_size = config['model']['memory']['segment_size']

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)

        # Neural long-term memory
        self.neural_memory = NeuralMemory(
            self.d_model,
            config['model']['memory_layers']
        )

        # Persistent memory (learnable task-independent parameters)
        self.persistent_memory = nn.Parameter(
            torch.randn(config['model']['persistent_memory_size'], self.d_model) * 0.02
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            self.d_model,
            config['model']['n_heads'],
            batch_first=True
        )

        # Layer norm and output
        self.ln = nn.LayerNorm(self.d_model)

        if config['data']['dataset'] == "needle_haystack":
            # For position prediction, we need a different output head
            self.position_proj = nn.Linear(self.d_model, self.max_seq_len)
        else:
            self.output_proj = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x, task_type="language_modelling"):
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)

        # Process in segments (simplified - just use full sequence for now)
        segment = x  # TODO: Add proper segmentation

        # Retrieve from long-term memory
        queries = self.neural_memory.W_Q(segment)
        retrieved_memory = self.neural_memory.retrieve_memory(queries)

        # Concatenate: persistent memory + retrieved memory + current segment
        persistent_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([
            persistent_expanded,
            retrieved_memory,
            segment
        ], dim=1)

        # Attention over combined sequence
        attn_out, _ = self.attention(combined, combined, combined)

        # Extract output for original sequence length
        output = attn_out[:, -seq_len:]  # Take the last seq_len tokens

        # Layer norm and projection
        output = self.ln(output)

        # Update memory with attention output
        self.neural_memory.forward(output)

        # Task-specific output heads
        if task_type == "needle_haystack":
            return self.position_proj(output)
        else:
            return self.output_proj(output)