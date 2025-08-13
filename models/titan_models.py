# models/titan_models.py

import torch
import torch.nn as nn

from models.memory_module import NeuralMemory

class TitansMAC(nn.Module):
    """Memory as Context (MAC) architecture - simplified version, no chunking."""

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

        # Gate for combining attention + memory readout (Equation 25)
        self.gate_proj = nn.Linear(2 * self.d_model, self.d_model)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, -0.5)

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

        # Process in segments (simplified - just using full sequence for now)
        segment = x  # TODO: would need to add the chunking logic for segments to match paper

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

        # Update memory with attention output
        self.neural_memory.forward(attn_out)

        # Final memory readout (Equation 25)
        queries_final = self.neural_memory.W_Q(attn_out)
        final_memory_readout = self.neural_memory.retrieve_memory(queries_final)

        # # Combine attention + memory (⊗ operation)
        # combined_output = attn_out + final_memory_readout

        # Combine attention + memory with learnable gating
        gate_input = torch.cat([attn_out, final_memory_readout], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        combined_output = gate * attn_out + (1 - gate) * final_memory_readout

        # Extract output for original sequence length
        output = combined_output[:, -seq_len:]

        # Layer norm and projection
        output = self.ln(output)

        # Task-specific output heads
        if task_type == "needle_haystack":
            return self.position_proj(output)
        else:
            return self.output_proj(output)



class TitansMAG(nn.Module):
    """Memory as Gate (MAG) architecture - simplified version, no sliding window"""

    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.vocab_size = config['model']['vocab_size']
        self.max_seq_len = config['model']['max_seq_len']
        self.window_size = config['model']['memory']['window_size']

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(
            self.max_seq_len + config['model']['persistent_memory_size'],
            self.d_model
        )

        # Neural long-term memory
        self.neural_memory = NeuralMemory(
            self.d_model,
            config['model']['memory_layers']
        )

        # Persistent memory (learnable task-independent parameters)
        self.persistent_memory = nn.Parameter(
            torch.randn(config['model']['persistent_memory_size'], self.d_model) * 0.02
        )

        # Sliding window attention (simplified as regular attention for now)
        self.attention = nn.MultiheadAttention(
            self.d_model,
            config['model']['n_heads'],
            batch_first=True
        )

        # Gate for combining attention + memory branches (Equation 28)
        self.gate_proj = nn.Linear(2 * self.d_model, self.d_model)

        # Initialize gate to initially favor attention slightly
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, -0.5)  # sigmoid(-0.5) ≈ 0.38

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

        # Add persistent memory to start (Equation 26)
        persistent_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        x_with_persistent = torch.cat([persistent_expanded, x], dim=1)

        # Branch 1: Sliding window attention (Equation 27) # TODO: would need to add the sliding-window attention to match paper
        # SW-Attn*(x̃) - simplified as regular attention
        attn_out, _ = self.attention(x_with_persistent, x_with_persistent, x_with_persistent)

        # Extract attention output for original sequence (remove persistent memory portion)
        attention_branch = attn_out[:, self.persistent_memory.size(0):, :]

        # Branch 2: Neural memory module (Equation 28)
        # M(x̃) - process through memory
        memory_branch = self.neural_memory(x_with_persistent, update_memory=True)

        # Extract memory output for original sequence (remove persistent memory portion)
        memory_branch = memory_branch[:, self.persistent_memory.size(0):, :]

        # Ensure both branches have correct sequence length
        if attention_branch.size(1) != seq_len:
            attention_branch = attention_branch[:, :seq_len, :]
        if memory_branch.size(1) != seq_len:
            memory_branch = memory_branch[:, :seq_len, :]

        # Combine branches with learnable gating (Equation 28)
        gate_input = torch.cat([attention_branch, memory_branch], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        combined_output = gate * attention_branch + (1 - gate) * memory_branch

        # Layer norm and projection
        output = self.ln(combined_output)

        # Task-specific output heads
        if task_type == "needle_haystack":
            return self.position_proj(output)
        else:
            return self.output_proj(output)



class TitansMAL(nn.Module):
    """Memory as Layer (MAL) architecture - simplified version, no sliding window"""

    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.vocab_size = config['model']['vocab_size']
        self.max_seq_len = config['model']['max_seq_len']

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(
            self.max_seq_len + config['model']['persistent_memory_size'],
            self.d_model
        )

        # Neural long-term memory (processes first)
        self.neural_memory = NeuralMemory(
            self.d_model,
            config['model']['memory_layers']
        )

        # Persistent memory
        self.persistent_memory = nn.Parameter(
            torch.randn(config['model']['persistent_memory_size'], self.d_model) * 0.02
        )

        # Sliding window attention - just using standard MHSA here   # TODO: would need to add the sliding-window attention to match paper
        self.attention = nn.MultiheadAttention(
            self.d_model,
            config['model']['n_heads'],
            batch_first=True
        )

        # Layer norms
        self.ln_memory = nn.LayerNorm(self.d_model)
        self.ln_final = nn.LayerNorm(self.d_model)

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

        # Add persistent memory to start (Equation 29)
        persistent_expanded = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        x_with_persistent = torch.cat([persistent_expanded, x], dim=1)

        # Step 1: Memory layer processes input (Equation 30)
        memory_output = self.neural_memory(x_with_persistent, update_memory=True)
        memory_output = self.ln_memory(memory_output)

        # Step 2: Sliding window attention processes memory output (Equation 31)
        attention_output, _ = self.attention(memory_output, memory_output, memory_output)

        # Extract output for original sequence (remove persistent memory tokens)
        final_output = attention_output[:, self.persistent_memory.size(0):, :]

        # Ensure correct sequence length
        if final_output.size(1) != seq_len:
            final_output = final_output[:, :seq_len, :]

        # Final layer norm and projection
        output = self.ln_final(final_output)

        if task_type == "needle_haystack":
            return self.position_proj(output)
        else:
            return self.output_proj(output)


class TitansLMM(nn.Module):
    """Long-term Memory Module only (no attention)"""

    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.vocab_size = config['model']['vocab_size']
        self.max_seq_len = config['model']['max_seq_len']

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)

        # Neural memory module
        self.neural_memory = NeuralMemory(
            self.d_model,
            config['model']['memory_layers']
        )

        # Layer norm and output
        self.ln = nn.LayerNorm(self.d_model)

        if config['data']['dataset'] == "needle_haystack":
            self.position_proj = nn.Linear(self.d_model, self.max_seq_len)
        else:
            self.output_proj = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x, task_type="language_modelling"):
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)

        # Process through neural memory only
        memory_output = self.neural_memory(x, update_memory=True)

        # Layer norm and projection
        output = self.ln(memory_output)

        if task_type == "needle_haystack":
            return self.position_proj(output)
        else:
            return self.output_proj(output)