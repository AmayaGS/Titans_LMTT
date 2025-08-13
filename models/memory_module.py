# models/memory_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemory(nn.Module):
    """Neural memory module for Titans"""

    def __init__(self, d_model, memory_layers=2, init_scale=0.02):
        super().__init__()
        self.d_model = d_model
        self.memory_layers = memory_layers

        # Key, Value, Query projections (Equation 11)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)

        # Memory MLP - 2 layers with SiLU activation apparently
        layers = []
        for i in range(memory_layers):
            layers.append(nn.Linear(d_model, d_model))
            if i < memory_layers - 1:
                layers.append(nn.SiLU())
        self.memory_network = nn.Sequential(*layers)

        # Initialize memory network with small weights
        for module in self.memory_network:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_scale)
                nn.init.zeros_(module.bias)

        # Data-dependent parameters for momentum + forgetting
        self.alpha_proj = nn.Linear(d_model, 1)  # Forget gate α_t
        self.theta_proj = nn.Linear(d_model, 1)  # Momentary surprise θ_t
        self.eta_proj = nn.Linear(d_model, 1)    # Past surprise decay η_t

        # Initialize these to output small, reasonable values
        for proj in [self.alpha_proj, self.theta_proj, self.eta_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=0.01)
            nn.init.constant_(proj.bias, -2.0)  # sigmoid(-2) approx 0.12

        # Track surprise momentum S_t with no weight updates
        self.register_buffer('surprise_momentum', None)

    def compute_associative_loss(self, keys, values):
        """
        Compute associative memory loss: ℓ(M; x_t) = ||M(k_t) - v_t||²
        This is the "inner loss" from Equation 12 in the paper
        """
        # Forward pass through memory network
        memory_output = self.memory_network(keys)  # M(k_t)

        # L2 loss (no reduction to match paper: ||M(k_t) - v_t||²)
        loss = F.mse_loss(memory_output, values, reduction='sum')

        return loss

    def update_memory_with_momentum_and_forgetting(self, keys, values, x_t):
        """
        Update memory using momentum and forgetting mechanism.
        Equations 9-10 and 13-14 from paper.
        """
        # Compute data-dependent parameters
        alpha_t = torch.sigmoid(self.alpha_proj(x_t.mean(dim=1)))  # [0, 1]  # using x_t mean is likely a simplification
        theta_t = torch.sigmoid(self.theta_proj(x_t.mean(dim=1)))  # [0, 1]
        eta_t = torch.sigmoid(self.eta_proj(x_t.mean(dim=1)))  # [0, 1]

        # Compute associative memory loss and gradients
        loss = self.compute_associative_loss(keys, values)

        if torch.isnan(loss):
            return

        gradients = torch.autograd.grad(loss, self.memory_network.parameters(),
                                        create_graph=True, retain_graph=True)

        # Flatten gradients for momentum computation
        grad_flat = torch.cat([g.flatten() for g in gradients])

        # Normalize gradients to help prevent gradient explosion
        grad_norm = grad_flat.norm()
        if grad_norm > 1.0:
            grad_flat = grad_flat / grad_norm

        # Initialize surprise momentum if first time
        if self.surprise_momentum is None:
            self.surprise_momentum = torch.zeros_like(grad_flat)

        # Update surprise momentum: S_t = η_t S_{t-1} - θ_t ∇ℓ (Equation 10)
        self.surprise_momentum = (eta_t.mean() * self.surprise_momentum -    # likewise, using the mean here is a likely a simplfication
                                  theta_t.mean() * grad_flat)

        # Update memory parameters: M_t = (1 - α_t)M_{t-1} + S_t (Equation 13)
        param_idx = 0
        with torch.no_grad():
            for param in self.memory_network.parameters():
                param_size = param.numel()
                param_update = self.surprise_momentum[param_idx:param_idx + param_size]
                param_update = param_update.view(param.shape)

                # Apply forgetting + momentum update
                param.data = (1 - alpha_t.mean()) * param.data + param_update

                param_idx += param_size

                if torch.isnan(param).any():
                    print("NaN in parameter after momentum update!")
                    return

    def retrieve_memory(self, queries):
        """
        Retrieve from memory: M*(q_t) - forward pass WITHOUT weight updates
        Equation 15 from paper
        """
        with torch.no_grad():  # No gradients for retrieval
            retrieved = self.memory_network(queries)
        return retrieved

    def forward(self, x, update_memory=True):
        """
        Process sequence and optionally update memory.
        update_memory=True for training/test-time learning
        update_memory=False for pure retrieval
        """
        # Project to keys and values (Equation 11)
        keys = self.W_K(x)  # k_t = x_t W_K
        values = self.W_V(x)  # v_t = x_t W_V

        if update_memory:
            # Update memory using surprise mechanism
            self.update_memory_with_momentum_and_forgetting(keys, values, x)

        # Return the current memory output (for use in architecture)
        return self.memory_network(keys)