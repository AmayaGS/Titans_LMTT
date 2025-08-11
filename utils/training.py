# utils/training.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_associative_memory_loss(memory_output, target_values):
    """
    Associative memory loss from Eq. 12 in paper.
    L(M; x_t) = ||M(k_t) - v_t||^2
    """
    return F.mse_loss(memory_output, target_values, reduction='sum') # TODO: check reduction type


def compute_main_task_loss(logits, targets, task_type="copy_task"):
    """Main task loss (still cross-entropy for token prediction)."""
    if task_type == "copy_task":
        # For copy task, only compute loss on the "copy" portion
        # Assuming format: [sequence, delimiter, copy_target]
        # We want loss only on the copy_target part
        batch_size, seq_len, vocab_size = logits.shape

        # Find delimiter position (token 0)
        delimiter_pos = (targets == 0).nonzero(as_tuple=True)[1]

        # Compute loss only after delimiter
        copy_logits = logits[:, delimiter_pos[0] + 1:]
        copy_targets = targets[:, delimiter_pos[0] + 1:]

        return nn.CrossEntropyLoss()(
            copy_logits.contiguous().view(-1, vocab_size),
            copy_targets.contiguous().view(-1)
        )

    elif task_type == "language_modeling":
        # Standard next-token prediction
        return nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    task_type = config['data']['dataset']

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        logits = model(inputs)

        # Task-specific loss
        loss = compute_main_task_loss(logits, targets, task_type)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping - this is because it's a sequence model and we want to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['gradient_clip']
        )

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_task_metrics(logits, targets, loss, task_type="copy_task"):
    """Compute evaluation metrics (not used for training)."""
    if task_type == "copy_task":
        batch_size, seq_len, vocab_size = logits.shape
        delimiter_pos = (targets == 0).nonzero(as_tuple=True)[1]

        copy_logits = logits[:, delimiter_pos[0] + 1:]
        copy_targets = targets[:, delimiter_pos[0] + 1:]

        predictions = copy_logits.argmax(dim=-1)
        accuracy = (predictions == copy_targets).float().mean()

        return {"copy_accuracy": accuracy.item()}

    elif task_type == "language_modeling":
        return {"perplexity": torch.exp(loss).item()}


def evaluate(model, dataloader, device, config):
    """Evaluate with both loss and metrics."""
    model.eval()
    total_loss = 0
    all_metrics = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)

            loss = compute_main_task_loss(logits, targets, config['data']['dataset'])
            metrics = compute_task_metrics(logits, targets, loss, config['data']['dataset'])  # Pass loss

            total_loss += loss.item()
            all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return total_loss / len(dataloader), avg_metrics


def create_optimizer(model, config):
    """Create optimizer based on config."""
    optimizer_name = config['training']['optimizer'].lower()

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer