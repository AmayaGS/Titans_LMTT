# utils/training.py
import torch
import torch.nn as nn


def compute_main_task_loss(logits, targets, task_type="copy_task"):
    """Main task loss (still cross-entropy for token prediction)."""

    # Check for invalid targets
    if targets.min() < 0 or targets.max() >= logits.shape[-1]:
        print(f"ERROR: Invalid target indices! Range should be [0, {logits.shape[-1]})")
        return torch.tensor(0.0, requires_grad=True)

    if task_type == "copy_task":
        # Assuming format: [sequence, delimiter, copy_target]
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

    elif task_type == "needle_haystack":
        # Position prediction - logits should be [batch, seq_len] for position classes
        query_logits = logits[:, -1, :logits.size(1)]  # [batch, seq_len]
        return nn.CrossEntropyLoss()(query_logits, targets.squeeze(-1))

    else:
        raise ValueError(f"Unknown task type: {task_type}")


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

    elif task_type == "needle_haystack":
        # Position accuracy
        query_logits = logits[:, -1, :logits.size(1)]
        predictions = query_logits.argmax(dim=-1)
        accuracy = (predictions == targets.squeeze(-1)).float().mean()
        return {"position_accuracy": accuracy.item()}

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
        logits = model(inputs, task_type=task_type)

        # Task-specific loss
        loss = compute_main_task_loss(logits, targets, task_type)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping - this is because we want to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['gradient_clip']
        )

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, config):
    """Evaluate with test-time learning for Titans."""
    model.eval()  # Set to eval mode for other components
    total_loss = 0
    all_metrics = []

    # For Titans - we need grad here - memory needs gradients
    is_titans = config['model']['variant'] in ['MAC', 'MAG', 'MAL', 'LMM']

    if is_titans:
        # Titans needs gradients for memory updates
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs, task_type=config['data']['dataset'])

            loss = compute_main_task_loss(logits, targets, config['data']['dataset'])
            metrics = compute_task_metrics(logits, targets, loss, config['data']['dataset'])

            total_loss += loss.item()
            all_metrics.append(metrics)
    else:
        # Baseline: use no_grad for efficiency
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                logits = model(inputs, task_type=config['data']['dataset'])

                loss = compute_main_task_loss(logits, targets, config['data']['dataset'])
                metrics = compute_task_metrics(logits, targets, loss, config['data']['dataset'])

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