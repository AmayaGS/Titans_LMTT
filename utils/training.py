# utils/training.py

import torch
import torch.nn as nn
import logging


def compute_main_task_loss(logits, targets, task_type="copy_task"):
    """Main task loss - outer loop is standard cross-entropy loss"""

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
    """Compute evaluation metrics for the copy-task, language modelling, needle in a haystack"""

    if task_type == "copy_task": # probably could reuse compute_main_task_loss
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

        # outer loop loss
        loss = compute_main_task_loss(logits, targets, task_type)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping here - to avoid exploding gradients, stabilise learning
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['gradient_clip']
        )

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, config):
    """Evaluate models. No grad for baseline, with grad for test-time learning with Titans."""
    model.eval()  # Set to eval mode for other components
    total_loss = 0
    all_metrics = []

    # For Titans - we need grad here - memory needs gradients
    is_titans = config['model']['variant'] in ['MAC', 'MAG', 'MAL', 'LMM']

    if is_titans:
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs, task_type=config['data']['dataset'])

            loss = compute_main_task_loss(logits, targets, config['data']['dataset'])
            metrics = compute_task_metrics(logits, targets, loss, config['data']['dataset'])

            total_loss += loss.item()
            all_metrics.append(metrics)
    else:
        # Baseline - no_grad here as no test time learning
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
        logging.info(f"Using AdamW optimizer with lr={config['training']['learning_rate']}, "
                     f"weight_decay={config['training']['weight_decay']}, batch size: {config['training']['batch_size']}")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer