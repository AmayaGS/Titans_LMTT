# utils/data.py
import torch
import numpy as np
import logging


def generate_copy_task_data(config):
    """
    Generate copy task dataset: [1,2,3,4,0,?,?,?,?] -> [1,2,3,4,0,1,2,3,4]
    The model should learn to copy the sequence after seeing the delimiter (0).
    """
    num_samples = config['data']['num_samples']
    seq_len = config['data']['seq_len']
    vocab_size = config['data']['vocab_size']

    # Copy length is half the sequence (minus delimiter)
    copy_len = (seq_len - 1) // 2

    data = []
    for _ in range(num_samples):
        # Generate random sequence to copy (avoid using 0, that's our delimiter)
        sequence_to_copy = torch.randint(1, vocab_size, (copy_len,))

        delimiter = torch.tensor([0], dtype=torch.long)

        # Create prompt
        copy_prompt = torch.zeros(copy_len, dtype=torch.long)

        # Combine: [sequence, delimiter, prompt]
        input_seq = torch.cat([sequence_to_copy, delimiter, copy_prompt])

        # Target: [sequence, delimiter, sequence_to_copy]
        target_seq = torch.cat([sequence_to_copy, delimiter, sequence_to_copy])

        data.append((input_seq, target_seq))

    logging.info(f"Generated {num_samples} copy task samples")
    logging.info(f"Sample input length: {len(data[0][0])}")
    logging.info(f"Copy length: {copy_len}")

    return data


def generate_simple_lm_data(config):
    """Generate simple language modeling data with repeating patterns."""
    num_samples = config['data']['num_samples']
    seq_len = config['data']['seq_len']
    vocab_size = config['data']['vocab_size']

    data = []
    for _ in range(num_samples):
        # Create simple repeating pattern: [1,2,3,1,2,3,1,2,3...]
        pattern_len = np.random.randint(2, 5)  # Pattern length 2-4
        pattern = torch.randint(1, vocab_size, (pattern_len,), dtype=torch.long)

        # Repeat pattern to fill sequence
        full_seq = pattern.repeat((seq_len // pattern_len) + 1)[:seq_len]

        # Input is sequence, target is sequence shifted by 1
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]

        data.append((input_seq, target_seq))

    logging.info(f"Generated {num_samples} simple LM samples")
    return data


def generate_needle_haystack_data(config):
    """Find a special token in a long sequence - tests memory retrieval."""
    num_samples = config['data']['num_samples']
    seq_len = config['data']['seq_len']
    vocab_size = config['data']['vocab_size']

    data = []
    needle_token = vocab_size - 1  # Special token

    for _ in range(num_samples):
        # Random sequence (haystack)
        sequence = torch.randint(1, vocab_size - 1, (seq_len - 2,), dtype=torch.long)

        # Insert needle at random position (avoid edges)
        needle_pos = torch.randint(5, seq_len - 10, (1,)).item()
        sequence[needle_pos] = needle_token

        # Query: [sequence, needle_token] -> model should output position
        input_seq = torch.cat([sequence, torch.tensor([needle_token], dtype=torch.long)])
        target = torch.tensor([needle_pos], dtype=torch.long)

        data.append((input_seq, target))

    logging.info(f"Generated {num_samples} needle-in-haystack samples")
    logging.info(f"Needle token: {needle_token}, sequence length: {seq_len}")

    return data


def split_dataset(dataset, train_ratio):
    """Simple train/test split."""
    n_train = int(len(dataset) * train_ratio)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]
    return train_data, test_data


def load_dataset(config):
    """Load and split dataset."""
    dataset_name = config['data']['dataset']

    if dataset_name == 'copy_task':
        full_dataset = generate_copy_task_data(config)
    elif dataset_name == 'language_modeling':
        full_dataset = generate_simple_lm_data(config)
    elif dataset_name == 'needle_haystack':
        full_dataset = generate_needle_haystack_data(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split using config
    train_data, test_data = split_dataset(full_dataset, config['data']['train_ratio'])

    logging.info(f"Dataset split: {len(train_data)} train, {len(test_data)} test")
    return train_data, test_data


def create_dataloader(dataset, batch_size, shuffle=True):
    """Create PyTorch DataLoader from dataset."""
    from torch.utils.data import DataLoader, TensorDataset

    # Separate inputs and targets
    inputs = torch.stack([item[0] for item in dataset])
    targets = torch.stack([item[1] for item in dataset])

    # Create TensorDataset and DataLoader
    tensor_dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader