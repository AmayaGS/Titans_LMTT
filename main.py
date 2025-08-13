# main.py
import argparse
import logging
import random
import torch
import numpy as np
from pathlib import Path
import yaml

from utils.data import load_dataset, create_dataloader
from models.baselines import SimpleTransformer
from utils.training import train_epoch, evaluate, create_optimizer

from models.titan_models import TitansMAC, TitansMAG, TitansMAL, TitansLMM  # Import Titans models


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Titans Model')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup_logging(config):
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['log_level'])

    # Use the logs directory from config
    log_file = Path(config['paths']['log_dir']) / 'training.log'

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Now goes to ./logs/training.log
            logging.StreamHandler()
        ]
    )


def create_directories(config):
    """Create necessary directories for saving data, models, and logs."""
    for path_key in ['data_dir', 'model_dir', 'log_dir']:
        path = Path(config['paths'][path_key])
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {path}")


def create_model(config, device):
    """Create model based on config variant."""
    variant = config['model']['variant']

    if variant == 'baseline':
        model = SimpleTransformer(config)
    elif variant == 'LMM':
        model = TitansLMM(config)
    elif variant == 'MAC':
        model = TitansMAC(config)
    elif variant == 'MAG':
        model = TitansMAG(config)
    elif variant == 'MAL':
        model = TitansMAL(config)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Created {variant} model with {total_params:,} parameters")

    return model


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)

    set_seed(config['seed'])
    setup_logging(config)
    create_directories(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Model variant: {config['model']['variant']}")
    logging.info(f"Dataset: {config['data']['dataset']}")

    # Load dataset
    logging.info("Loading dataset...")
    train_data, test_data = load_dataset(config)  # NOTE: I'm only defining train/test because I won't be doing any hyperparameter tuning here
    train_loader = create_dataloader(train_data, config['training']['batch_size'])
    test_loader = create_dataloader(test_data, config['training']['batch_size'], shuffle=False)
    logging.info(f"Created train loader with {len(train_data)} samples")
    logging.info(f"Created test loader with {len(test_data)} samples")

    # Create model and optimizer
    model = create_model(config, device)
    optimizer = create_optimizer(model, config)

    # Training loop
    logging.info("Starting training...")
    for epoch in range(config['training']['max_epochs']):

        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        test_loss, test_metrics = evaluate(model, test_loader, device, config)

        # Log progress
        if epoch % config['logging']['log_every'] == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
            logging.info(f"Epoch {epoch:3d}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}, {metrics_str}")

    logging.info("Training complete!")

    logging.info("Training setup complete!")


if __name__ == "__main__":
    main()