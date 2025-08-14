# main.py

import argparse
import logging
import random
import torch
import numpy as np
from pathlib import Path
import yaml

from utils.data import load_dataset, create_dataloader
from utils.training import train_epoch, evaluate, create_optimizer
from utils.memory_check import check_memory_updates
from utils.model_comparison import run_all_variants_comparison, plot_comparison_results

from models.baselines import SimpleTransformer # Import baseline model
from models.titan_models import TitansMAC, TitansMAG, TitansMAL, TitansLMM  # Import Titans models


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Titans Model')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Set random seeds so it's reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """Setup logging configs"""
    log_level = getattr(logging, config['logging']['log_level'])

    # Ensure log directory exists before creating FileHandler
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use the logs directory from config
    log_file = Path(config['paths']['log_dir']) / f"training_{config['model']['variant']}_{config['data']['dataset']}.log" # would need to update this for different

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_directories(config):
    """Create directories for saving data and logs"""
    for path_key in ['data_dir', 'results_dir', 'log_dir']:
        path = Path(config['paths'][path_key])
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {path}")


def create_model(config, device):
    """Create model based on config variant"""
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
    logging.info(f"Created mini {variant} model with {total_params:,} parameters")

    return model


def main():
    """Main training function"""
    args = parse_args()
    config = load_config(args.config)

    set_seed(config['seed'])
    setup_logging(config)
    create_directories(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_comparison = config['training'].get('run_comparison', False)

    if run_comparison:
        # Run all variants and plot comparison
        results = run_all_variants_comparison(config, device)
        plot_comparison_results(results, config)
    else:

        logging.info(f"Using device: {device}")
        logging.info(f"Model variant: {config['model']['variant']}")
        logging.info(f"Dataset: {config['data']['dataset']}")

        # Load dataset
        logging.info("Loading dataset...")
        train_data, test_data = load_dataset(config)  # I'm only defining train/test because I won't be doing any hyperparameter tuning here
        train_loader = create_dataloader(train_data, config['training']['batch_size'])
        test_loader = create_dataloader(test_data, config['training']['batch_size'], shuffle=False)
        logging.info(f"Created train loader with {len(train_data)} samples")
        logging.info(f"Created test loader with {len(test_data)} samples")

        # Create model and optimizer
        model = create_model(config, device)
        optimizer = create_optimizer(model, config)

        check_memory_updates(model, config, device)

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