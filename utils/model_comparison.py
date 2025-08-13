# utils/model_comparison.py

import json
import os
import matplotlib.pyplot as plt
import logging


def save_results(results, config):
    """Save results to JSON file."""
    dataset = config['data']['dataset']
    results_file = f"results_{dataset}.json"

    # Convert to JSON-serializable format
    json_results = {}
    for variant, data in results.items():
        json_results[variant] = {
            'train_losses': data['train_losses'],
            'test_losses': data['test_losses'],
            'test_metrics': data['test_metrics']
        }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"✓ Saved results to {results_file}")


def load_existing_results(config):
    """Load existing results if they exist."""
    dataset = config['data']['dataset']
    results_file = f"results_{dataset}.json"

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"✓ Loaded existing results from {results_file}")
            print(f"  Found results for: {list(results.keys())}")
            return results
        except Exception as e:
            print(f"⚠ Failed to load {results_file}: {e}")
            return {}
    else:
        print(f"No existing results file found ({results_file})")
        return {}


def run_all_variants_comparison(config, device):
    """Run all variants and collect results for comparison with resume capability."""

    from main import (
        create_model, create_optimizer, check_memory_updates,
        load_dataset, create_dataloader, train_epoch, evaluate
    )

    variants = ['baseline', 'LMM', 'MAC', 'MAG', 'MAL']

    # Load existing results
    results = load_existing_results(config)

    print("Running comparison across all variants...")

    for variant in variants:
        # Skip if already completed
        if variant in results:
            print(f"✓ Skipping {variant} - already completed")
            continue

        print(f"\n{'=' * 50}")
        print(f"Training {variant}...")
        print(f"{'=' * 50}")

        try:
            # Update config for this variant
            config['model']['variant'] = variant

            # Create model and optimizer
            model = create_model(config, device)
            optimizer = create_optimizer(model, config)

            # Check memory updates
            check_memory_updates(model, config, device)

            # Load dataset (same for all variants)
            train_data, test_data = load_dataset(config)
            train_loader = create_dataloader(train_data, config['training']['batch_size'])
            test_loader = create_dataloader(test_data, config['training']['batch_size'], shuffle=False)

            # Store results for this variant
            variant_results = {
                'train_losses': [],
                'test_losses': [],
                'test_metrics': []
            }

            # Training loop
            for epoch in range(config['training']['max_epochs']):
                train_loss = train_epoch(model, train_loader, optimizer, device, config)
                test_loss, test_metrics = evaluate(model, test_loader, device, config)

                variant_results['train_losses'].append(train_loss)
                variant_results['test_losses'].append(test_loss)
                variant_results['test_metrics'].append(test_metrics)

                # Log progress occasionally
                if epoch % (config['logging']['log_every'] * 2) == 0:
                    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
                    print(f"  Epoch {epoch:3d}: Train {train_loss:.4f}, Test {test_loss:.4f}, {metrics_str}")

            # Save this variant's results
            results[variant] = variant_results
            print(f"✓ {variant} training complete!")

            # Save intermediate results after each variant
            save_results(results, config)

        except Exception as e:
            print(f"✗ {variant} failed with error: {e}")
            print(f"Results saved up to this point. You can resume by running again.")
            # Save what we have so far
            if results:
                save_results(results, config)
            raise  # Re-raise the exception to stop execution

    print(f"\n✓ All variants completed! Final results: {list(results.keys())}")
    return results


def plot_comparison_results(results, config):
    """Plot comparison results with losses and metrics."""
    if not results:
        print("⚠ No results to plot!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Losses
    for variant, data in results.items():
        epochs = range(len(data['train_losses']))
        ax1.plot(epochs, data['train_losses'], '--', label=f'{variant} (train)', alpha=0.7)
        ax1.plot(epochs, data['test_losses'], '-', label=f'{variant} (test)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Test Losses')
    ax1.grid(True, alpha=0.3)

    # Move legend to bottom with multiple columns
    ax1.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)

    # Right panel: Final metrics
    dataset = config['data']['dataset']
    if dataset == 'copy_task':
        metric_key = 'copy_accuracy'
        ylabel = 'Copy Accuracy'
    elif dataset == 'language_modeling':
        metric_key = 'perplexity'
        ylabel = 'Perplexity'
    elif dataset == 'needle_haystack':
        metric_key = 'position_accuracy'
        ylabel = 'Position Accuracy'

    variants = list(results.keys())
    final_metrics = []

    # Handle case where some variants might not be complete
    for variant in variants:
        if results[variant]['test_metrics']:
            final_metrics.append(results[variant]['test_metrics'][-1][metric_key])
        else:
            final_metrics.append(0)  # Fallback for incomplete runs

    bars = ax2.bar(variants, final_metrics)
    ax2.set_ylabel(ylabel)
    ax2.set_title(f'Final {ylabel} by Variant')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, final_metrics):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(final_metrics) * 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # Adjust layout to accommodate bottom legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for legend

    # Save figure
    fig_name = f"titans_comparison_{dataset}.png"
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {fig_name}")
    plt.show()