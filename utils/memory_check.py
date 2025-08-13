# utils/memory_update_check.py

import torch

def check_memory_updates(model, config, device):
    """Check memory updates: single pass, continual learning, vs baseline."""
    variant = config['model']['variant']

    print(f"Checking memory behavior for {variant}...")

    # Create test input
    x = torch.randint(0, config['model']['vocab_size'], (2, 32)).to(device)

    if variant == 'baseline':
        # Baseline check - should have no memory to update
        model.eval()
        with torch.no_grad():  # Baseline can use no_grad
            _ = model(x, task_type=config['data']['dataset'])
        print("✓ Baseline: No memory to update (expected)")
        return

    # Titans variants - check memory updates
    if not hasattr(model, 'neural_memory'):
        print(f"⚠ {variant}: No neural_memory found")
        return

    # Store initial memory state
    initial_params = [p.clone() for p in model.neural_memory.memory_network.parameters()]

    # First pass
    model.eval()
    _ = model(x, task_type=config['data']['dataset'])  # We only care about memory updates

    # Check change after first pass
    mid_params = [p.clone() for p in model.neural_memory.memory_network.parameters()]
    change1 = max(torch.abs(initial - mid).max().item()
                  for initial, mid in zip(initial_params, mid_params))

    # Second pass (same input to test continuous learning)
    _ = model(x, task_type=config['data']['dataset'])  # Again, only care about memory updates

    # Check change after second pass
    final_params = list(model.neural_memory.memory_network.parameters())
    change2 = max(torch.abs(mid - final).max().item()
                  for mid, final in zip(mid_params, final_params))

    # Report results
    if change1 > 1e-6:
        print(f"✓ Single pass learning: Memory changed by {change1:.6f}")
    else:
        print(f"⚠ Single pass: No memory change ({change1:.6f})")

    if change2 > 1e-6:
        print(f"✓ Continual learning: Memory changed by {change2:.6f} on 2nd pass")
    else:
        print(f"⚠ Continual learning: No change on 2nd pass ({change2:.6f})")

    if change1 > 1e-6 and change2 > 1e-6:
        print(f"✓ Test-time learning WORKING: {variant} continuously adapts memory!")
    elif change1 > 1e-6:
        print(f"⚠ Partial test-time learning: {variant} updates once but not continuously")
    else:
        print(f"✗ Test-time learning FAILED: {variant} memory not updating")