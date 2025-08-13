# Titans - Learning to Memorize at Test Time

A PyTorch implementation of the neural memory mechanism and Titans variants from "Titans: 
Learning to Memorize at Test Time" (Behrouz et al., 2024).

This implementation focuses on the core neural memory components that enable test-time adaptation through surprise-based learning, momentum mechanisms, and adaptive forgetting.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/AmayaGS/Titans_LTMM
cd Titans_LTMM
```

#### General Requirements
- Python 3.11.7
- PyTorch 2.5
- NVIDIA GPU with CUDA 12.4

```bash
# Virtual Environment
python -m venv titans_env
source titans_env/bin/activate

# Install dependencies
pip install torch pyyaml numpy

# PyTorch with cuda capabilities
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

## Run with default configuration (MAC variant on copy task)
`python main.py`

## Try different model variants
`python main.py --config config.yaml  # Edit config.yaml to change settings`

## Configuration

Modify `config.yaml` to experiment with different settings:

```yaml
# config.yaml
model:
  variant: "MAC"                    # Options: baseline, MAC, MAG, MAL, LMM
  d_model: 32
  n_heads: 2
  memory_layers: 2                  # Depth of neural memory module
  persistent_memory_size: 8

data:
  dataset: "copy_task"              # Options: copy_task, language_modeling, needle_haystack
  seq_len: 96
  vocab_size: 10
  num_samples: 1000

training:
  batch_size: 4
  learning_rate: 0.0001
  max_epochs: 200
  gradient_clip: 1.0
```

## Model Variants

- **baseline**: Standard transformer for comparison
- **LMM**: Neural memory module only (no attention)
- **MAC**: Memory as Context - concatenates retrieved memory with input for attention
- **MAG**: Memory as Gate - parallel attention and memory branches combined via gating
- **MAL**: Memory as Layer - sequential processing (memory → attention)

## Synthetic Datasets

**Copy Task**: Learn to copy sequences after a delimiter token
- Input: `[3,1,7,2,0,0,0,0,0]`
- Target: `[3,1,7,2,0,3,1,7,2]`

**Language Modeling**: Predict next token in repeating patterns
- Sequences like `[1,2,3,1,2,3,1,2,3...]`

**Needle in Haystack**: Find special token position in long sequence
- Locate position of "needle" token within distractor sequence

## Key Features

- **Neural Memory Module**: Surprise-based parameter updates during inference  
- **Momentum Mechanism**: Temporal integration of surprise signals  
- **Adaptive Forgetting**: Data-dependent memory capacity management  
- **Test-Time Adaptation**: Memory updates during evaluation  
- **Multiple Integration Strategies**: Four architectural variants  

## Technical Details

For detailed explanation of the neural memory mechanism, mathematical formulation, and implementation choices, see:

**[📋 Technical Report](TECHNICAL_REPORT.md)**

## Project Structure

```
Titans_LTMM/
├── README.md                # This file
├── TECHNICAL_REPORT.md      # Detailed technical explanation
├── config.yaml             # Configuration file
├── main.py                  # Training script
├── models/
│   ├── baselines.py         # Standard transformer baseline
│   ├── memory_module.py     # Core neural memory implementation
│   └── titan_models.py      # Titans architectural variants (MAC, MAG, MAL, LMM)
└── utils/
    ├── data.py              # Synthetic dataset generation
    └── training.py          # Training and evaluation loops          
    └── memory_check.py      # check memory is updating at test time
```

## Implementation Notes

This is a proof-of-concept implementation focusing on the core neural memory mechanisms. Some simplifications compared to the original paper:

- Uses full sequence processing instead of chunked segmentation
- Standard attention instead of sliding window attention
- Small-scale synthetic tasks for rapid experimentation

The goal is to validate the mathematical correctness of the memory mechanisms rather than achieve state-of-the-art performance.

### Example Output

```Epoch   0: Train Loss 2.1543, Test Loss 2.0891, copy_accuracy: 0.1250
Epoch  10: Train Loss 1.8234, Test Loss 1.7456, copy_accuracy: 0.3750  
Epoch  50: Train Loss 0.8901, Test Loss 0.9123, copy_accuracy: 0.8750
Epoch 100: Train Loss 0.2345, Test Loss 0.2567, copy_accuracy: 0.9375
```

##Citation

```bibtex

bibtex@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}
```