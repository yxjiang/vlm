# VLM - Vision Language Model Learning

A project for learning and experimenting with Vision Language Models (VLMs), set up with modern Python tooling.

## 🚀 Features

- **PyTorch 2.9** - Latest PyTorch with Apple Silicon (MPS) acceleration
- **uv** - Fast, modern Python package manager
- **Conda** - Environment management for reproducibility
- **Cross-platform** - Compatible with M3 Mac (local) and A100/B100 GPUs (remote)

## 📁 Project Structure

```
vlm/
├── src/vlm/
│   ├── models/      # VLM model implementations
│   ├── data/        # Data loading and preprocessing
│   ├── utils/       # Utility functions
│   ├── configs/     # Configuration files
│   └── scripts/     # Standalone scripts
├── pyproject.toml   # Project dependencies (managed by uv)
└── uv.lock         # Locked dependencies
```

## 🛠️ Setup

### Prerequisites

- Conda installed
- Python 3.11+

### Installation

1. **Activate the conda environment:**
   ```bash
   conda activate vlm
   ```

2. **Verify PyTorch installation:**
   ```bash
   python src/vlm/scripts/verify_pytorch.py
   ```

### Development

The project uses `uv` for dependency management:

```bash
# Add a new dependency
uv add package-name

# Remove a dependency
uv remove package-name

# Sync dependencies
uv sync
```

## 🧪 Verification

Run the verification script to ensure PyTorch is correctly configured:

```bash
conda activate vlm
python src/vlm/scripts/verify_pytorch.py
```

This will verify:
- ✓ PyTorch 2.9.0 installation
- ✓ MPS (Apple Silicon) acceleration availability
- ✓ Device compatibility (CPU and MPS)

## 💻 Hardware Compatibility

- **Local (M3 Mac)**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **Remote (A100/B100)**: Compatible with CUDA for distributed training

## 📝 License

See [LICENSE](LICENSE) file for details.