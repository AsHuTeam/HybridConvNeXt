import os
import torch

# ====================== CONFIGURATION ======================
class Config:
    # Data
    data_root = r"Path\dataset"
    img_size = 224
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2

    # Training
    epochs = 20
    batch_size = 64
    lr = 1e-4
    weight_decay = 1e-4
    seed = 42
    use_amp = True

    # Output
    output_dir = "ablation_results_dual_attentionnew"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

def init_output_dirs():
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(f"{config.output_dir}/models", exist_ok=True)
    os.makedirs(f"{config.output_dir}/plots", exist_ok=True)
    os.makedirs(f"{config.output_dir}/reports", exist_ok=True)

def print_banner():
    print("="*80)
    print("🚀 DUAL-ATTENTION ABLATION STUDY")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Output directory: {config.output_dir}")
    print(f"Image size: {config.img_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print("="*80)
