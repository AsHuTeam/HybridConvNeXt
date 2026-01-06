import random
import numpy as np
import torch
from torchvision import transforms

# ====================== UTILITIES ======================
def set_seed(s=42):
    """Set random seed for reproducibility"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_indices(labels, train_ratio, val_ratio, test_ratio=None, seed=42):
    """Split dataset indices while maintaining class balance"""
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    idxs = np.arange(len(labels))
    train_ids, val_ids, test_ids = [], [], []

    for c in np.unique(labels):
        c_idx = idxs[labels == c]
        rng.shuffle(c_idx)
        n = len(c_idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val if (test_ratio is None) else int(n * test_ratio)

        train_ids += c_idx[:n_train].tolist()
        val_ids += c_idx[n_train:n_train+n_val].tolist()
        test_ids += c_idx[n_train+n_val:n_train+n_val+n_test].tolist()

    return train_ids, val_ids, test_ids

def build_transforms(img_size=224):
    """Build training and evaluation transforms"""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf
