from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from .utils import build_transforms, split_indices

def load_data_and_build_loaders(config):
    print("\n" + "="*80)
    print("📂 LOADING DATA")
    print("="*80)

    train_tf, eval_tf = build_transforms(config.img_size)

    full_ds_base = datasets.ImageFolder(config.data_root, transform=None)
    class_names = full_ds_base.classes
    labels = [y for _, y in full_ds_base.samples]

    print(f"Classes found: {class_names}")
    print(f"Total samples: {len(labels)}")

    for i, cls in enumerate(class_names):
        count = sum(1 for label in labels if label == i)
        print(f"  {cls}: {count} samples ({count/len(labels)*100:.1f}%)")

    tr_idx, va_idx, te_idx = split_indices(labels, config.train_ratio,
                                           config.val_ratio, config.test_ratio,
                                           seed=config.seed)

    train_ds_full = datasets.ImageFolder(config.data_root, transform=train_tf)
    eval_ds_full = datasets.ImageFolder(config.data_root, transform=eval_tf)

    ds_train = Subset(train_ds_full, tr_idx)
    ds_val = Subset(eval_ds_full, va_idx)
    ds_test = Subset(eval_ds_full, te_idx)

    train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(ds_train)} samples ({len(ds_train)/len(labels)*100:.1f}%)")
    print(f"   Val:   {len(ds_val)} samples ({len(ds_val)/len(labels)*100:.1f}%)")
    print(f"   Test:  {len(ds_test)} samples ({len(ds_test)/len(labels)*100:.1f}%)")

    print("\n✓ Data integrity check:")
    train_set = set(tr_idx)
    val_set = set(va_idx)
    test_set = set(te_idx)
    print(f"   Train ∩ Val: {len(train_set & val_set)} (should be 0) {'✓' if len(train_set & val_set) == 0 else '✗'}")
    print(f"   Train ∩ Test: {len(train_set & test_set)} (should be 0) {'✓' if len(train_set & test_set) == 0 else '✗'}")
    print(f"   Val ∩ Test: {len(val_set & test_set)} (should be 0) {'✓' if len(val_set & test_set) == 0 else '✗'}")

    return train_loader, val_loader, test_loader, class_names, labels, tr_idx, va_idx, te_idx
