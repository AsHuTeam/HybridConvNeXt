import time
import json
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from .viz import (
    plot_confusion_matrix, plot_roc_curves,
    save_classification_report, plot_training_history
)

def train_one_epoch(model, loader, optimizer, device, scaler, use_amp=True):
    """Train model for one epoch"""
    model.train()
    total_loss, total, correct = 0.0, 0, 0

    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, device, class_names):
    """Evaluate model on given loader"""
    model.eval()
    y_true, y_pred, probs = [], [], []

    for imgs, y in loader:
        imgs, y = imgs.to(device), y.to(device)
        out = model(imgs)
        p = F.softmax(out, dim=1)

        y_true.append(y.cpu().numpy())
        y_pred.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    probs = np.concatenate(probs)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    try:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        auc_score = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    except:
        auc_score = float('nan')

    return report, cm, auc_score, y_true, y_pred, probs

def run_experiment(study_name, model_class, train_loader, val_loader, test_loader,
                   class_names, config):
    """Run single ablation study experiment"""
    print(f"\n{'='*80}")
    print(f"🔬 Running: {study_name}")
    print(f"{'='*80}")

    start_time = time.time()

    model = model_class(num_classes=len(class_names)).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}

    print(f"\n🏋️ Training for {config.epochs} epochs...")
    for ep in range(1, config.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          config.device, scaler, config.use_amp)
        val_report, _, _, _, _, _ = eval_epoch(model, val_loader, config.device, class_names)
        val_acc = val_report['accuracy']
        val_f1 = val_report['macro avg']['f1-score']

        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"  Epoch {ep:02d}/{config.epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
              f"Val Acc: {val_acc:.3f} F1: {val_f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'history': history,
            }, f"{config.output_dir}/models/{study_name}_best.pt")
            print(f"    ✓ Best model saved (Val Acc: {val_acc:.4f})")

    print(f"\n📥 Loading best model from epoch {best_epoch}...")
    ckpt = torch.load(f"{config.output_dir}/models/{study_name}_best.pt",
                      map_location=config.device)
    model.load_state_dict(ckpt['model'])

    print(f"🧪 Evaluating on test set...")
    test_report, test_cm, test_auc, y_true, y_pred, probs = eval_epoch(
        model, test_loader, config.device, class_names
    )

    print(f"💾 Saving results...")
    plot_confusion_matrix(test_cm, class_names, study_name,
                          f"{config.output_dir}/plots/{study_name}_confusion_matrix.png")
    plot_roc_curves(y_true, probs, class_names, study_name,
                    f"{config.output_dir}/plots/{study_name}_roc_curves.png")
    plot_training_history(history, study_name,
                          f"{config.output_dir}/plots/{study_name}_training_history.png")
    save_classification_report(test_report, class_names, study_name,
                               f"{config.output_dir}/reports/{study_name}_report.txt")

    with open(f"{config.output_dir}/reports/{study_name}_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time

    results = {
        'study_name': study_name,
        'test_accuracy': test_report['accuracy'],
        'test_precision': test_report['macro avg']['precision'],
        'test_recall': test_report['macro avg']['recall'],
        'test_f1': test_report['macro avg']['f1-score'],
        'test_auc': test_auc,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time_sec': total_time,
        'training_time_min': total_time / 60,
    }

    for cls in class_names:
        results[f'{cls}_precision'] = test_report[cls]['precision']
        results[f'{cls}_recall'] = test_report[cls]['recall']
        results[f'{cls}_f1'] = test_report[cls]['f1-score']
        results[f'{cls}_support'] = int(test_report[cls]['support'])

    print(f"\n✅ {study_name} completed in {total_time/60:.2f} minutes")
    print(f"   Test Acc: {results['test_accuracy']:.4f} | "
          f"F1: {results['test_f1']:.4f} | AUC: {results['test_auc']:.4f}")

    return results, history
