# ====================== COMPLETE DUAL-ATTENTION ABLATION STUDY ======================


import pandas as pd

from src.config import config, init_output_dirs, print_banner
from src.utils import set_seed
from src.data import load_data_and_build_loaders
from src.models import MODEL_REGISTRY
from src.train_eval import run_experiment
from src.compare import generate_comparison_and_summary

def main():
    init_output_dirs()
    print_banner()

    # ====================== DATA LOADING ======================
    set_seed(config.seed)

    train_loader, val_loader, test_loader, class_names, labels, tr_idx, va_idx, te_idx =         load_data_and_build_loaders(config)

    # ====================== RUN ALL EXPERIMENTS ======================
    print("\n" + "="*80)
    print("🚀 STARTING ABLATION STUDIES")
    print("="*80)

    all_results = []

    for study_name, model_class in MODEL_REGISTRY.items():
        try:
            results, history = run_experiment(
                study_name, model_class, train_loader, val_loader, test_loader,
                class_names, config
            )
            all_results.append(results)

        except Exception as e:
            print(f"\n❌ Error in {study_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ====================== GENERATE COMPARISON REPORT ======================
    df_results = pd.DataFrame(all_results)
    generate_comparison_and_summary(df_results, class_names, config)

if __name__ == "__main__":
    main()
