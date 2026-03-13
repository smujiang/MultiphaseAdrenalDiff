
import pandas as pd
from FusionStrategy_utils import compare_fusion_strategies  

if __name__ == "__main__":
    import os 
    from config import DATA_ROOT_DIR
    input_csv = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_unified/all_groups_features_unified_normalized.csv")
    df = pd.read_csv(input_csv)

    results, summary = compare_fusion_strategies(df, n_splits=5, seed=42, group_col="MRN", with_washout=True)

    results_df = {"Strategy":[], "AUC":[], "ACC":[], "Precision":[], "Recall(Sens)":[], "Specificity":[], "F1":[]}
    # Display the key metrics
    for name, summ in summary.items():
        print("\n", "="*30, name, "="*30)
        print(summ.loc[["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]])
        results_df["Strategy"].append(name)
        for metric in ["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]:
            results_df[metric].append(f"{summ.loc[metric, 'mean']:.4f}±{summ.loc[metric, 'std']:.4f}")

    save_fn = os.path.join(DATA_ROOT_DIR, "output", "fusion_strategy_eval", "overall_comparison_results.csv")
    pd.DataFrame(results_df).to_csv(save_fn, index=False)
    print(f"\nResults saved to {save_fn}")



    results, summary = compare_fusion_strategies(df, n_splits=5, seed=42, group_col="MRN", with_washout=False)

    results_df = {"Strategy":[], "AUC":[], "ACC":[], "Precision":[], "Recall(Sens)":[], "Specificity":[], "F1":[]}
    # Display the key metrics
    for name, summ in summary.items():
        print("\n", "="*30, name, "="*30)
        print(summ.loc[["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]])
        results_df["Strategy"].append(name)
        for metric in ["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]:
            results_df[metric].append(f"{summ.loc[metric, 'mean']:.4f}±{summ.loc[metric, 'std']:.4f}")

    save_fn = os.path.join(DATA_ROOT_DIR, "output", "fusion_strategy_eval", "overall_comparison_results_no_washout.csv")
    pd.DataFrame(results_df).to_csv(save_fn, index=False)
    print(f"\nResults saved to {save_fn}")

