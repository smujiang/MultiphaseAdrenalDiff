
import pandas as pd
from FusionStrategy_utils import compare_fusion_strategies 
from config import DATA_ROOT_DIR
import os

# -------------------------
# Example
# -------------------------

if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    import os
    input_csv = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_unified/all_groups_features_unified_normalized.csv")
    df = pd.read_csv(input_csv)
    print("Loaded data:", df.shape)

    label_col = "malignancy"
    group_col = "group"
    patient_col = "MRN"
    oof_splits = 5
    seed = 42

    # only run on specific groups 
    groups = ["Group5", "Group7"]
    # groups = sorted(df[group_col].dropna().unique().tolist())

    df = df.copy()
    df[label_col] = df[label_col].astype(int)

    for g in groups:
        df_in_group = df[df[group_col] == g]
        results, summary = compare_fusion_strategies(df_in_group, n_splits=oof_splits, seed=seed, group_col=patient_col, with_washout=True)
        
        results_df = {"Strategy":[], "AUC":[], "ACC":[], "Precision":[], "Recall(Sens)":[], "Specificity":[], "F1":[]}
        # Display the key metrics
        for name, summ in summary.items():
            print("\n", "="*30, name, "="*30)
            print(summ.loc[["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]])
            results_df["Strategy"].append(name)
            for metric in ["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]:
                results_df[metric].append(f"{summ.loc[metric, 'mean']:.4f}±{summ.loc[metric, 'std']:.4f}")

        save_fn = os.path.join(DATA_ROOT_DIR, "output", "fusion_strategy_eval", f"{g}_groupwise_comparison_results.csv")
        pd.DataFrame(results_df).to_csv(save_fn, index=False)
        print(f"\nResults saved to {save_fn}")


    for g in groups:
        df_in_group = df[df[group_col] == g]
        results, summary = compare_fusion_strategies(df_in_group, n_splits=oof_splits, seed=seed, group_col=patient_col, with_washout=False)
        
        results_df = {"Strategy":[], "AUC":[], "ACC":[], "Precision":[], "Recall(Sens)":[], "Specificity":[], "F1":[]}
        # Display the key metrics
        for name, summ in summary.items():
            print("\n", "="*30, name, "="*30)
            print(summ.loc[["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]])
            results_df["Strategy"].append(name)
            for metric in ["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]:
                results_df[metric].append(f"{summ.loc[metric, 'mean']:.4f}±{summ.loc[metric, 'std']:.4f}")

        save_fn = os.path.join(DATA_ROOT_DIR, "output", "fusion_strategy_eval", f"{g}_groupwise_comparison_results_no_washout.csv")
        pd.DataFrame(results_df).to_csv(save_fn, index=False)
        print(f"\nResults saved to {save_fn}")




