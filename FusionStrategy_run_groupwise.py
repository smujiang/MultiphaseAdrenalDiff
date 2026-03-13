
import os
import pandas as pd
from FusionStrategy_utils import compare_fusion_strategies 
from group_features import ALL_FEATURES_FOR_DATA_FRAME_EXCLUDE_WASHOUT, morph_features, atten_features, ALL_FEATURES_FOR_DATA_FRAME, PHASES, META_COLS


def expand_features_csv(csv_path, all_features):
    """
    Read a CSV file and expand it so that it contains all columns
    listed in `all_features`. Missing columns are filled with blank "".

    Parameters
    ----------
    csv_path : str
        Path to input CSV
    all_features : list
        List of all feature column names (desired final columns)
    save_path : str or None
        If provided, save expanded CSV to this path

    Returns
    -------
    df : pandas.DataFrame
        Expanded dataframe
    """

    df = pd.read_csv(csv_path)

    # Add missing feature columns
    for col in all_features:
        if col not in df.columns:
            df[col] = None

    return df

# -------------------------
# Example
# -------------------------

if __name__ == "__main__":
    import os 
    from config import DATA_ROOT_DIR
    input_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_augmented")
    # input_csv = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_unified/all_groups_features_unified_normalized.csv")    

    label_col = "malignancy"
    group_col = "original_group"
    patient_col = "MRN"
    oof_splits = 5
    seed = 42

    # Run on all groups 
    groups = ["Group1", "Group2", "Group3", "Group4", "Group5", "Group6", "Group7", "Group8", "Group9", "Group10", "Group11", "Group12", "Group13", "Group14", "Group15"]

    for g in groups:
        input_csv = os.path.join(input_dir, f"{g}_features_augmented.csv")
       
        expanded_df = expand_features_csv(input_csv, ALL_FEATURES_FOR_DATA_FRAME)
        expanded_df[label_col] = expanded_df[label_col].astype(int)
        if g in ["Group5", "Group7"]:
            with_washout = True
        else:            
            with_washout = False
        print("evaluating group:", g)
        results, summary = compare_fusion_strategies(expanded_df, n_splits=oof_splits, seed=seed, group_col=patient_col, with_washout=with_washout)
        
        results_df = {"Strategy":[], "AUC":[], "ACC":[], "Precision":[], "Recall(Sens)":[], "Specificity":[], "F1":[]}
        # Display the key metrics
        for name, summ in summary.items():
            print("\n", "="*30, name, "="*30)
            print(summ.loc[["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]])
            results_df["Strategy"].append(name)
            for metric in ["AUC", "ACC", "Precision", "Recall(Sens)", "Specificity", "F1"]:
                results_df[metric].append(f"{summ.loc[metric, 'mean']:.4f}±{summ.loc[metric, 'std']:.4f}")

        save_fn = os.path.join(DATA_ROOT_DIR, "output", "fusion_strategy_eval_groupwise", f"{g}_groupwise_comparison_results.csv")
        pd.DataFrame(results_df).to_csv(save_fn, index=False)
        print(f"\nResults saved to {save_fn}")

