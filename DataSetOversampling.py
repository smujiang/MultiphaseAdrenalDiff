'''
There is a folder contains "Group1_features.csv Group2_features.csv Group5_features.csv Group6_features.csv Group7_features.csv Group9_features.csv Group10_features.csv Group12_features.csv Group14_features.csv Group15_features.csv"
each csv file contains features for that group.
The available features for each group are defined in group_features.py. For example, Group1_features contains morph_features + NC phase texture features.

write a function to load the features for a given group from the corresponding csv file

augment the dataset by propagating the data in a group to anther group share the same available features. For example, data in Group5, Group7, Group9, Group12, Group14, Group15 can be propagated to Group1 since they all contain morph_features + NC phase texture features. The new data should be saved in a new folder called
"grouped_instance_features_augmented", and the new csv files should be named as "Group1_features_augmented.csv".

Create Group3, Group4, Group8, Group11, Group13 augmented datasets similarly.

Group1_aug_sources = [1,5,7,9,12,14]
Group2_aug_sources = [2,5,6,9,12,15]
Group3_aug_sources = [5,6,7,9,10,14,15]
Group4_aug_sources = [5,6,7,10]
Group5_aug_sources = [5]
Group6_aug_sources = [6,5]
Group7_aug_sources = [7,5]
Group8_aug_sources = [5]
Group9_aug_sources = [9,5]
Group10_aug_sources = [10,5,6,7]
Group11_aug_sources = [5,7]
Group12_aug_sources = [12,5,9]
Group13_aug_sources = [5,6]
Group14_aug_sources = [14,5,7,9]
Group15_aug_sources = [15,5,6,9]
'''
import pandas as pd
import os

from group_features import *

# Define the augmentation source groups for each target group
Group1_aug_sources = [1,5,7,9,12,14]
Group2_aug_sources = [2,5,6,9,12,15]
Group3_aug_sources = [5,6,7,9,10,14,15]
Group4_aug_sources = [5,6,7,10]
Group5_aug_sources = [5]
Group6_aug_sources = [6,5]
Group7_aug_sources = [7,5]
Group8_aug_sources = [5]
Group9_aug_sources = [9,5]
Group10_aug_sources = [10,5,6,7]
Group11_aug_sources = [5,7]
Group12_aug_sources = [12,5,9]
Group13_aug_sources = [5,6]
Group14_aug_sources = [14,5,7,9]
Group15_aug_sources = [15,5,6,9]

def load_group_csv(group_idx, input_dir: str = ".") -> pd.DataFrame:
    """
    Load the CSV file for a given group.
    group: int or 'GroupN' or 'N'
    Returns DataFrame or None if file not found.
    """
    p = os.path.join(input_dir, f"Group{group_idx}_features.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)

def propagate_features_to_other_groups(target_group_idx, input_dir: str = ".", output_dir: str = "."):
    """
    For a target group, find all source groups that can augment it,
    load their CSVs, concatenate them, and save the augmented CSV.
    """
    aug_sources = eval(f"Group{target_group_idx}_aug_sources")
    dfs = []
    for src_idx in aug_sources:
        df = load_group_csv(src_idx, input_dir)
        # only include the features defined for the target group
        group_features = eval(f"Group{target_group_idx}_features")
        if df is not None:
            dfs.append(df[["MRN","StudyDate", "group", "available_phases", "malignancy"]+group_features])  # keep MRN, StudyDate, malignancy for identification
    if not dfs:
        print(f"No source CSVs found for Group{target_group_idx}.")
        return
    augmented_df = pd.concat(dfs, ignore_index=True)
    # rename column "group" to "original_group" to avoid confusion
    augmented_df = augmented_df.rename(columns={"group": "original_group"})
    # save the augmented dataframe
    os.makedirs(output_dir, exist_ok=True)
    sv_path = os.path.join(output_dir, f"Group{target_group_idx}_features_augmented.csv")
    augmented_df.to_csv(sv_path, index=False)
    print(f"Augmented CSV for Group{target_group_idx} saved to {sv_path}.")

if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    input_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features")
    output_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_augmented")
    for target_group_idx in range(1, 16):
        propagate_features_to_other_groups(target_group_idx, input_dir, output_dir)
        # save a normalized version as well
        p = os.path.join(output_dir, f"Group{target_group_idx}_features_augmented.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            feature_cols = [col for col in df.columns if col not in META_COLS and col != "original_group"]
            df_norm = df.copy()
            for col in feature_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df_norm[col] = (df[col] - mean) / std
                else:
                    df_norm[col] = 0.0  # if std is 0, set normalized value to 0
            norm_path = os.path.join(output_dir, f"Group{target_group_idx}_features_augmented_normalized.csv")
            df_norm.to_csv(norm_path, index=False)
            print(f"Normalized augmented CSV for Group{target_group_idx} saved to {norm_path}.")
    
    # save all groups data into a single csv, add a column "assigned_group" to indicate the source group
    all_dfs = []
    for target_group_idx in range(1, 16):
        p = os.path.join(output_dir, f"Group{target_group_idx}_features_augmented.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["assigned_group"] = target_group_idx
            all_dfs.append(df)
    if all_dfs:
        all_data_df = pd.concat(all_dfs, ignore_index=True)
        all_data_path = os.path.join(output_dir, "All_Groups_features_augmented.csv")
        all_data_df.to_csv(all_data_path, index=False)
        print(f"All groups augmented data saved to {all_data_path}.")   
    # save normalized version of all groups data into a single csv
    if all_dfs:
        all_data_df = pd.concat(all_dfs, ignore_index=True)
        feature_cols = [col for col in all_data_df.columns if col not in META_COLS and col not in ["original_group", "assigned_group"]]
        all_data_df_norm = all_data_df.copy()
        for col in feature_cols:
            mean = all_data_df[col].mean()
            std = all_data_df[col].std()
            if std > 0:
                all_data_df_norm[col] = (all_data_df[col] - mean) / std
            else:
                all_data_df_norm[col] = 0.0
        all_data_norm_path = os.path.join(output_dir, "All_Groups_features_augmented_normalized.csv")
        all_data_df_norm.to_csv(all_data_norm_path, index=False)
        print(f"All groups normalized augmented data saved to {all_data_norm_path}.")

