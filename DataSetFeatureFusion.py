'''
I would like to put all the dataset features together into one table for model training and evaluation. The dataset was originally grouped into 15 groups based on the available phases and features. 
The created table should contain all the features from all the groups, with missing features filled with NaN. The complete list of features is in ALL_FEATURES_FOR_DATA_FRAME.
Please write python code to create a pandas DataFrame to save all the data, filling missing features with NaN.

ALL_FEATURES_FOR_DATA_FRAME =  morph_features + atten_features + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity","avg_HU_NC",
                                                      "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity","avg_HU_AR",
                                                      "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity","avg_HU_PV",
                                                      "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity","avg_HU_Delay"
                                                      ]

 

The dataset was grouped as follows:
Group_Dict = {"Group1": "NC",
               "Group2": "AR",
               "Group3": "PV",
               "Group4": "Delay",
               "Group5": "NC; AR; PV; Delay",
               "Group6": "AR; PV; Delay",
               "Group7": "NC; PV; Delay",
               "Group8": "NC; AR; Delay",
               "Group9": "NC; AR; PV",
               "Group10": "PV; Delay",
               "Group11": "NC; Delay",
               "Group12": "NC; AR",
               "Group13": "AR; Delay",
               "Group14": "NC; PV",
               "Group15": "AR; PV"}

The available features for each group are defined as follows:

morph_features = ["area", "perimeter", "eccentricity", "axis_major_length", "axis_minor_length"]
atten_features = ["absolute_washout", "relative_washout", "absolute_washout_rate", "relative_washout_rate"]

Group1_features = morph_features + ["avg_HU_NC"] +["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity"]
Group2_features = morph_features + ["avg_HU_AR"] + ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity"]
Group3_features = morph_features + ["avg_HU_PV"] + ["PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity"]
Group4_features = morph_features + ["avg_HU_Delay"] + ["Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"]
Group5_features = morph_features + ["avg_HU_NC", "avg_HU_AR", "avg_HU_PV", "avg_HU_Delay"] + atten_features + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                                      "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                                      "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                                      "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                                      ]
Group6_features = morph_features + ["avg_HU_AR", "avg_HU_PV", "avg_HU_Delay"] + ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]
Group7_features = morph_features + ["avg_HU_NC", "avg_HU_PV", "avg_HU_Delay"] + atten_features + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]
Group8_features = morph_features + ["avg_HU_NC", "avg_HU_AR", "avg_HU_Delay"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]
Group9_features = morph_features + ["avg_HU_NC", "avg_HU_AR", "avg_HU_PV"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity"
                                    ]
Group10_features = morph_features + ["avg_HU_PV", "avg_HU_Delay"] + ["PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]


Group11_features = morph_features + ["avg_HU_NC", "avg_HU_Delay"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]

Group12_features = morph_features + ["avg_HU_NC", "avg_HU_AR"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity"
                                    ]

Group13_features = morph_features + ["avg_HU_AR", "avg_HU_Delay"] + ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity"
                                    ]


Group14_features = morph_features + ["avg_HU_NC", "avg_HU_PV"] + ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity"
                                    ]


Group15_features = morph_features + ["avg_HU_AR", "avg_HU_PV"] + ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity",
                                    "PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity"
                                    ]



'''

import numpy as np
import pandas as pd
import os
from group_features import Group1_features, Group2_features, Group3_features, Group4_features, \
    Group5_features, Group6_features, Group7_features, Group8_features, Group9_features, Group10_features, \
    Group11_features, Group12_features, Group13_features, Group14_features, Group15_features, ALL_FEATURES_FOR_DATA_FRAME, META_COLS


GROUP_FEATURES = {
    "Group1": Group1_features,
    "Group2": Group2_features,
    "Group3": Group3_features,
    "Group4": Group4_features,
    "Group5": Group5_features,
    "Group6": Group6_features,
    "Group7": Group7_features,
    "Group8": Group8_features,
    "Group9": Group9_features,
    "Group10": Group10_features,
    "Group11": Group11_features,
    "Group12": Group12_features,
    "Group13": Group13_features,
    "Group14": Group14_features,
    "Group15": Group15_features,
}

def build_unified_row(meta: dict, feature_dict: dict, group_name: str) -> dict:
    """
    meta must contain keys in META_COLS (or a subset; missing -> NaN)
    feature_dict contains extracted features for this sample
    group_name is one of Group1..Group15
    """
    row = {c: meta.get(c, np.nan) for c in META_COLS}
    row["group"] = group_name  # ensure group is consistent

    # initialize all features as NaN
    row.update({f: np.nan for f in ALL_FEATURES_FOR_DATA_FRAME})

    # only fill features that are defined as available in this group
    valid_feats = GROUP_FEATURES[group_name]
    for f in valid_feats:
        if f in feature_dict:
            row[f] = feature_dict[f]

    return row


# -----------------------------
# read all group CSVs, unify columns, concat
# -----------------------------
def build_feature_table_from_group_csvs(group_csv_map: dict, output_csv: str) -> pd.DataFrame:
    """
    group_csv_map example:
      {
        "Group1": "/path/to/group1.csv",
        "Group2": "/path/to/group2.csv",
        ...
      }

    Each group CSV is expected to contain:
      - metadata columns: MRN, StudyDate, group, available_phases, malignancy
      - a subset of feature columns (group-specific)
    Missing features will be created and filled with NaN.
    """
    dfs = []

    for group_name, csv_path in group_csv_map.items():
        df_g = pd.read_csv(csv_path)

        # Ensure metadata columns exist (create if missing)
        for c in META_COLS:
            if c not in df_g.columns:
                df_g[c] = np.nan

        # Enforce group label from loop (more reliable than CSV contents)
        df_g["group"] = group_name

        # Make sure all expected feature cols exist (create missing as NaN)
        for f in ALL_FEATURES_FOR_DATA_FRAME:
            if f not in df_g.columns:
                df_g[f] = np.nan

        # Keep only meta + all features (in a stable order)
        df_g = df_g[META_COLS + ALL_FEATURES_FOR_DATA_FRAME]
        dfs.append(df_g)

    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    # Optional: make sure numeric feature columns are numeric (coerce errors -> NaN)
    for f in ALL_FEATURES_FOR_DATA_FRAME:
        df_all[f] = pd.to_numeric(df_all[f], errors="coerce")

    if output_csv is not None:
        df_all.to_csv(output_csv, index=False)

    return df_all



if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    input_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features")
    output_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_unified")
    os.makedirs(output_dir, exist_ok=True)
    
    group_csv_map = {}
    csv_list = os.listdir(input_dir)
    for group_idx in csv_list:
        if group_idx.startswith("Group") and group_idx.endswith("_features.csv"):
            group_name = group_idx.split("_")[0]  # e.g., "Group1"
            group_csv_map[group_name] = os.path.join(input_dir, group_idx)
    
    output_csv = os.path.join(output_dir, "all_groups_features_unified.csv")
    df_features = build_feature_table_from_group_csvs(group_csv_map, output_csv=output_csv)

    print(df_features.shape)
    print(df_features.head())

    # create a normalized version (z-score) of the features, ignoring NaNs
    # Let XGBoost natively handle NaN
    df_normalized = df_features.copy()
    for f in ALL_FEATURES_FOR_DATA_FRAME:
        mean = df_normalized[f].mean(skipna=True)
        std = df_normalized[f].std(skipna=True)
        if std > 0:
            df_normalized[f] = (df_normalized[f] - mean) / std
        else:
            df_normalized[f] = 0.0  # if std is 0, set to 0.0

    output_normalized_csv = os.path.join(output_dir, "all_groups_features_unified_normalized.csv")
    df_normalized.to_csv(output_normalized_csv, index=False)

    # create a version with missing features filled with median of that feature across the dataset
    df_filled = df_normalized.copy()
    for f in ALL_FEATURES_FOR_DATA_FRAME:
        median = df_filled[f].median(skipna=True)
        df_filled[f] = df_filled[f].fillna(median)
    output_filled_csv = os.path.join(output_dir, "all_groups_features_unified_normalized_medianfilled.csv")
    df_filled.to_csv(output_filled_csv, index=False)



