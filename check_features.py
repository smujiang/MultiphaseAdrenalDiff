import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from group_features import *  # import feature definitions for each group
from config import DATA_ROOT_DIR

plt.rcParams.update({'font.size': 10})

for group_idx in range(1, 16):

    # fn = f"{DATA_ROOT_DIR}/processed_data/grouped_instance_features_augmented/Group{group_idx}_features_augmented.csv"
    # output_dir = f"{DATA_ROOT_DIR}/processed_data/feature_distributions_augmented"
    fn = f"{DATA_ROOT_DIR}/processed_data/grouped_instance_features/Group{group_idx}_features.csv"
    output_dir = f"{DATA_ROOT_DIR}/processed_data/feature_distributions"

    os.makedirs(output_dir, exist_ok=True)

    # check feature distributions to show differences in malignancy prediction features
    print(f"Checking feature distributions for Group {group_idx}...")
    if not os.path.exists(fn):
        print(f"File {fn} does not exist. Skipping Group {group_idx}.")
        continue
    df_aug = pd.read_csv(fn)
    df_aug = df_aug.rename(columns={"original_group": "group"})
    # get features according to group index
    features_to_check = eval(f"Group{group_idx}_features")  
    for feature in features_to_check:
        if "washout" in feature:
            plt.figure(figsize=(6, 3), dpi=250)
            sns.kdeplot(data=df_aug, x=feature, hue='malignancy', palette={0: 'green', 1: 'red'}, common_norm=False)
            plt.title(f'Distribution of {feature} in Group {group_idx}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend(title='Malignancy', labels=['Malignant','Benign'])
            plt.tight_layout()
            sv_fn = os.path.join(output_dir, f"Group{group_idx}_{feature}_distribution.png")
            plt.savefig(sv_fn)
            # plt.show()

