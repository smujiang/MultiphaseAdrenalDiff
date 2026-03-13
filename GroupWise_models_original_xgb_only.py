import numpy as np
import pandas as pd
import os
from sklearn.discriminant_analysis import StandardScaler
from group_features import *
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import visualize_train_test_sample_UMAP, visualize_train_test_sample_tSNE
from models import make_xgb


if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    input_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features")
    output_dir = os.path.join(DATA_ROOT_DIR, "processed_data/models_output")

    performance_records = {"Group": [], "Avg_Accuracy": [], "Avg_AUC": [], "Avg_Accuracy_top_features": [], "Avg_AUC_top_features": [], "Top_Features": []}

    os.makedirs(output_dir, exist_ok=True)
    for group_idx in range(1, 16):
        df_fn = os.path.join(input_dir, f"Group{group_idx}_features.csv")
        if not os.path.exists(df_fn):
            print(f"File {df_fn} does not exist. Skipping Group {group_idx}.")
            continue
        df = pd.read_csv(df_fn)
        features = eval(f"Group{group_idx}_features")
        X = df[features].values
        # Normalize features if needed
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df["malignancy"].values
        
        if len(y) < 20:
            print(f"Not enough samples in Group {group_idx} for evaluation. Skipping.")
            continue

        # split data into train and test for 5-fold cross-validation

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_list = []
        auc_list = []
        fold = 0
        feature_importances = np.zeros(len(features))
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(f"Group {group_idx}, Fold {fold}: Train size: {X_train.shape}, Test size: {X_test.shape}")
            fold += 1   

            # count positive and negative samples in training set
            num_positive = np.sum(y_train == 1)
            num_negative = np.sum(y_train == 0)
            print(f"Group {group_idx}, Fold {fold}: Training set - Positive samples: {num_positive}, Negative samples: {num_negative}")

            # visualize UMAP & t-SNE of training and testing samples
            sv_fn = os.path.join(output_dir, f"Group{group_idx}_Fold{fold}_umap.png")
            visualize_train_test_sample_UMAP(X_train, y_train, X_test, y_test, save_path=sv_fn)
            sv_fn = os.path.join(output_dir, f"Group{group_idx}_Fold{fold}_tsne.png")
            visualize_train_test_sample_tSNE(X_train, y_train, X_test, y_test, save_path=sv_fn)


            # train and test with xgboost
            model = make_xgb(scale_pos_weight = (num_negative / num_positive))

            model.fit(X_train, y_train)

            feature_importances += model.feature_importances_
            # print training accuracy and auc
            y_train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            try:
                train_auc = roc_auc_score(y_train, y_train_pred)
            except:
                train_auc = None
            print(f"Group {group_idx}, Fold {fold}: Training Accuracy: {train_acc}, Training AUC: {train_auc}")

            # print testing accuracy and auc
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            acc_list.append(acc)
            try:
                auc = roc_auc_score(y_test, y_pred)
            except:
                auc = None
            auc_list.append(auc)
            print(f"Group {group_idx}, Fold {fold}: Accuracy: {acc}, AUC: {auc}")
        
        
        # average results over folds can be computed here
        avg_acc = sum(acc_list) / len(acc_list)
        avg_auc = sum([a for a in auc_list if a is not None]) / len([a for a in auc_list if a is not None]) if any(a is not None for a in auc_list) else None
        
        performance_records["Group"].append(f"Group{group_idx}")
        performance_records["Avg_Accuracy"].append(avg_acc)
        performance_records["Avg_AUC"].append(avg_auc)
        
        avg_feature_importances = feature_importances / len(acc_list)
        feature_importance_dict = dict(zip(features, avg_feature_importances))
        # sort the feature importance dict by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"Sorted feature importances for Group {group_idx}:")
        for feat, imp in sorted_features:
            print(f"  {feat}: {imp}")
        print(f"Group {group_idx}: Average Accuracy: {avg_acc}, Average AUC: {avg_auc}")
        
        # Use only the top 8 features for each group for final model training and evaluation
        print("Use only the top 8 features for each group for final model training and evaluation")
        top_k = 8
        top_features = [feat for feat, imp in sorted_features[:top_k]]
        X_top = df[top_features].values
        X_top = scaler.fit_transform(X_top)
        acc_list_top = []
        auc_list_top = []
        fold = 0
        for train_index, test_index in skf.split(X_top, y):
            fold += 1 
            X_train, X_test = X_top[train_index], X_top[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(f"Group {group_idx}, Fold {fold}: Train size: {X_train.shape}, Test size: {X_test.shape}")
            
            # count positive and negative samples in training set
            num_positive = np.sum(y_train == 1)
            num_negative = np.sum(y_train == 0)
            print(f"Group {group_idx}, Fold {fold}: Training set - Positive samples: {num_positive}, Negative samples: {num_negative}")
            # train and test with xgboost
            model = make_xgb(scale_pos_weight = (num_negative / num_positive))  

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            acc_list_top.append(acc)
            try:
                auc = roc_auc_score(y_test, y_pred)
            except:
                auc = None
            auc_list_top.append(auc)
        avg_acc_top = sum(acc_list_top) / len(acc_list_top)
        avg_auc_top = sum([a for a in auc_list_top if a is not None]) / len([a for a in auc_list_top if a is not None]) if any(a is not None for a in auc_list_top) else None
        
        performance_records["Avg_Accuracy_top_features"].append(avg_acc_top)
        performance_records["Avg_AUC_top_features"].append(avg_auc_top)
        performance_records["Top_Features"].append(top_features)

        print(f"Group {group_idx} with top {top_k} features: Average Accuracy: {avg_acc_top}, Average AUC: {avg_auc_top}")
        print(f"Completed evaluation for Group {group_idx}.")

    # save performance records to a csv file
    performance_df = pd.DataFrame(performance_records)
    performance_df.to_csv(os.path.join(output_dir, "original_data_groupwise_model_performance.csv"), index=False)
    print("Saved groupwise model performance to original_data_groupwise_model_performance.csv")