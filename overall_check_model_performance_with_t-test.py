# To prove that model performance can be different if the data missingness was handled differently.
# For example, the XGBoost, Random Forest, Decision Tree can deal with data missingness by design, 
# while Logistic Regression and SVM cannot handle missing data and require imputation or other handling strategies.
# 

# Use the data with all the samples no matter which imaging phases are available, put all the features into one table. 
# Train and evaluate the ML models on this table, and compare the performance with the models trained on the data with only samples with all phases available.

from pyexpat import features
import numpy as np
import pandas as pd
import os
import shap
from sklearn.discriminant_analysis import StandardScaler
from group_features import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import visualize_train_test_sample_UMAP, visualize_train_test_sample_tSNE
from matplotlib import pyplot as plt
from models import make_xgb, make_rf, make_dt, make_lr, make_svc, ML_model_list
from scipy import stats


# add t-test to compare the performance of models trained on data with median filling vs. original data with missing values, to see if the difference is statistically significant.

if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    all_normalized_data_median_filled_fn = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_unified/all_groups_features_unified_normalized_medianfilled.csv")
    all_normalized_data_fn = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_unified/all_groups_features_unified_normalized.csv")
    output_dir = os.path.join(DATA_ROOT_DIR, "processed_data/overall_eval")

    output_csv = os.path.join(output_dir, "Overall_model_results_all_samples_multiple_models.csv")
    performance_records = {"Data": [], "Model": [], "Avg_Accuracy": [], "Avg_AUC": []}
    # Store individual fold results for statistical testing
    fold_results = {}
    # Store individual fold results for statistical testing
    fold_results = {}
    # Store individual fold results for statistical testing
    fold_results = {}
    for all_data_fn in [all_normalized_data_median_filled_fn, all_normalized_data_fn]:
        if "medianfilled" in all_data_fn:
            data_label = "Median_Filled"
        else:
            data_label = "Original"
        
        df = pd.read_csv(all_data_fn)
        all_features = [col for col in df.columns if col not in ["MRN", "StudyDate", "group", "available_phases", "malignancy"]]
        X = df[all_features].values
        y = df["malignancy"].values
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if data_label == "Median_Filled":
            print(f"Evaluating all models on data with median filling (imputation) for missing values...")
            for model_name in ML_model_list:
                print(f"Evaluating model: {model_name} ")
                fold = 0
                acc_list = []
                auc_list = []
                feature_importances = np.zeros(len(all_features))
                for train_index, test_index in skf.split(X, y):
                    fold += 1 
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # print(f"Fold {fold}: Train size: {X_train.shape}, Test size: {X_test.shape}")
                    
                    # count positive and negative samples in training set
                    num_positive = np.sum(y_train == 1)
                    num_negative = np.sum(y_train == 0)
                    # print(f"Fold {fold}: Training set - Positive samples: {num_positive}, Negative samples: {num_negative}")

                    if model_name == "xgboost":
                        model = make_xgb(scale_pos_weight = (num_negative / num_positive), seed=42)
                    elif model_name == "random_forest": 
                        model = make_rf(scale_pos_weight = (num_negative / num_positive), seed=42)
                    elif model_name == "decision_tree":
                        model = make_dt(scale_pos_weight = (num_negative / num_positive), seed=42)
                    elif model_name == "logistic_regression":
                        model = make_lr(scale_pos_weight = (num_negative / num_positive), seed=42)
                    elif model_name == "svm":
                        model = make_svc(scale_pos_weight = (num_negative / num_positive), seed=42)

                    model.fit(X_train, y_train)

                    # print testing accuracy and auc
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_pred)
                    except:
                        auc = None
                    # print(f"Fold {fold}, Model {model_name}: Accuracy: {acc}, AUC: {auc}")
                    acc_list.append(acc)
                    auc_list.append(auc)
                avg_acc = sum(acc_list) / len(acc_list)
                avg_auc = sum([a for a in auc_list if a is not None]) / len([a for a in auc_list if a is not None]) if any(a is not None for a in auc_list) else None
                performance_records["Avg_Accuracy"].append(avg_acc)
                performance_records["Avg_AUC"].append(avg_auc)
                performance_records["Model"].append(model_name)
                performance_records["Data"].append(data_label)
                
                # Store individual fold results for statistical testing
                key = f"{model_name}_{data_label}"
                fold_results[key] = {"accuracy": acc_list, "auc": [a for a in auc_list if a is not None]}
        else:
            print(f"Evaluating models that can handle missing data on original data ")
            for model_name in ["xgboost", "random_forest", "decision_tree"]:
            # for model_name in ML_model_list:
                print(f"Evaluating model: {model_name} ")
                fold = 0
                acc_list = []
                auc_list = []
                feature_importances = np.zeros(len(all_features))
                for train_index, test_index in skf.split(X, y):
                    fold += 1 
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    # print(f"Fold {fold}: Train size: {X_train.shape}, Test size: {X_test.shape}")
                    
                    # count positive and negative samples in training set
                    num_positive = np.sum(y_train == 1)
                    num_negative = np.sum(y_train == 0)
                    # print(f"Fold {fold}: Training set - Positive samples: {num_positive}, Negative samples: {num_negative}")

                    if model_name == "xgboost":
                        model = make_xgb(scale_pos_weight = (num_negative / num_positive), seed=42)
                    elif model_name == "random_forest": 
                        model = make_rf(scale_pos_weight = (num_negative / num_positive), seed=42)
                    elif model_name == "decision_tree":
                        model = make_dt(scale_pos_weight = (num_negative / num_positive), seed=42)
                    elif model_name == "logistic_regression":
                        model = make_lr(scale_pos_weight = (num_negative / num_positive), seed=42)
                    elif model_name == "svm":
                        model = make_svc(scale_pos_weight = (num_negative / num_positive), seed=42)


                    model.fit(X_train, y_train)

                    # print testing accuracy and auc
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_pred)
                    except:
                        auc = None
                    # print(f"Fold {fold}, Model {model_name}: Accuracy: {acc}, AUC: {auc}")
                    acc_list.append(acc)
                    auc_list.append(auc)

                avg_acc = sum(acc_list) / len(acc_list)
                avg_auc = sum([a for a in auc_list if a is not None]) / len([a for a in auc_list if a is not None]) if any(a is not None for a in auc_list) else None
                performance_records["Avg_Accuracy"].append(avg_acc)
                performance_records["Avg_AUC"].append(avg_auc)
                performance_records["Model"].append(model_name)
                performance_records["Data"].append(data_label)
                
                # Store individual fold results for statistical testing
                key = f"{model_name}_{data_label}"
                fold_results[key] = {"accuracy": acc_list, "auc": [a for a in auc_list if a is not None]}
            
    results_df = pd.DataFrame(performance_records)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved overall results to {output_csv}")
    
    # Perform statistical tests comparing median-filled vs original data
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON: Median Filled vs Original Data")
    print("="*60)
    
    # Models that can be compared (available in both datasets)
    comparable_models = ["xgboost", "random_forest", "decision_tree"]
    
    # Store statistical test results
    stat_test_results = []
    
    for model_name in comparable_models:
        median_key = f"{model_name}_Median_Filled"
        original_key = f"{model_name}_Original"
        
        if median_key in fold_results and original_key in fold_results:
            print(f"\n{model_name.upper()} Model Comparison:")
            print("-" * 40)
            
            # Compare accuracy
            median_acc = fold_results[median_key]["accuracy"]
            original_acc = fold_results[original_key]["accuracy"]
            
            if len(median_acc) == len(original_acc) and len(median_acc) > 1:
                t_stat_acc, p_val_acc = stats.ttest_rel(median_acc, original_acc)
                print(f"Accuracy - Median Filled: {np.mean(median_acc):.4f} ± {np.std(median_acc):.4f}")
                print(f"Accuracy - Original: {np.mean(original_acc):.4f} ± {np.std(original_acc):.4f}")
                print(f"Paired t-test (Accuracy): t={t_stat_acc:.4f}, p={p_val_acc:.4f}")
                
                if p_val_acc < 0.05:
                    print(f"*** Statistically significant difference in accuracy (p < 0.05) ***")
                else:
                    print(f"No statistically significant difference in accuracy (p ≥ 0.05)")
                    
                stat_test_results.append({
                    "Model": model_name,
                    "Metric": "Accuracy",
                    "Median_Filled_Mean": np.mean(median_acc),
                    "Median_Filled_Std": np.std(median_acc),
                    "Original_Mean": np.mean(original_acc),
                    "Original_Std": np.std(original_acc),
                    "t_statistic": t_stat_acc,
                    "p_value": p_val_acc,
                    "Significant": p_val_acc < 0.05
                })
            
            # Compare AUC if available
            median_auc = fold_results[median_key]["auc"]
            original_auc = fold_results[original_key]["auc"]
            
            if len(median_auc) == len(original_auc) and len(median_auc) > 1:
                t_stat_auc, p_val_auc = stats.ttest_rel(median_auc, original_auc)
                print(f"AUC - Median Filled: {np.mean(median_auc):.4f} ± {np.std(median_auc):.4f}")
                print(f"AUC - Original: {np.mean(original_auc):.4f} ± {np.std(original_auc):.4f}")
                print(f"Paired t-test (AUC): t={t_stat_auc:.4f}, p={p_val_auc:.4f}")
                
                if p_val_auc < 0.05:
                    print(f"*** Statistically significant difference in AUC (p < 0.05) ***")
                else:
                    print(f"No statistically significant difference in AUC (p ≥ 0.05)")
                    
                stat_test_results.append({
                    "Model": model_name,
                    "Metric": "AUC",
                    "Median_Filled_Mean": np.mean(median_auc),
                    "Median_Filled_Std": np.std(median_auc),
                    "Original_Mean": np.mean(original_auc),
                    "Original_Std": np.std(original_auc),
                    "t_statistic": t_stat_auc,
                    "p_value": p_val_auc,
                    "Significant": p_val_auc < 0.05
                })
    
    # Save statistical test results
    if stat_test_results:
        stat_df = pd.DataFrame(stat_test_results)
        stat_output_csv = os.path.join(output_dir, "Statistical_comparison_median_vs_original.csv")
        stat_df.to_csv(stat_output_csv, index=False)
        print(f"\nSaved statistical test results to {stat_output_csv}")
        
        # Summary of significant differences
        significant_results = stat_df[stat_df["Significant"] == True]
        if len(significant_results) > 0:
            print(f"\nSUMMARY: Found {len(significant_results)} statistically significant differences:")
            for _, row in significant_results.iterrows():
                print(f"  - {row['Model']} ({row['Metric']}): p = {row['p_value']:.4f}")
        else:
            print(f"\nSUMMARY: No statistically significant differences found between median-filled and original data.")
    
    print("\n" + "="*60)


