import pandas as pd
import os
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from group_features import *
import shap
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from utils import visualize_train_test_sample_UMAP, visualize_train_test_sample_tSNE
from models import make_xgb, make_rf, make_dt, make_lr, make_svc, ML_model_list

# add t-test to compare the performance of different models on each group, and save the p-values to a csv file


if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    input_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_augmented")
    output_dir = os.path.join(DATA_ROOT_DIR, "processed_data/models_output_augmented")
    os.makedirs(output_dir, exist_ok=True)

    model_list = ML_model_list
    output_csv = os.path.join(output_dir, "GroupWise_model_results_oversampled_multiple_models.csv")
    performance_records = {"Group": [], "Model": [], "Avg_Accuracy": [], "Avg_AUC": []}
    
    # Store fold results for statistical testing
    all_fold_results = {}  # {group: {model: {accuracy: [...], auc: [...]}}}
    t_test_results = []  # Store t-test results for all groups
    for group_idx in range(1, 16):
        print(f"Evaluating Group {group_idx} with oversampled data...")
        df = pd.read_csv(os.path.join(input_dir, f"Group{group_idx}_features_augmented.csv"))
        features = eval(f"Group{group_idx}_features")
        X = df[features].values
        # Normalize features if needed
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = df["malignancy"].values
        # split data into train and test for 5-fold cross-validation

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize fold results for this group
        group_fold_results = {}

        # use different models
        for model_name in model_list:
            fold = 0
            acc_list = []
            auc_list = []
            feature_importances = np.zeros(len(features))
            for train_index, test_index in skf.split(X, y):
                fold += 1 
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                print(f"Group {group_idx}, Fold {fold}: Train size: {X_train.shape}, Test size: {X_test.shape}")
                
                # count positive and negative samples in training set
                num_positive = np.sum(y_train == 1)
                num_negative = np.sum(y_train == 0)
                print(f"Group {group_idx}, Fold {fold}: Training set - Positive samples: {num_positive}, Negative samples: {num_negative}")

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
                print(f"Group {group_idx}, Fold {fold}, Model {model_name}: Accuracy: {acc}, AUC: {auc}")
                acc_list.append(acc)
                auc_list.append(auc)

                # SHAP analysis for tree-based models
                if model_name in {"xgboost", "random_forest", "decision_tree"}:
                    expl_ = shap.TreeExplainer(model)
                    shap_values= expl_.shap_values(X_test)
                    sv_fn = os.path.join(output_dir, f"Group{group_idx}_Fold{fold}_{model_name}_shap_summary.png")
                    shap.summary_plot(shap_values, X_test, feature_names=features, show=False, plot_size=(10,7))
                    plt.savefig(sv_fn)
                    plt.close()
                    
                    # if shap_values is 3D (e.g. for multi-class), take first dim as feature importance
                    if dimensions := len(shap_values.shape) == 3:
                        shap_values = np.array(shap_values)[:, :, 1] 
                        
                    shap_values_df = pd.DataFrame(shap_values, columns=features)
                    shap_values_df["True_Label"] = y_test
                    shap_values_fn = os.path.join(output_dir, f"Group{group_idx}_Fold{fold}_{model_name}_shap_values.csv")
                    shap_values_df.to_csv(shap_values_fn, index=False)
                    print(f"Saved SHAP values to {shap_values_fn}")
                else:
                    print(f"SHAP analysis not supported for model: {model_name}")
                
            # average results over folds can be computed here
            avg_acc = sum(acc_list) / len(acc_list)
            avg_auc = sum([a for a in auc_list if a is not None]) / len([a for a in auc_list if a is not None]) if any(a is not None for a in auc_list) else None
            
            performance_records["Group"].append(f"Group{group_idx}")
            performance_records["Avg_Accuracy"].append(avg_acc)
            performance_records["Avg_AUC"].append(avg_auc)
            performance_records["Model"].append(model_name)
            
            # Store fold results for statistical testing
            group_fold_results[model_name] = {
                "accuracy": acc_list,
                "auc": [a for a in auc_list if a is not None]
            }
        
        # Store group results
        all_fold_results[f"Group{group_idx}"] = group_fold_results
        
        # Perform pairwise t-tests between models for this group
        print(f"\nPerforming t-tests for Group {group_idx}...")
        model_combinations = [(i, j) for i in range(len(model_list)) for j in range(i+1, len(model_list))]
        
        for i, j in model_combinations:
            model1, model2 = model_list[i], model_list[j]
            
            if model1 in group_fold_results and model2 in group_fold_results:
                # T-test for accuracy
                acc1 = group_fold_results[model1]["accuracy"]
                acc2 = group_fold_results[model2]["accuracy"]
                
                if len(acc1) == len(acc2) and len(acc1) > 1:
                    t_stat_acc, p_val_acc = stats.ttest_rel(acc1, acc2)
                    
                    t_test_results.append({
                        "Group": f"Group{group_idx}",
                        "Model1": model1,
                        "Model2": model2,
                        "Metric": "Accuracy",
                        "Model1_Mean": np.mean(acc1),
                        "Model1_Std": np.std(acc1),
                        "Model2_Mean": np.mean(acc2),
                        "Model2_Std": np.std(acc2),
                        "t_statistic": t_stat_acc,
                        "p_value": p_val_acc,
                        "Significant": p_val_acc < 0.05
                    })
                    
                    print(f"  {model1} vs {model2} (Accuracy): p = {p_val_acc:.4f} {'*' if p_val_acc < 0.05 else ''}")
                
                # T-test for AUC
                auc1 = group_fold_results[model1]["auc"]
                auc2 = group_fold_results[model2]["auc"]
                
                if len(auc1) == len(auc2) and len(auc1) > 1:
                    t_stat_auc, p_val_auc = stats.ttest_rel(auc1, auc2)
                    
                    t_test_results.append({
                        "Group": f"Group{group_idx}",
                        "Model1": model1,
                        "Model2": model2,
                        "Metric": "AUC",
                        "Model1_Mean": np.mean(auc1),
                        "Model1_Std": np.std(auc1),
                        "Model2_Mean": np.mean(auc2),
                        "Model2_Std": np.std(auc2),
                        "t_statistic": t_stat_auc,
                        "p_value": p_val_auc,
                        "Significant": p_val_auc < 0.05
                    })
                    
                    print(f"  {model1} vs {model2} (AUC): p = {p_val_auc:.4f} {'*' if p_val_auc < 0.05 else ''}")


    # Save performance results
    results_df = pd.DataFrame(performance_records)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved overall results to {output_csv}")
    
    # Save t-test results
    if t_test_results:
        t_test_df = pd.DataFrame(t_test_results)
        t_test_output_csv = os.path.join(output_dir, "GroupWise_model_ttest_results.csv")
        t_test_df.to_csv(t_test_output_csv, index=False)
        print(f"Saved t-test results to {t_test_output_csv}")
        
        # Summary of significant differences
        significant_results = t_test_df[t_test_df["Significant"] == True]
        print(f"\n" + "="*60)
        print(f"T-TEST SUMMARY: Found {len(significant_results)} statistically significant differences (p < 0.05)")
        print("="*60)
        
        if len(significant_results) > 0:
            for _, row in significant_results.iterrows():
                print(f"{row['Group']}: {row['Model1']} vs {row['Model2']} ({row['Metric']}) - p = {row['p_value']:.4f}")
        else:
            print("No statistically significant differences found between models within any group.")
            
        # Group-wise summary
        print(f"\nGROUP-WISE SIGNIFICANT DIFFERENCES:")
        print("-" * 40)
        for group_idx in range(1, 16):
            group_name = f"Group{group_idx}"
            group_significant = significant_results[significant_results["Group"] == group_name]
            print(f"{group_name}: {len(group_significant)} significant differences")
            
        print("\n" + "="*60)


