import pandas as pd
import os
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from group_features import *
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from utils import visualize_train_test_sample_UMAP, visualize_train_test_sample_tSNE
from models import make_xgb, make_rf, make_dt, make_lr, make_svc, ML_model_list



if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    input_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features_augmented")
    output_dir = os.path.join(DATA_ROOT_DIR, "processed_data/models_output_augmented")
    os.makedirs(output_dir, exist_ok=True)

    model_list = ML_model_list
    output_csv = os.path.join(output_dir, "GroupWise_model_results_oversampled_multiple_models.csv")
    performance_records = {"Group": [], "Model": [], "Avg_Accuracy": [], "Std_Accuracy": [], "Avg_AUC": [], "Std_AUC": []}
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
            std_acc = np.std(acc_list)
            avg_auc = sum([a for a in auc_list if a is not None]) / len([a for a in auc_list if a is not None]) if any(a is not None for a in auc_list) else None
            std_auc = np.std([a for a in auc_list if a is not None]) if any(a is not None for a in auc_list) else None
            print(f"Group {group_idx}, Model {model_name}: Average Accuracy: {avg_acc} ± {std_acc}, Average AUC: {avg_auc} ± {std_auc}")
            performance_records["Group"].append(f"Group{group_idx}")
            performance_records["Avg_Accuracy"].append(avg_acc)
            performance_records["Std_Accuracy"].append(std_acc)
            performance_records["Avg_AUC"].append(avg_auc)
            performance_records["Std_AUC"].append(std_auc)
            performance_records["Model"].append(model_name)


        results_df = pd.DataFrame(performance_records)
        results_df.to_csv(output_csv, index=False)
        print(f"Saved overall results to {output_csv}")


