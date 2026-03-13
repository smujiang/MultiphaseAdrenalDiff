import numpy as np
import pandas as pd
import os
from sklearn.discriminant_analysis import StandardScaler
from group_features import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import visualize_train_test_sample_UMAP, visualize_train_test_sample_tSNE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import shap
from sklearn.base import is_classifier
from sklearn.pipeline import make_pipeline


def _get_shap_values_array(shap_values, positive_class=1):
    """
    Normalize SHAP output shapes across explainers.

    Returns:
      values: (n_samples, n_features) array for the requested class (binary default positive class).
    """
    # Newer SHAP returns an Explanation object with .values
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values  # older API

    # Possible shapes:
    # 1) (n_samples, n_features)  -> already ok
    # 2) (n_samples, n_features, n_classes) -> pick class
    # 3) list of (n_samples, n_features) per class -> pick
    if isinstance(vals, list):
        # list per class
        return np.asarray(vals[positive_class])

    vals = np.asarray(vals)
    if vals.ndim == 2:
        return vals
    if vals.ndim == 3:
        return vals[:, :, positive_class]

    raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")


def shap_global_importance(
    model,
    X: pd.DataFrame,
    model_name: str = "",
    background_size: int = 200,
    nsamples_kernel: int = 200,   # KernelExplainer approximation (SVC non-linear)
    positive_class: int = 1,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Compute SHAP global feature importance (mean(|SHAP|)) for a fitted model on X.

    Parameters
    ----------
    model : fitted sklearn/xgboost model
    X : pd.DataFrame (n_samples, n_features)
    background_size : number of background rows for SHAP (subsampled from X)
    nsamples_kernel : KernelExplainer nsamples for speed/quality tradeoff
    positive_class : class index used for binary-class explanations
    """
    assert isinstance(X, pd.DataFrame), "Pass X as a pandas DataFrame so feature names are preserved."

    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X), size=min(background_size, len(X)), replace=False)
    X_bg = X.iloc[bg_idx]

    # Choose an explainer strategy
    # Tree models (fast)
    tree_types = (
        "XGBClassifier",
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
    )
    model_cls = model.__class__.__name__

    if model_cls in tree_types:
        explainer = shap.TreeExplainer(model)
        shap_exp = explainer(X)  # Explanation

    # Logistic regression / linear models (fast)
    elif model_cls in ("LogisticRegression", "LinearSVC", "SGDClassifier", "RidgeClassifier"):
        # For linear models, SHAP prefers a masker / background.
        # Using Independent masker works well with tabular data.
        masker = shap.maskers.Independent(X_bg)
        explainer = shap.LinearExplainer(model, masker=masker)
        shap_exp = explainer(X)

    # SVC (non-linear kernel) and other black-box models
    else:
        # KernelExplainer needs a prediction function returning probabilities
        if not hasattr(model, "predict_proba"):
            raise ValueError(
                f"{model_cls} has no predict_proba. "
                "For SVC, set probability=True when training, or use decision_function-based SHAP with care."
            )

        f = lambda data: model.predict_proba(pd.DataFrame(data, columns=X.columns))
        explainer = shap.KernelExplainer(f, X_bg, link="logit")
        # KernelExplainer returns numpy arrays or list per class
        shap_vals = explainer.shap_values(X, nsamples=nsamples_kernel)
        # Wrap in a lightweight object-like container
        class Obj: pass
        shap_exp = Obj()
        shap_exp.values = shap_vals

    vals = _get_shap_values_array(shap_exp, positive_class=positive_class)  # (n, p)
    importance = np.mean(np.abs(vals), axis=0)

    out = pd.DataFrame({
        "feature": X.columns,
        "importance": importance,
        "model": model_name if model_name else model_cls
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return out



def make_xgb(scale_pos_weight, seed=42):        
    model = xgb.XGBClassifier(objective="binary:logistic",
                                    eval_metric= ["auc", "aucpr"],
                                    use_label_encoder=False,
                                    scale_pos_weight = scale_pos_weight,
                                    verbosity=0,
                                    n_estimators=300,
                                    max_depth=3,
                                    eta=0.05,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    reg_lambda=2,
                                    reg_alpha=1)
    return model

def make_rf(scale_pos_weight, seed=42):
    model = RandomForestClassifier(n_estimators=300,
                                    max_depth=10,
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    class_weight={0: 1, 1: scale_pos_weight},
                                    random_state=seed,
                                    n_jobs=-1)
    return model

def make_dt(scale_pos_weight, seed=42):
    model = DecisionTreeClassifier(max_depth=10,
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    class_weight={0: 1, 1: scale_pos_weight},
                                    random_state=seed)
    return model

def make_lr(scale_pos_weight, seed=42):
    model = LogisticRegression(penalty='l2',
                                C=1.0,
                                class_weight={0: 1, 1: scale_pos_weight},
                                random_state=seed,
                                max_iter=1000,
                                n_jobs=-1)
    return model

def make_svc(scale_pos_weight, seed=42):
    model = SVC(kernel='rbf',
                C=1.0,
                class_weight={0: 1, 1: scale_pos_weight},
                random_state=seed,
                probability=True)
    return model

if __name__ == "__main__":
    from config import DATA_ROOT_DIR
    import os
    input_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features")
    output_dir = os.path.join(DATA_ROOT_DIR, "processed_data/models_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # only evaluate feature importance for Group 5 which has the complete set of features and enough samples for training 
    group_idx = 5

    df_fn = os.path.join(input_dir, f"Group{group_idx}_features.csv")
    if not os.path.exists(df_fn):
        raise Exception(f"File {df_fn} does not exist. Skipping Group {group_idx}.")
    
    df = pd.read_csv(df_fn)
    features = eval(f"Group{group_idx}_features")
    X = df[features].values
    # Normalize features if needed
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = df["malignancy"].values
    
    if len(y) < 20:
        raise Exception(f"Not enough samples in Group {group_idx} for evaluation. Skipping.")

    # split data into train and test for 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    xgb_acc_list = []
    xgb_auc_list = []
    xgb_feature_importances = np.zeros(len(features))

    rf_acc_list = []
    rf_auc_list = []
    rf_feature_importances = np.zeros(len(features))

    dt_acc_list = []
    dt_auc_list = []
    dt_feature_importances = np.zeros(len(features))

    lr_acc_list = []
    lr_auc_list = []
    lr_feature_importances = np.zeros(len(features))

    svc_acc_list = []
    svc_auc_list = []
    svc_feature_importances = np.zeros(len(features))

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f"Group {group_idx}, Fold {fold}: Train size: {X_train.shape}, Test size: {X_test.shape}")
        fold += 1   

        # count positive and negative samples in training set
        num_positive = np.sum(y_train == 1)
        num_negative = np.sum(y_train == 0)
        print(f"Group {group_idx}, Fold {fold}: Training set - Positive samples: {num_positive}, Negative samples: {num_negative}")

        # # visualize UMAP & t-SNE of training and testing samples
        # sv_fn = os.path.join(output_dir, f"Group{group_idx}_Fold{fold}_umap.png")
        # visualize_train_test_sample_UMAP(X_train, y_train, X_test, y_test, save_path=sv_fn)
        # sv_fn = os.path.join(output_dir, f"Group{group_idx}_Fold{fold}_tsne.png")
        # visualize_train_test_sample_tSNE(X_train, y_train, X_test, y_test, save_path=sv_fn)


        # train and test with xgboost
        xgb_model = make_xgb(scale_pos_weight=(num_negative / num_positive))
        xgb_model.fit(X_train, y_train)

        xgb_feature_importances += xgb_model.feature_importances_
        # train and test with random forest
        rf_model = make_rf(scale_pos_weight=(num_negative / num_positive))
        rf_model.fit(X_train, y_train)

        rf_feature_importances += rf_model.feature_importances_
        # train and test with decision tree
        dt_model = make_dt(scale_pos_weight=(num_negative / num_positive))
        dt_model.fit(X_train, y_train)
        dt_feature_importances += dt_model.feature_importances_

        # train and test with logistic regression
        lr_model = make_lr(scale_pos_weight=(num_negative / num_positive))
        lr_model.fit(X_train, y_train)
        lr_feature_importances += np.abs(lr_model.coef_[0])

        # train and test with SVM
        svc_model = make_svc(scale_pos_weight=(num_negative / num_positive))
        svc_model.fit(X_train, y_train)
        svc_feature_importances += np.abs(svc_model.coef0)

        # print XGBoost training accuracy and auc
        xgb_y_train_pred = xgb_model.predict(X_train)
        xgb_train_acc = accuracy_score(y_train, xgb_y_train_pred)
        try:
            xgb_train_auc = roc_auc_score(y_train, xgb_y_train_pred)
        except:
            xgb_train_auc = None
        print(f"XGBoost Group {group_idx}, Fold {fold}: Training Accuracy: {xgb_train_acc}, Training AUC: {xgb_train_auc}")

        # print XGBoost testing accuracy and auc
        xgb_y_test_pred = xgb_model.predict(X_test)
        xgb_test_acc = accuracy_score(y_test, xgb_y_test_pred)
        xgb_acc_list.append(xgb_test_acc)
        try:
            xgb_test_auc = roc_auc_score(y_test, xgb_y_test_pred)
        except:
            xgb_test_auc = None
        xgb_auc_list.append(xgb_test_auc)
        print(f"XGBoost Group {group_idx}, Fold {fold}: Accuracy: {xgb_test_acc}, AUC: {xgb_test_auc}")

        # print random forest training accuracy and auc
        rf_y_train_pred = rf_model.predict(X_train)
        rf_train_acc = accuracy_score(y_train, rf_y_train_pred)
        try:
            rf_train_auc = roc_auc_score(y_train, rf_y_train_pred)
        except:
            rf_train_auc = None
        print(f"Random Forest Group {group_idx}, Fold {fold}: Training Accuracy: {rf_train_acc}, Training AUC: {rf_train_auc}")

        # print random forest testing accuracy and auc
        rf_y_test_pred = rf_model.predict(X_test)
        rf_test_acc = accuracy_score(y_test, rf_y_test_pred)
        rf_acc_list.append(rf_test_acc)
        try:
            rf_test_auc = roc_auc_score(y_test, rf_y_test_pred)
        except:
            rf_test_auc = None
        rf_auc_list.append(rf_test_auc)
        print(f"Random Forest Group {group_idx}, Fold {fold}: Accuracy: {rf_test_acc}, AUC: {rf_test_auc}")
    
        # print decision tree training accuracy and auc
        dt_y_train_pred = dt_model.predict(X_train)
        dt_train_acc = accuracy_score(y_train, dt_y_train_pred)
        try:
            dt_train_auc = roc_auc_score(y_train, dt_y_train_pred)
        except:
            dt_train_auc = None
        print(f"Decision Tree Group {group_idx}, Fold {fold}: Training Accuracy: {dt_train_acc}, Training AUC: {dt_train_auc}")

        # print decision tree testing accuracy and auc
        dt_y_test_pred = dt_model.predict(X_test)
        dt_test_acc = accuracy_score(y_test, dt_y_test_pred)
        dt_acc_list.append(dt_test_acc)
        try:            
            dt_test_auc = roc_auc_score(y_test, dt_y_test_pred)
        except:
            dt_test_auc = None
        dt_auc_list.append(dt_test_auc)
        print(f"Decision Tree Group {group_idx}, Fold {fold}: Accuracy: {dt_test_acc}, AUC: {dt_test_auc}")
        # print logistic regression training accuracy and auc
        lr_y_train_pred = lr_model.predict(X_train)
        lr_train_acc = accuracy_score(y_train, lr_y_train_pred)
        try:
            lr_train_auc = roc_auc_score(y_train, lr_y_train_pred)
        except:
            lr_train_auc = None
        print(f"Logistic Regression Group {group_idx}, Fold {fold}: Training Accuracy: {lr_train_acc}, Training AUC: {lr_train_auc}")
        # print logistic regression testing accuracy and auc
        lr_y_test_pred = lr_model.predict(X_test)
        lr_test_acc = accuracy_score(y_test, lr_y_test_pred)
        lr_acc_list.append(lr_test_acc)
        try:
            lr_test_auc = roc_auc_score(y_test, lr_y_test_pred)
        except:
            lr_test_auc = None
        lr_auc_list.append(lr_test_auc)
        print(f"Logistic Regression Group {group_idx}, Fold {fold}: Accuracy: {lr_test_acc}, AUC: {lr_test_auc}")
        # print SVM training accuracy and auc
        svc_y_train_pred = svc_model.predict(X_train)
        svc_train_acc = accuracy_score(y_train, svc_y_train_pred)
        try:
            svc_train_auc = roc_auc_score(y_train, svc_y_train_pred)
        except:
            svc_train_auc = None
        print(f"SVM Group {group_idx}, Fold {fold}: Training Accuracy: {svc_train_acc}, Training AUC: {svc_train_auc}")
        # print SVM testing accuracy and auc
        svc_y_test_pred = svc_model.predict(X_test)
        svc_test_acc = accuracy_score(y_test, svc_y_test_pred)
        svc_acc_list.append(svc_test_acc)
        try:        
            svc_test_auc = roc_auc_score(y_test, svc_y_test_pred)
        except:
            svc_test_auc = None
        svc_auc_list.append(svc_test_auc)
        print(f"SVM Group {group_idx}, Fold {fold}: Accuracy: {svc_test_acc}, AUC: {svc_test_auc}")

    # save feature importance to csv
    feature_importance_df = pd.DataFrame({
        "feature": features,
        "xgb_importance": xgb_feature_importances / len(xgb_acc_list),
        "rf_importance": rf_feature_importances / len(rf_acc_list),
        "dt_importance": dt_feature_importances / len(dt_acc_list),
        "lr_importance": lr_feature_importances / len(lr_acc_list),
        "svc_importance": svc_feature_importances / len(svc_acc_list),
    })
    feature_importance_csv_fn = os.path.join(output_dir, f"Group{group_idx}_feature_importances.csv")
    feature_importance_df.to_csv(feature_importance_csv_fn, index=False)
    print(f"Saved feature importances to {feature_importance_csv_fn}")
    
    # average results over folds can be computed here
    avg_acc = sum(xgb_acc_list) / len(xgb_acc_list)
    avg_auc = sum([a for a in xgb_auc_list if a is not None]) / len([a for a in xgb_auc_list if a is not None]) if any(a is not None for a in xgb_auc_list) else None
    avg_xgb_feature_importances = xgb_feature_importances / len(xgb_acc_list)
    feature_importance_dict = dict(zip(features, avg_xgb_feature_importances))
    # sort the feature importance dict by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"Sorted feature importances for Group {group_idx}:")
    for feat, imp in sorted_features:
        print(f"  {feat}: {imp}")
    print(f"XGBoost Group {group_idx}: Average Accuracy: {avg_acc}, Average AUC: {avg_auc}")
    
    # Get the top 10 features for XGBoost and print them
    top_k = 10
    top_features = [feat for feat, imp in sorted_features[:top_k]]
    X_top = df[top_features].values
    print(f"Top {top_k} features for Group {group_idx}: {top_features}")

    # print feature importance for random forest
    avg_rf_feature_importances = rf_feature_importances / len(rf_acc_list)
    rf_feature_importance_dict = dict(zip(features, avg_rf_feature_importances))
    # print the feature importance that highlight in the XGBoost model
    print(f"Random Forest feature importances for Group {group_idx} (highlighting XGBoost top features):")
    for feat in features:
        imp = rf_feature_importance_dict[feat]
        highlight = " <-- XGBoost Top Feature" if feat in top_features else ""
        print(f"  {feat}: {imp}{highlight}")
    
    # print feature importance for decision tree
    avg_dt_feature_importances = dt_feature_importances / len(dt_acc_list)
    dt_feature_importance_dict = dict(zip(features, avg_dt_feature_importances))
    print(f"Decision Tree feature importances for Group {group_idx} (highlighting XGBoost top features):")
    for feat in features:
        imp = dt_feature_importance_dict[feat]
        highlight = " <-- XGBoost Top Feature" if feat in top_features else ""
        print(f"  {feat}: {imp}{highlight}")
    # print feature importance for logistic regression
    avg_lr_feature_importances = lr_feature_importances / len(lr_acc_list)
    lr_feature_importance_dict = dict(zip(features, avg_lr_feature_importances))
    print(f"Logistic Regression feature importances for Group {group_idx} (highlighting XGBoost top features):")
    for feat in features:
        imp = lr_feature_importance_dict[feat]
        highlight = " <-- XGBoost Top Feature" if feat in top_features else ""
        print(f"  {feat}: {imp}{highlight}")
    # print feature importance for SVM
    avg_svc_feature_importances = svc_feature_importances / len(svc_acc_list)
    svc_feature_importance_dict = dict(zip(features, avg_svc_feature_importances))
    print(f"SVM feature importances for Group {group_idx} (highlighting XGBoost top features):")
    for feat in features:
        imp = svc_feature_importance_dict[feat]
        highlight = " <-- XGBoost Top Feature" if feat in top_features else ""
        print(f"  {feat}: {imp}{highlight}")
    
    # normalize feature importance to relative importance for better visualization
    f_max = feature_importance_df["xgb_importance"].max()
    feature_importance_df["xgb_importance"] = feature_importance_df["xgb_importance"] / feature_importance_df["xgb_importance"].max()
    feature_importance_df["rf_importance"] = feature_importance_df["rf_importance"] / feature_importance_df["rf_importance"].max()
    feature_importance_df["dt_importance"] = feature_importance_df["dt_importance"] / feature_importance_df["dt_importance"].max()
    feature_importance_df["lr_importance"] = feature_importance_df["lr_importance"] / feature_importance_df["lr_importance"].max()
    feature_importance_df["svc_importance"] = feature_importance_df["svc_importance"] / feature_importance_df["svc_importance"].max()
    # sort features by XGBoost importance for better visualization
    feature_importance_df = feature_importance_df.sort_values("xgb_importance", ascending=False).reset_index(drop=True)

    # plot a heatmap of feature importance across models
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_importance_melted = feature_importance_df.melt(id_vars="feature", var_name="model", value_name="importance")
    plt.figure(figsize=(10, 8))
    # plot a heatmap with seaborn, with feature on y-axis and model on x-axis, and importance as color, and also show the importance value on each cell
    sns.heatmap(feature_importance_df.set_index("feature").T, annot=False, cmap="Reds", cbar_kws={'label': 'Normalized Feature Importance'})
    plt.title(f"Feature Importances Analysis")
    plt.tight_layout()
    heatmap_fn = os.path.join(output_dir, f"Group{group_idx}_feature_importance_comparison.png")
    plt.savefig(heatmap_fn)
    print(f"Saved feature importance comparison heatmap to {heatmap_fn}")
    plt.show()

