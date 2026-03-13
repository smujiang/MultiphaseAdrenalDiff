import numpy as np
import pandas as pd
import os
from sklearn.discriminant_analysis import StandardScaler
from group_features import *
from sklearn.model_selection import StratifiedKFold, train_test_split
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
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})  

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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X), size=min(background_size, len(X)), replace=False)
    X_bg = X_scaled[bg_idx]

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
        shap_exp = explainer(X_scaled)  # Explanation
    # Logistic regression / linear models (fast)
    elif model_cls in ("LogisticRegression", "LinearSVC", "SGDClassifier", "RidgeClassifier"):
        # For linear models, SHAP prefers a masker / background.
        # Using Independent masker works well with tabular data.
        masker = shap.maskers.Independent(X_bg)
        explainer = shap.LinearExplainer(model, masker=masker)
        shap_exp = explainer(X_scaled)
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
        shap_vals = explainer.shap_values(X_scaled, nsamples=nsamples_kernel)
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
    input_dir = os.path.join(DATA_ROOT_DIR, "processed_data/grouped_instance_features")
    output_dir = os.path.join(DATA_ROOT_DIR, "processed_data/models_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # only evaluate feature importance for Group 5 which has the complete set of features and enough samples for training 
    # group_idx = 7

    group_idx = 5

    df_fn = os.path.join(input_dir, f"Group{group_idx}_features.csv")
    if not os.path.exists(df_fn):
        raise Exception(f"File {df_fn} does not exist. Skipping Group {group_idx}.")
    
    df = pd.read_csv(df_fn)
    features = eval(f"Group{group_idx}_features")
    X = df[features]
    # Normalize features if needed
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    y = df["malignancy"].values
    
    if len(y) < 20:
        raise Exception(f"Not enough samples in Group {group_idx} for evaluation. Skipping.")

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    
    # count positive and negative samples in training set
    num_positive = np.sum(y_train == 1)
    num_negative = np.sum(y_train == 0)
    print(f"Group {group_idx}: Training set - Positive samples: {num_positive}, Negative samples: {num_negative}")

    # train and test with xgboost
    xgb_model = make_xgb(scale_pos_weight=(num_negative / num_positive))
    xgb_model.fit(X_train, y_train)

    # train and test with random forest
    rf_model = make_rf(scale_pos_weight=(num_negative / num_positive))
    rf_model.fit(X_train, y_train)

    # train and test with decision tree
    dt_model = make_dt(scale_pos_weight=(num_negative / num_positive))
    dt_model.fit(X_train, y_train)

    # train and test with logistic regression
    lr_model = make_lr(scale_pos_weight=(num_negative / num_positive))
    lr_model.fit(X_train, y_train)

    # train and test with SVM
    svc_model = make_svc(scale_pos_weight=(num_negative / num_positive))
    svc_model.fit(X_train, y_train)

    imp_xgb = shap_global_importance(xgb_model, X_test, model_name="XGBoost")
    imp_rf  = shap_global_importance(rf_model,  X_test, model_name="RandomForest")
    imp_dt  = shap_global_importance(dt_model,  X_test, model_name="DecisionTree")
    imp_lr  = shap_global_importance(lr_model,  X_test, model_name="LogisticRegression")
    imp_svc = shap_global_importance(svc_model, X_test, model_name="SVC-RBF", nsamples_kernel=150)

    test_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    test_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    test_auc_dt = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])
    test_auc_lr = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
    test_auc_svc = roc_auc_score(y_test, svc_model.predict_proba(X_test)[:, 1])


    print(f"Group {group_idx} Test AUC: XGBoost: {test_auc_xgb:.4f}, RandomForest: {test_auc_rf:.4f}, DecisionTree: {test_auc_dt:.4f}, LogisticRegression: {test_auc_lr:.4f}, SVC-RBF: {test_auc_svc:.4f}")

    all_imp = pd.concat([imp_xgb, imp_rf, imp_dt, imp_lr, imp_svc], ignore_index=True)

    feature_importance_csv_fn = os.path.join(output_dir, f"Group{group_idx}_feature_importances.csv")
    all_imp.to_csv(feature_importance_csv_fn, index=False)
    print(f"Saved feature importances to {feature_importance_csv_fn}")
    
    all_imp["rel_importance"] = all_imp.groupby("model")["importance"].transform(lambda s: s / (s.sum() + 1e-12))
    feature_importance_comparison_csv_fn = os.path.join(output_dir, f"Group{group_idx}_feature_importance_comparison.csv")
    all_imp.to_csv(feature_importance_comparison_csv_fn, index=False)
    print(f"Saved feature importance comparison to {feature_importance_comparison_csv_fn}")

    sorted_features = imp_xgb.sort_values("importance", ascending=False)[["feature", "importance"]].values.tolist()

    # Get the top 10 features for XGBoost and print them
    top_k = 20
    top_features = [feat for feat, imp in sorted_features[:top_k]]
    X_top = df[top_features].values
    print(f"Top {top_k} features for Group {group_idx}: {top_features}")

    # sort feature importance according to XGBoost for random forest
    rf_feature_importances = imp_rf.set_index("feature").loc[top_features]["importance"].values
    rf_feature_importance_dict = dict(zip(top_features, rf_feature_importances))
    # print the feature importance that highlight in the XGBoost model
    print(f"Random Forest feature importances for Group {group_idx} (highlighting XGBoost top features):")
    for feat in top_features:
        imp = rf_feature_importance_dict[feat]
        highlight = " <-- XGBoost Top Feature" if feat in top_features else ""
        print(f"  {feat}: {imp}{highlight}")
    
    # print feature importance for decision tree
    dt_feature_importances = imp_dt.set_index("feature").loc[top_features]["importance"].values
    dt_feature_importance_dict = dict(zip(top_features, dt_feature_importances))
    print(f"Decision Tree feature importances for Group {group_idx} (highlighting XGBoost top features):")
    for feat in top_features:
        imp = dt_feature_importance_dict[feat]
        highlight = " <-- XGBoost Top Feature" if feat in top_features else ""
        print(f"  {feat}: {imp}{highlight}")
    # print feature importance for logistic regression
    lr_feature_importances = imp_lr.set_index("feature").loc[top_features]["importance"].values
    lr_feature_importance_dict = dict(zip(top_features, lr_feature_importances))
    print(f"Logistic Regression feature importances for Group {group_idx} (highlighting XGBoost top features):")
    for feat in top_features:
        imp = lr_feature_importance_dict[feat]
        highlight = " <-- XGBoost Top Feature" if feat in top_features else ""
        print(f"  {feat}: {imp}{highlight}")
    # print feature importance for SVM
    svc_feature_importances = imp_svc.set_index("feature").loc[top_features]["importance"].values
    svc_feature_importance_dict = dict(zip(top_features, svc_feature_importances))
    print(f"SVM feature importances for Group {group_idx} (highlighting XGBoost top features):")
    for feat in top_features:
        imp = svc_feature_importance_dict[feat]
        highlight = " <-- XGBoost Top Feature" if feat in top_features else ""
        print(f"  {feat}: {imp}{highlight}")
    
    # plot a heatmap of feature importance across models

    
    # exclude Logistic regression and SVM for better visualization since they have very different importance distribution compared to tree-based models
    all_imp_subset = all_imp[~all_imp["model"].isin(["LogisticRegression", "SVC-RBF"])]
    # all_imp_subset = all_imp
    normalized_importance = all_imp_subset.groupby("model")["importance"].transform(lambda s: s / (s.sum() + 1e-12))
    all_imp_subset["importance"] = normalized_importance
    # sort the features in the heatmap according to XGBoost importance
    all_imp_subset["feature"] = pd.Categorical(all_imp_subset["feature"], categories=top_features, ordered=True)
    heat = all_imp_subset.pivot_table(
        index="model",
        columns="feature",
        values="importance",
        aggfunc="mean"      # or np.median, "max", etc.
    )
    # all_imp_subset = all_imp_subset.sort_values(["model", "feature"])
    plt.figure(dpi=250)
    # use heatmap to visualize the feature importance across models, with x-axis as feature and y-axis as model, and color as importance
    # add angle to x-axis labels for better visualization

    sns.heatmap(heat, annot=False, fmt=".4f", cmap="YlGnBu")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0, ha="right")
    # sns.heatmap(all_imp_subset.pivot(index="model", columns="feature", values="importance"), annot=False, fmt=".4f", cmap="YlGnBu")
    plt.title(f"Feature Importance Heatmap (top {top_k})")
    # plt.xlabel("Feature", fontsize=10)
    # plt.ylabel("Model", fontsize=10)
    plt.tight_layout()
    heatmap_fn = os.path.join(output_dir, f"Group{group_idx}_feature_importance_heatmap.png")
    plt.savefig(heatmap_fn)
    print(f"Saved feature importance heatmap to {heatmap_fn}")
    plt.show()  
    
