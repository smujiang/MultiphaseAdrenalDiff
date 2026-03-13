from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ML_model_list = ["xgboost", "random_forest", "decision_tree", "logistic_regression", "svm"]
def make_xgb(scale_pos_weight, seed=42):        
    model = XGBClassifier(objective="binary:logistic",
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

