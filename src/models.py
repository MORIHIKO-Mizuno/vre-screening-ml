# --- Models ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC


def make_model(model_type, n_tree=1000, seed=42):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=n_tree,
            random_state=seed,
            class_weight="balanced"
        )

    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=n_tree,
            eval_metric="logloss",
            random_state=seed
        )

    elif model_type == "lgb":
        return LGBMClassifier(
            n_estimators=n_tree,
            random_state=seed,
            is_unbalance=True,   # class_weight="balanced"より確実
            verbose=-1,          # 警告抑制        
        )

    elif model_type == "cb":
        return CatBoostClassifier(
            iterations=n_tree,
            random_seed=seed,
            verbose=0,
            auto_class_weights="Balanced"
        )

    elif model_type == "lr":
        return LogisticRegression(
            random_state=seed,
            max_iter=20000,
            class_weight="balanced"
        )

    elif model_type == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            random_state=seed,
            max_iter=20000,
            class_weight="balanced"
        )

    elif model_type == "svc":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
            class_weight="balanced"
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
