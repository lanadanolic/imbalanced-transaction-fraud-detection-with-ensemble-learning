from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def create_logistic_regression(
    random_state: int = 42
):

    return LogisticRegression(
        max_iter=1000,
        random_state=random_state
    )


def create_random_forest(
    random_state: int = 42
):

    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )


def create_xgboost(
    random_state: int = 42
):

    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=random_state
    )