from xgboost import XGBClassifier

def build_model(num_class: int, random_state: int = 42):
    # Multi-class softprob gives probability for each class (needed for AUC)
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_class,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )