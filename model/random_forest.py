from sklearn.ensemble import RandomForestClassifier

def build_model(random_state: int = 42):
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        max_depth=None
    )