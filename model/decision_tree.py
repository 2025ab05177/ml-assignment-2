from sklearn.tree import DecisionTreeClassifier

def build_model(random_state: int = 42):
    return DecisionTreeClassifier(
        random_state=random_state,
        max_depth=None
    )