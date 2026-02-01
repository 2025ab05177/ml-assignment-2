import os
import pandas as pd

# Expected folder layout after unzipping:
# UCI_HAR_Dataset/
#   ├── features.txt
#   ├── activity_labels.txt
#   ├── train/
#   │    ├── X_train.txt
#   │    └── y_train.txt
#   └── test/
#        ├── X_test.txt
#        └── y_test.txt

def read_features(base_dir: str):
    feat_path = os.path.join(base_dir, "features.txt")
    feats = pd.read_csv(feat_path, sep=r"\s+", header=None, names=["idx", "name"])
    # make names unique (HAR has duplicates)
    counts = {}
    fixed = []
    for n in feats["name"].tolist():
        counts[n] = counts.get(n, 0) + 1
        fixed.append(f"{n}__{counts[n]}")
    return fixed

def read_activity_labels(base_dir: str):
    path = os.path.join(base_dir, "activity_labels.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["id", "label"])
    return dict(zip(df["id"], df["label"]))

def load_split(base_dir: str, split: str, feature_names):
    X_path = os.path.join(base_dir, split, f"X_{split}.txt")
    y_path = os.path.join(base_dir, split, f"y_{split}.txt")

    X = pd.read_csv(X_path, sep=r"\s+", header=None)
    X.columns = feature_names

    y = pd.read_csv(y_path, sep=r"\s+", header=None, names=["ActivityID"])
    return X, y["ActivityID"]

def main():
    base_dir = "UCI_HAR_Dataset"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            "Folder 'UCI_HAR_Dataset' not found. Download UCI HAR zip, unzip, and rename/extract to this folder."
        )

    feature_names = read_features(base_dir)
    act_map = read_activity_labels(base_dir)

    X_train, y_train = load_split(base_dir, "train", feature_names)
    X_test, y_test = load_split(base_dir, "test", feature_names)

    train = X_train.copy()
    train["Activity"] = y_train.map(act_map)

    test = X_test.copy()
    test["Activity"] = y_test.map(act_map)

    full = pd.concat([train, test], axis=0, ignore_index=True)

    train.to_csv("har_train.csv", index=False)
    test.to_csv("har_test.csv", index=False)
    full.to_csv("har_full.csv", index=False)

    print("Created: har_train.csv, har_test.csv, har_full.csv")
    print("Rows:", full.shape[0], "Cols:", full.shape[1])

if __name__ == "__main__":
    main()