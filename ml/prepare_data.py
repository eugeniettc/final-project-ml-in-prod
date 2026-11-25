
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Original dataset
DATA_PATH = ROOT / "data/raw/ai4i_2020.csv"

# Model input variables
FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

TARGET = "Machine failure"
RANDOM_STATE = 42


def main():
    # 1. Load original dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Validate required columns and labels
    required = ["UID"] + FEATURES + [TARGET]

    assert all(col in df.columns for col in required)
    assert df[required].notna().all().all()
    assert df["UID"].is_unique
    assert set(df[TARGET].unique()) == {0, 1}

    print("Original dataset:", df.shape)

    # 3. First split: 80% train+validation, 20% test
    train_val, test = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df[TARGET]
    )

    # 4. Second split: 60% train, 20% validation
    train_full, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_val[TARGET]
    )

    # 5. Version 1: half of the training data
    train_v1, _ = train_test_split(
        train_full,
        train_size=0.50,
        random_state=RANDOM_STATE,
        stratify=train_full[TARGET]
    )

    # Version 2: all available training data
    train_v2 = train_full.copy()

    # 6. Verify there is no overlap
    train_ids = set(train_v2["UID"])
    val_ids = set(validation["UID"])
    test_ids = set(test["UID"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert set(train_v1["UID"]).issubset(train_ids)

    print("No data leakage between splits.")

    # 7. Keep only model features and target
    columns = ["UID"] + FEATURES + [TARGET]
    datasets = {
        "data/releases/v1/train.csv": train_v1,
        "data/releases/v2/train.csv": train_v2,
        "data/splits/validation.csv": validation,
        "data/splits/test.csv": test,
    }

    # 8. Save the datasets
    for filename, subset in datasets.items():
        path = ROOT / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        subset[columns].to_csv(path, index=False)

        failures = int(subset[TARGET].sum())

        print(
            f"{filename}: {len(subset)} rows, "
            f"{failures} failures "
            f"({failures / len(subset):.2%})"
        )

    print("\nData preparation completed successfully!")


if __name__ == "__main__":
    main()