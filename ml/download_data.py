
from pathlib import Path
from ucimlrepo import fetch_ucirepo


def main():
    # 1. Download the AI4I 2020 dataset from UCI
    print("Downloading AI4I 2020 dataset...")
    dataset = fetch_ucirepo(id=601)

    # 2. Retrieve the original dataset
    df = dataset.data.original

    # 3. Validate the dataset
    assert len(df) == 10000, "Unexpected number of rows"
    assert "Machine failure" in df.columns, "Target column missing"

    # 4. Save the original data locally
    output_path = Path("data/raw/ai4i_2020.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # 5. Display useful information
    print("\nDataset downloaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMachine failure distribution:")
    print(df["Machine failure"].value_counts())


if __name__ == "__main__":
    main()