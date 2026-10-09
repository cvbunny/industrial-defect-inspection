from pathlib import Path

from src.data.dataset import KolektorSDD2Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kolektorsdd2-DatasetNinja"
)

TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"


def main():
    dataset = KolektorSDD2Dataset(
        dataset_root=DATASET_ROOT,
        split_csv=TRAIN_CSV,
    )

    print("Dataset size:", len(dataset))

    defect_index = dataset.samples.index[
        dataset.samples["has_defect"] == 1
    ][0]

    image, mask = dataset[defect_index]

    print("Defect sample index:", defect_index)
    print("Filename:", dataset.samples.iloc[defect_index]["filename"])

    print("Image tensor shape:", image.shape)
    print("Image dtype:", image.dtype)

    print("Mask tensor shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Mask unique values:", mask.unique())

    print("Defect pixels:", mask.sum().item())


if __name__ == "__main__":
    main()