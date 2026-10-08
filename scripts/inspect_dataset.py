from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    print(f"Dataset directory: {DATA_DIR}")
    print()

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATA_DIR}"
        )

    files = list(DATA_DIR.rglob("*"))

    directories = [p for p in files if p.is_dir()]
    regular_files = [p for p in files if p.is_file()]

    print(f"Number of directories: {len(directories)}")
    print(f"Number of files: {len(regular_files)}")
    print()

    print("Top-level contents:")
    for path in sorted(DATA_DIR.iterdir()):
        print(f"  {path.name}")

    print()
    print("Example files:")

    for path in sorted(regular_files)[:40]:
        print(f"  {path.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()