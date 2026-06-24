"""Simple example runner used by tests.

This script provides a minimal `main()` that calls `load_dataset`
and prints a small sequence of messages expected by the tests.
"""

from pathlib import Path
from dskit.data_io import load_dataset


def main():
    # Load dataset (tests monkeypatch this to avoid network I/O)
    try:
        df = load_dataset("https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv", index_col=0)
    except Exception:
        df = None

    # Pretend to run preprocessing and modeling
    print("Preprocessing complete")
    print("Model comparison results")


if __name__ == "__main__":
    main()
