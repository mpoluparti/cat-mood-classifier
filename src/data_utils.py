"""Build a labeled file index from the raw dataset directory.

Assumes the CatMeows dataset naming convention: filenames start with a
context code (B = brushing, F = waiting for food, I = isolation), e.g.
"B_ANI01_MC_FN_SIM01_101.wav". Verify this against the files you actually
download -- dataset naming can vary by source/version.
"""

from pathlib import Path

import pandas as pd

CONTEXT_CODES = {
    "B": "brushing",
    "F": "waiting_for_food",
    "I": "isolation",
}


def build_index(raw_dir: Path) -> pd.DataFrame:
    """Scan raw_dir for .wav files and label each by its filename prefix."""
    rows = []
    for path in sorted(Path(raw_dir).rglob("*.wav")):
        code = path.stem.split("_")[0].upper()
        label = CONTEXT_CODES.get(code)
        if label is None:
            continue
        rows.append({"path": str(path), "label": label})

    df = pd.DataFrame(rows, columns=["path", "label"])
    if df.empty:
        raise ValueError(
            f"No labeled .wav files found under {raw_dir}. "
            "Check that the dataset was extracted there and filenames "
            "match the expected B_/F_/I_ prefix convention."
        )
    return df
