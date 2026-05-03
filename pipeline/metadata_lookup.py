"""
CSV annotation lookup for prompt_en and environmental parameters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from utils.config_handler import pipeline_conf


_annotation_df = None


def _load_csv():
    """Lazy-load the annotation CSV, cached in module-level variable."""
    global _annotation_df
    if _annotation_df is not None:
        return
    csv_path = pipeline_conf["annotation"]["csv_path"]
    _annotation_df = pd.read_csv(csv_path)


def lookup_metadata(filename: str) -> dict:
    """Look up metadata by segmented filename in the annotation CSV.

    Args:
        filename: The wav filename to match against the CSV 'segmented_filename' column.

    Returns:
        dict with keys:
          - prompt_en: str
          - channel_depth_m: float | None
          - wind: int | None
          - distance: str | None
          - source: "csv" | "default"
    """
    _load_csv()
    conf = pipeline_conf["annotation"]
    fn_col = conf["filename_column"]

    # Strip file extension for matching
    stem = Path(filename).stem

    # Try exact match first (for pre-segmented files)
    match = _annotation_df[_annotation_df[fn_col] == filename]
    if match.empty:
        match = _annotation_df[_annotation_df[fn_col] == stem]

    # Try matching by stem prefix (CSV has segment suffixes like _0, _1)
    if match.empty:
        match = _annotation_df[_annotation_df[fn_col].str.startswith(stem)]

    if match.empty:
        return {
            "prompt_en": "Hydrophone recording of underwater acoustic signals.",
            "channel_depth_m": None,
            "wind": None,
            "distance": None,
            "source": "default",
        }

    row = match.iloc[0]
    return {
        "prompt_en": str(row[conf["prompt_column"]]),
        "channel_depth_m": _safe_float(row.get(conf["channel_depth_column"])),
        "wind": _safe_int(row.get(conf["wind_column"])),
        "distance": _safe_str(row.get(conf["distance_column"])),
        "source": "csv",
    }


def _safe_float(val):
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    if val is None or pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_str(val):
    if val is None or pd.isna(val):
        return None
    return str(val)
