"""Build the bundled PIAAC prior from the OECD Cycle 2 US public-use file.

Input:  prgusap2.csv (semicolon-separated, ~35MB; download from
        https://nces.ed.gov/surveys/piaac/datafiles.asp →
        "2023 U.S. PUF" entry).
Output: skillinfer/datasets/piaac_prior.npz with keys
        {feature_names, mean, cov, n, scale}.

We ship only summary statistics (mean and covariance). No individual
records leave the OECD distribution. Scale: each dimension is min-max
scaled to [0, 1] within the sample so the prior is consumable by
`Population.from_covariance` and Profile predictions stay inside the
package's [0, 1] clipping contract. Min-max preserves correlations.

Run:
    python scripts/prepare_piaac_prior.py path/to/prgusap2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Same dimension list and source columns used in the exp7 prep:
#   literacy / numeracy / problem_solving = mean of PVLIT*, PVNUM*, PVAPS*
#   work-use scales = the *_WLE_CA columns (Cycle 2 names).
PV_PREFIXES = {
    "literacy": "PVLIT",
    "numeracy": "PVNUM",
    "problem_solving": "PVAPS",
}
WORK_USE_COLS = {
    "readwork":  "READWORKC2_WLE_CA_T1",
    "writwork":  "WRITWORKC2_WLE_CA",
    "numwork":   "NUMWORKC2_WLE_CA",
    "ictwork":   "ICTWORKC2_WLE_CA",
    "influence": "INFLUENCEC2_WLE_CA_T1",
    "taskdisc":  "TASKDISCC2_WLE_CA_T1",
}
FEATURE_NAMES = list(PV_PREFIXES.keys()) + list(WORK_USE_COLS.keys())


def load_dimensions(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, sep=";", low_memory=False)
    out = pd.DataFrame(index=raw.index)
    for name, prefix in PV_PREFIXES.items():
        cols = [f"{prefix}{i}" for i in range(1, 11) if f"{prefix}{i}" in raw.columns]
        if not cols:
            raise ValueError(f"No plausible-value columns found for {name!r} (prefix {prefix})")
        out[name] = raw[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    for name, col in WORK_USE_COLS.items():
        if col not in raw.columns:
            raise ValueError(f"Missing work-use column: {col}")
        out[name] = pd.to_numeric(raw[col], errors="coerce")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Path to prgusap2.csv (PIAAC Cycle 2 US PUF)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "skillinfer" / "datasets" / "piaac_prior.npz",
        help="Output .npz path (default: bundled location).",
    )
    args = parser.parse_args()

    df = load_dimensions(args.csv)
    df = df.dropna()
    n = len(df)
    if n < 100:
        raise SystemExit(f"Too few complete rows ({n}); something is off with the source file.")

    raw_min = df.min().to_numpy(dtype=float)
    raw_max = df.max().to_numpy(dtype=float)
    scale_range = raw_max - raw_min
    if (scale_range < 1e-9).any():
        raise SystemExit("Encountered a constant column — refusing to scale.")
    scaled = (df.to_numpy(dtype=float) - raw_min) / scale_range  # → [0, 1]

    mean = scaled.mean(axis=0)
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(scaled)
    cov = lw.covariance_
    shrinkage = float(lw.shrinkage_)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        feature_names=np.array(FEATURE_NAMES),
        mean=mean,
        cov=cov,
        raw_min=raw_min,
        raw_max=raw_max,
        n=np.array([n]),
        shrinkage=np.array([shrinkage]),
    )
    print(f"Wrote {args.out}")
    print(f"  n={n}, K={len(FEATURE_NAMES)}, shrinkage={shrinkage:.4f}")
    print(f"  mean range [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  cov diag range [{np.diag(cov).min():.4f}, {np.diag(cov).max():.4f}]")


if __name__ == "__main__":
    main()
