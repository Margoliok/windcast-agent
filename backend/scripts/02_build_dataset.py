"""2-қадам: станция қуаты + ауа райы (NWP, ERA5, Previous Runs) -> data/processed.

Іске қосу:  python scripts/02_build_dataset.py [--mode scada|grid_meter]
"""
import argparse

import _bootstrap  # noqa: F401

from app.ingest.dataset import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scada", "grid_meter"], default=None)
    build(ap.parse_args().mode)
