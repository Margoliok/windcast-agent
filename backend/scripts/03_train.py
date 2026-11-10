"""3-қадам: модельдерді оқыту (қуат қисығы, Persistence, физикалық, LightGBM, ансамбль).

Іске қосу:  python scripts/03_train.py [--trials 25]
"""
import argparse
import json

import _bootstrap  # noqa: F401

from app.models.training import train_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    meta = train_all(n_trials=ap.parse_args().trials)
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
