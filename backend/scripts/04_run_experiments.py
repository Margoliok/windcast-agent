"""4-қадам: эксперименттер E1–E5 -> reports/tables.

Іске қосу:  python scripts/04_run_experiments.py [--no-neural]
"""
import argparse
import json

import _bootstrap  # noqa: F401

from app.evaluation.experiments import run_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-neural", action="store_true", help="LSTM/TFT/Chronos эксперименттерін өткізу")
    s = run_all(with_neural=not ap.parse_args().no_neural)
    print(json.dumps(s, ensure_ascii=False, indent=2, default=float))
