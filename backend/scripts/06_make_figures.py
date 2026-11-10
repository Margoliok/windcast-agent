"""6-қадам: дипломның суреттері -> reports/figures (Сурет 5, 6, 7, 13, 16, 17, 18, 20 + Кесте 18).

Іске қосу:  python scripts/06_make_figures.py [--no-live]
"""
import argparse

import _bootstrap  # noqa: F401

from app.evaluation.figures import make_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-live", action="store_true", help="Ерейментаудың жедел болжамын (Сурет 20) салмау")
    make_all(live=not ap.parse_args().no_live)
