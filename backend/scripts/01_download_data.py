"""1-қадам: деректерді жүктеу.

  python scripts/01_download_data.py           # ЭТАП A: шағын файлдар (~2 МБ): WT_static, сигналдар, Grid Meter
  python scripts/01_download_data.py --full    # ЭТАП B: + Kelmarsh SCADA 2020–2024 (~2.7 ГБ) + Chronos-Bolt салмақтары (~200 МБ)

Жүктеу докачкамен және Zenodo MD5 тексерісімен орындалады (үзілсе — қайта іске қосыңыз).
"""
import argparse

import _bootstrap  # noqa: F401

from app.core.config import get_settings
from app.ingest.kelmarsh import SMALL_FILES, download_file, list_zenodo_files

FULL_YEARS = range(2020, 2025)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="SCADA 2020–2024 және Chronos салмақтарын жүктеу (этап B)")
    ap.add_argument("--no-chronos", action="store_true")
    a = ap.parse_args()
    dest = get_settings().raw_dir / "kelmarsh"
    files = {f["key"]: f for f in list_zenodo_files()}
    wanted = list(SMALL_FILES)
    if a.full:
        wanted += sorted(k for k in files if k.startswith("Kelmarsh_SCADA_") and int(k.split("_")[2]) in FULL_YEARS)
    total = sum(files[k]["size"] for k in wanted) / 1e9
    print(f"Жүктеледі: {len(wanted)} файл, ~{total:.2f} ГБ -> {dest}")
    for k in wanted:
        download_file(files[k], dest)
    if a.full and not a.no_chronos:
        from huggingface_hub import snapshot_download

        print("Chronos-Bolt (amazon/chronos-bolt-small) салмақтары…")
        snapshot_download("amazon/chronos-bolt-small")
    print("Дайын. Келесі қадам: python scripts/02_build_dataset.py")
