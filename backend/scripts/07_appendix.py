"""7-қадам: Қосымша 1 — негізгі код листингтері және гиперпараметрлер -> reports/appendix/."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.config import get_settings

BACKEND = Path(__file__).resolve().parents[1]
LISTINGS = [
    ("Қ1.1", "Open-Meteo-дан деректерді жүктеу", "app/ingest/openmeteo.py"),
    ("Қ1.2", "Физикалық белгілер: ауа тығыздығы, ығысу, қуат қисығы", "app/features/physics.py"),
    ("Қ1.3", "Модель белгілерін қалыптастыру", "app/features/build.py"),
    ("Қ1.4", "LightGBM кванттильдік регрессиясы және конформдық түзету", "app/models/lgbm.py"),
    ("Қ1.5", "Тексеру шлюзі (verification gate)", "app/agent/gate.py"),
    ("Қ1.6", "ЖИ-агенттің құралдары", "app/agent/tools.py"),
    ("Қ1.7", "ЖИ-агенттің LangGraph графы", "app/agent/graph.py"),
]

if __name__ == "__main__":
    s = get_settings()
    out = s.reports_dir / "appendix"
    out.mkdir(parents=True, exist_ok=True)
    parts = ["# Қосымша 1. Негізгі код листингтері\n"]
    for num, title, rel in LISTINGS:
        code = (BACKEND / rel).read_text(encoding="utf-8")
        parts.append(f"## Листинг {num} — {title} (`backend/{rel}`)\n\n```python\n{code}\n```\n")
    meta_p = s.models_dir / "kelmarsh" / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        cfg = {k: meta.get(k) for k in ("mode", "splits", "curve_source", "lgbm_params", "conformal_offsets", "best_model")}
        yaml_lines = ["# Гиперпараметрлер конфигурациясы (Optuna, rolling-origin CV)"]
        def emit(d, ind=0):
            for k, v in d.items():
                if isinstance(v, dict):
                    yaml_lines.append(" " * ind + f"{k}:")
                    emit(v, ind + 2)
                else:
                    yaml_lines.append(" " * ind + f"{k}: {json.dumps(v, ensure_ascii=False)}")
        emit(cfg)
        (out / "hyperparameters.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        parts.append("## Листинг Қ1.8 — Гиперпараметрлер (YAML)\n\n```yaml\n" + "\n".join(yaml_lines) + "\n```\n")
    (out / "appendix1_code.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Дайын: {out / 'appendix1_code.md'}")
