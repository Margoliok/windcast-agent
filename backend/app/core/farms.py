"""Жел электр станцияларының паспорттық сипаттамалары.

Kelmarsh: Kelmarsh_WT_static.csv (Zenodo 16807551) файлынан алынған.
Ерейментау: ашық дереккөздер бойынша (45 МВт, 51.596 N, 73.082 E). Турбина түрі мен
ступица биіктігі ашық дереккөздерде расталмағандықтан, Kelmarsh-тың нормаланған қуат
қисығы мен 2 МВт класындағы турбинаға тән 80 м ступица биіктігі ДОПУЩЕНИЕ ретінде қолданылады.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Farm:
    id: str
    name: str
    lat: float
    lon: float
    capacity_mw: float
    hub_height_m: float
    n_turbines: int
    turbine_model: str
    timezone: str
    country: str
    has_scada: bool
    curve_source: str  # қай станцияның қуат қисығы қолданылады
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Kelmarsh: 4 турбина 78.5 м, 2 турбина 68.5 м -> орташа 75.17 м
KELMARSH = Farm(
    id="kelmarsh",
    name="Kelmarsh",
    lat=52.4015,
    lon=-0.9431,
    capacity_mw=12.3,
    hub_height_m=round((4 * 78.5 + 2 * 68.5) / 6, 2),
    n_turbines=6,
    turbine_model="Senvion MM92 (6 × 2.05 MW)",
    timezone="Europe/London",
    country="UK",
    has_scada=True,
    curve_source="kelmarsh",
    notes={
        "kk": "Ашық SCADA деректері (CC-BY-4.0). Модельдер осы станцияда оқытылып, тексеріледі.",
        "ru": "Открытые SCADA-данные (CC-BY-4.0). Модели обучаются и проверяются на этой станции.",
    },
)

EREYMENTAU = Farm(
    id="ereymentau",
    name="Ерейментау ЖЭС",
    lat=51.596,
    lon=73.082,
    capacity_mw=45.0,
    hub_height_m=80.0,
    n_turbines=22,
    turbine_model="~2 МВт класы (допущение)",
    timezone="Asia/Almaty",
    country="KZ",
    has_scada=False,
    curve_source="kelmarsh",
    notes={
        "kk": "SCADA деректері ашық емес: болжам дәлдігі Kelmarsh станциясында тексерілген. "
        "Турбина түрі расталмағандықтан Kelmarsh-тың нормаланған қуат қисығы қолданылады (допущение).",
        "ru": "SCADA-данные закрыты: точность проверена на Kelmarsh. Тип турбин не подтверждён, "
        "используется нормированная кривая мощности Kelmarsh (допущение).",
    },
)

FARMS: dict[str, Farm] = {f.id: f for f in (KELMARSH, EREYMENTAU)}


def get_farm(farm_id: str) -> Farm:
    try:
        return FARMS[farm_id]
    except KeyError as e:
        raise ValueError(f"Белгісіз станция: {farm_id}. Қолжетімді: {', '.join(FARMS)}") from e
