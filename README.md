# WindCast Agent

**Ауа райы деректері негізінде жел электр станциясының қуатын болжайтын ЖИ-агент**
*(ИИ-агент, прогнозирующий мощность ВЭС на основе данных о погоде)*

**🌐 Демо (GitHub Pages): https://margoliok.github.io/windcast-agent/**

WindCast Agent — веб-платформа. Ол ЖЭС өндірісін ауа райы болжамы (Open-Meteo NWP) бойынша 1–72 сағат алға болжайды: P50 нүктелік болжам және P10–P90 сенімділік аралығы. Платформада ЖИ-агент бар, пайдаланушы онымен қазақ немесе орыс тілінде сөйлеседі. Агент **solver-grounded** қағидасымен жұмыс істейді: LLM сандарды өзі есептемейді, барлық сандар ML-модельдер мен құралдардан алынады. Жауаптағы әр санды **тексеру шлюзі** (код, LLM емес) тексереді.

---

## 1. Мүмкіндіктер

| Функция | Іске асыру |
|---|---|
| Ауа райы болжамын жинау (әр сағат) | `app/ingest/openmeteo.py`, `app/scheduler.py` (APScheduler) |
| Қуатты болжау P10/P50/P90 | `app/models/*` (Persistence, физикалық, LightGBM-Q, LSTM, TFT, Chronos, ансамбль) |
| Болжамды түсіндіру (SHAP) | `app/explain/shap_explain.py` |
| Тәуліктік кесте → CSV/XLSX | `app/services/schedule_export.py` |
| Бэктест, модельдерді салыстыру | `app/evaluation/*`, `app/services/history_service.py` |
| What-if сценарийі | `forecast_service.what_if` |
| Тоқтап тұру кезеңдерін анықтау | `history_service.detect_downtime` |
| Дашборд + агентпен чат (kk/ru) | `frontend/` (React + Vite + Plotly) |
| ЖИ-агент (10 құрал, LangGraph, шлюз) | `app/agent/*` (OpenAI GPT) |
| PDF/Markdown есеп | `app/services/report.py` |

Станциялар:
- **Kelmarsh**, Ұлыбритания, 6 × Senvion MM92, 12,3 МВт. Ашық SCADA (CC-BY-4.0); модельдер осы станцияда оқытылып, тексеріледі.
- **Ерейментау ЖЭС**, Қазақстан, 45 МВт. Қазақстандық кейс: жедел болжам мен тәуліктік кесте. SCADA деректері ашық емес, сондықтан дәлдік Kelmarsh-та тексерілген. Бұл шектеу интерфейсте де көрсетіледі.

## 2. Архитектура

```
Open-Meteo API ─┐                       ┌─ Модельдер (M0–M6) ── SHAP
Kelmarsh SCADA ─┴─ ETL/тазалау ─ БД ─ белгілер ─┤
                (APScheduler)  (SQLite/        └─ MLflow (тізілім)
                               PostgreSQL)
React (дашборд+чат) ⇄ FastAPI ⇄ ЖИ-агент (LangGraph: agent → tools → verify → END / safe_fail)
                                    └── 10 құрал ── сервистер ── модельдер / БД / Open-Meteo
```

Дерекқорда 7 кесте бар: `farms`, `scada_measurements`, `weather_forecasts`, `weather_history`, `forecasts`, `metrics`, `agent_logs` (`app/db/models.py`).

## 3. Әдістеме (қысқаша)

* **Деректер.**
  - Kelmarsh: SCADA 2020–2024 (этап B) немесе станция есептегіші 2017–2019 (этап A).
  - Ауа райы: Open-Meteo-дан үш дереккөз — тарихи NWP болжамдары (оқыту), **Previous Runs API** (1/2/3 тәулік бұрынғы нақты болжамдар, горизонт бойынша адал тест) және ERA5 («мінсіз болжам», жоғарғы шек).
* **Тазалау.**
  - Status файлындағы `Stop` оқиғалары алынады.
  - Жел > 4,5 м/с кезінде P ≈ 0 болған интервалдар алынады.
  - Шектеу (curtailment): қалақ бұрышы және setpoint бойынша.
  - Қуат қисығынан шығарындылар (MAD) алынады.
  - Сағаттық агрегация сағат белгісіне *орталанған* терезеде жасалады (`[t−30, t+30)` мин), себебі Open-Meteo желі — лездік мән.
* **Белгілер.** Тығыздық ρ = p/(R·T); ығысу α; ступица биіктігіндегі жылдамдық; тығыздыққа түзетілген жылдамдық; эмпирикалық қисық бойынша p_phys; бағыттың u/v және sin/cos компоненттері; уақыт белгілері. 1–6 сағаттық режимде қосымша қуат лагтары қолданылады.
* **Модельдер.**
  - LightGBM: әр τ ∈ {0.1…0.9} үшін бөлек модель.
  - Гиперпараметрлер: Optuna + rolling-origin CV (тек train ішінде).
  - Валидацияда конформдық түзету (CQR) жасалады.
  - Ансамбль салмақтары валидацияда CRPS бойынша таңдалады.
  - Соңғы модель train+val бойынша оқытылады; тест жиынтығы тек бағалауға қолданылады.
* **Метрикалар:** MAE, RMSE, nMAE, nRMSE, bias, skill score (persistence-ке қатысты), pinball, CRPS, PICP, аралық ені; агент үшін faithfulness (18).

## 4. Іске қосу (жергілікті)

```bash
# Python 3.11/3.12
python -m venv .venv && .venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -r backend/requirements.txt                   # + backend/requirements-research.txt (LSTM/TFT/Chronos)
copy .env.example .env                                    # OPENAI_API_KEY толтырыңыз

cd backend
python scripts/01_download_data.py          # этап A: шағын файлдар (~2 МБ)
python scripts/02_build_dataset.py          # қуат + Open-Meteo (NWP, ERA5, Previous Runs)
python scripts/03_train.py --trials 25      # қисық, M0–M2, ансамбль, Ерейментау моделі
python scripts/04_run_experiments.py        # E1–E5 -> reports/tables
python scripts/06_make_figures.py           # суреттер -> reports/figures
python scripts/05_eval_agent.py             # E6: агент vs құралсыз LLM (OPENAI_API_KEY керек)
python scripts/07_appendix.py               # Қосымша 1
pytest -q                                   # тесттер

uvicorn app.main:app --port 8000            # API: http://localhost:8000/docs
cd ../frontend && npm install && npm run dev   # UI: http://localhost:5173
```

**Этап B (толық деректер, ~2,9 ГБ):** `python scripts/01_download_data.py --full`, кейін 02–07 қадамдарды қайталаңыз. Режим автоматты түрде `scada`-ға ауысады:
- train 2020–2022, val 2023, test 2024;
- 2024 жылы горизонтқа тәуелді NWP қолданылады.

## 5. Docker

```bash
docker compose up --build      # UI: http://localhost:8080, API: http://localhost:8000/docs
```

`data/` және `reports/` бумалары контейнерге қосылады, сондықтан алдымен 01–04 қадамдарды орындаңыз.

## 6. GitHub Pages демосы

GitHub Pages — статикалық хостинг, онда Python-бэкенд жұмыс істемейді. Сондықтан демо деректердің **статикалық снимогін** көрсетеді. Барлық JSON файлдарды нақты сервистер `scripts/08_export_demo.py` арқылы жасайды; сандар толық нұсқамен бірдей.

Демода қолжетімді:
- дашборд (болжам горизонттары мен модельдері, SHAP, әлсіз жел, PDF есеп);
- модельдерді салыстыру мен бэктест;
- тәуліктік кесте (XLSX/CSV);
- Kelmarsh тарихы және тоқтаулар (кез келген кезең, есептеу браузерде);
- what-if (−50…+50%).

ЖИ-агент чаты демода **нақты жазылған диалогтарды** ойнатады (траектория + шлюз). Оларды жазу үшін `.env` ішіне `OPENAI_API_KEY` қойып, мына командаларды іске қосыңыз:

```bash
python scripts/09_record_agent_demo.py   # агенттің нақты жауаптарын жазу
python scripts/08_export_demo.py         # демо снимогін жаңарту -> frontend/public/demo
git add frontend/public/demo && git commit -m "Update demo snapshot" && git push   # Pages автоматты жаңарады
```

Агентпен тірі сөйлесу тек толық нұсқада мүмкін (жергілікті немесе Docker).

## 7. Дипломдағы суреттер мен кестелер

| Материал | Файл |
|---|---|
| Сурет 5, 6, 7, 13, 16, 17, 18, 20 | `reports/figures/fig*.png` (`06_make_figures.py`) |
| Сурет 14, 15 (скриншоттар) | `reports/figures/fig14_*.png`, `fig15_*.png` |
| Кесте 7 | `data/processed/kelmarsh_quality.csv` |
| Кесте 13–16 | `reports/tables/tab13…tab16*.md/csv` |
| Кесте 17 | `reports/tables/tab17_agent_vs_llm.md` (`05_eval_agent.py`) |
| Кесте 18 | `reports/tables/tab18_ereymentau_schedule.md` |
| Қосымша 1 / 3 | `reports/appendix/appendix1_code.md`, `reports/agent_eval/questions_appendix3.csv` |

> ⚠️ **Адалдық туралы.** Этап A-да модельдер станция есептегішінің 2017–2019 деректерімен оқытылады. Бұл кезеңде горизонтқа тәуелді NWP архиві жоқ, сондықтан нәтижелер — алдын ала нұсқа (черновик). Дипломдағы соңғы сандар этап B-дан кейін алынады. Ойдан шығарылған сан жоқ: барлық кестелер скрипттермен қайта есептеледі.

---

## Кратко по-русски

**WindCast Agent** — платформа прогноза выработки ВЭС на 1–72 ч с интервалами P10–P90 и ИИ-агентом (OpenAI GPT + LangGraph). Агент вызывает 10 инструментов. Каждое число в ответе проверяет верификационный шлюз: если число не подтверждается выходом инструментов, агент повторяет попытку, а затем честно отказывает.

- Модели обучаются на открытых данных ВЭС Kelmarsh.
- Казахстанский кейс — ВЭС Ерейментау (прогноз и суточный график).
- Порядок запуска — в разделе 4. Полные данные (этап B) скачиваются флагом `--full`.
- Интерфейс на казахском, есть переключатель на русский.
