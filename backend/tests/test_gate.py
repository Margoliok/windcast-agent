from app.agent.gate import check_series, extract_numbers, faithfulness, verify_answer

ROWS_OK = [{"p10": 1.0, "p50": 2.0, "p90": 3.0}, {"p10": 0.0, "p50": 0.5, "p90": 1.0}]


def test_series_passes_when_physical():
    assert check_series(ROWS_OK, capacity_mw=12.3, expected_len=2).passed


def test_series_fails_above_capacity():
    rows = [{"p10": 1.0, "p50": 2.0, "p90": 13.0}]
    g = check_series(rows, capacity_mw=12.3)
    assert not g.passed
    assert any(c["check"] == "bounds" and not c["passed"] for c in g.checks)


def test_series_fails_on_quantile_crossing():
    rows = [{"p10": 3.0, "p50": 2.0, "p90": 4.0}]
    g = check_series(rows, capacity_mw=12.3)
    assert any(c["check"] == "quantile_order" and not c["passed"] for c in g.checks)


def test_series_fails_on_length_and_nan():
    g = check_series([{"p10": float("nan"), "p50": 1, "p90": 2}], 12.3, expected_len=24)
    failed = {c["check"] for c in g.checks if not c["passed"]}
    assert {"no_nan", "length"} <= failed


def test_extract_numbers_skips_dates_times_and_labels():
    nums = [v for _, v in extract_numbers("2026-11-11 10:00–18:00 аралығында P50 бойынша 45,3 МВт·сағ, P10 = 30.1")]
    assert nums == [45.3, 30.1]


def test_faithfulness_supported_numbers():
    tool = {"summary": {"energy_mwh_p50": 45.3, "mean_mw_p50": 5.66, "capacity_factor_pct": 46.0}}
    f = faithfulness("Ертең 45,3 МВт·сағ өндіріледі, орташа қуат 5.7 МВт, коэффициент 46%.", [tool])
    assert f["score"] == 1.0 and f["n_numbers"] == 3


def test_faithfulness_detects_fabricated_number():
    tool = {"summary": {"energy_mwh_p50": 45.3}}
    f = faithfulness("Ертең 45.3 МВт·сағ, ал келесі күні шамамен 60 МВт·сағ болады.", [tool])
    assert f["score"] == 0.5
    assert f["unsupported"] == ["60"]


def test_faithfulness_unit_conversion_and_question_numbers():
    tool = {"mae_mw": 0.782}
    f = faithfulness("MAE 782 кВт құрады; сіз сұраған 15% сценарий бойынша...", [tool], question="жел 15% күшті болса?")
    assert f["score"] == 1.0


def test_verify_answer_fails_when_tool_gate_failed():
    out = {"tool": "predict_power", "gate": {"passed": False}, "summary": {"energy_mwh_p50": 10.0}}
    g = verify_answer("Өндіріс 10.0 МВт·сағ.", [out])
    assert not g.passed


def test_no_numbers_answer_passes():
    g = verify_answer("Кешіріңіз, бұл сұрақ жел энергетикасына қатысты емес.", [])
    assert g.passed
