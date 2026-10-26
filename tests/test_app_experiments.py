"""App: jedes Preset, der Permalink, jedes Experiment-Auswahlfeld einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import tdp_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["rho_slider"] == preset["rho"] and at.session_state["nu_slider"] == preset["nu"] and at.session_state["n_slider"] == preset["n"]
    assert at.session_state["level_select"] == preset["level"] and at.session_state["share_slider"] == preset["share"]
    assert len(at.get("plotly_chart")) == 10                  # Hauptansicht 3, E1: 1, E2: 2, E3: 1, E4: 2, E5: 1


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["rho"] = "0.9"
    at.query_params["nu"] = "999"
    at.query_params["n"] = "1"
    at.query_params["q"] = "0.999"
    at.query_params["share"] = "abc"                # ungültig: wird ignoriert
    at.run()
    assert not at.exception
    assert at.session_state["rho_slider"] == 0.9 and at.session_state["nu_slider"] == C.NU_MAX and at.session_state["n_slider"] == C.N_MIN
    assert at.session_state["level_select"] == 0.999 and at.session_state["share_slider"] == C.DEFAULT_SHARE


def test_permalink_ignores_an_unknown_level():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["q"] = "0.5"
    at.run()
    assert not at.exception and at.session_state["level_select"] == C.DEFAULT_LEVEL


def test_six_metrics_show_the_exact_numbers():
    at = _app()
    assert [m.label for m in at.metric] == ["Gauß: B oben, wenn A oben", "t-Kopula: B oben, wenn A oben", "λ der t-Kopula", "Pooling-Gewinn, Gauß", "Pooling-Gewinn, t-Kopula", "Kendalls τ (beide Kopulas)"]
    assert [m.value for m in at.metric[:3]] == ["0,129", "0,330", "0,313"] and at.metric[3].value == "13,4 %" and at.metric[5].value == "0,333"


def test_the_headline_verdicts_follow_the_settings():
    at = _app()
    assert "also das 2,5-Fache" in _texts(at)
    at.sidebar.select_slider(key="level_select").set_value(0.999).run()
    assert not at.exception and "also das 5,8-Fache" in _texts(at) and "Niveau 99,9 %" in _texts(at)


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("e1_rho", 0.3), ("e1_rho", 0.9), ("e2_nu", 5), ("e2_nu", 10), ("e2_rho", 0.0), ("e2_rho", 0.9), ("e3_rho", 0.7), ("e3_nu", 5), ("e3_nu", 10),
                       ("e4_share", 0.02), ("e4_share", 0.1), ("e4_n", 520), ("e4_n", 5000), ("e5_rho", 0.9), ("e5_rho", 0.3)):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("Bei Niveau 99,9 %: Gauß", "ρ = 0,5, t-Kopula mit ν = 3: Schranke 6 wird mit", "Pearson 0,487 gegen 0,500 (Gauß)", "n = 260 Wochen (5 Jahre), k = 13", "ν = 30: λ ist nur"):
        assert needle in text, needle


def test_every_slider_corner_runs_and_the_extremes_stay_finite():
    for edge in ("min", "max"):
        at = _app()
        for key in ("rho_slider", "nu_slider", "n_slider", "share_slider"):
            w = at.sidebar.slider(key=key)
            value = w.min if edge == "min" else w.max
            w.set_value(int(value) if isinstance(w.value, int) else value)
        at.sidebar.select_slider(key="level_select").set_value(C.LEVELS[0] if edge == "min" else C.LEVELS[-1])
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
        assert all(m.value for m in at.metric)


def test_a_missing_grid_shows_a_warning_instead_of_failing(monkeypatch, tmp_path):
    import streamlit as st

    import tdp_evaluation as ev

    st.cache_data.clear()                                  # zwischengespeicherte Raster aus anderen Tests verwerfen
    monkeypatch.setattr(ev, "GRID_PATH", tmp_path / "gibt-es-nicht.json")
    monkeypatch.setattr(ev, "load_grid", lambda path=tmp_path / "gibt-es-nicht.json": ev.json.loads("null"))
    try:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
        assert not at.exception, [str(e) for e in at.exception]
        assert any("Das vorgerechnete Raster" in w.value for w in at.warning)
    finally:
        st.cache_data.clear()
