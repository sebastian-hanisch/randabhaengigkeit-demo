"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Hilfetexte, Adresszeile."""

import pytest

import tdp_constants as C
import tdp_evaluation as ev
import tdp_presets as P


def test_presets_are_complete():
    assert "Standardfall (Voreinstellung)" in C.PRESETS
    assert set(C.PRESET_HELP) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name] and C.PRESET_HELP[name] != name         # ein echter Text, nicht der Name
        assert set(preset) == set(P.PRESET_KEYS), name
        assert (C.RHO_MIN <= preset["rho"] <= C.RHO_MAX and C.NU_MIN <= preset["nu"] <= C.NU_MAX and C.N_MIN <= preset["n"] <= C.N_MAX and preset["level"] in C.LEVELS
                and C.SHARE_MIN <= preset["share"] <= C.SHARE_MAX and 0 <= preset["seed"] <= C.SEED_MAX), name


def test_the_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    assert [P.SETTING_SPECS[k].default for k in ("rho_slider", "nu_slider", "n_slider", "level_select", "share_slider", "seed_input")] == [s.rho, s.nu, s.n, s.level, s.share, s.seed]


def test_every_preset_differs_from_every_other():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_the_presets_that_name_grid_cells_use_exactly_those_cells():
    """Die Hilfetexte lesen Zellen des Rasters: ρ, ν, n und Anteil der Presets müssen dort vorkommen."""
    pool = C.PRESETS["Poolen gegen Spitzen"]
    assert pool["rho"] in C.SUM_RHOS and pool["nu"] in C.SUM_NUS
    years = C.PRESETS["Wie viele Jahre Daten?"]
    assert years["n"] in C.EST_NS and years["share"] / 100 in C.EST_SHARES and years["rho"] == C.EST_RHO and years["nu"] == C.EST_NU
    for name in ("Standardfall (Voreinstellung)", "Gleiche Zahl, anderer Rand", "Fast Gauß (ν = 30)"):
        assert C.PRESETS[name]["rho"] == 0.5
    assert C.PRESETS["Gauß ohne Rand?"]["rho"] in C.COND_RHOS


@pytest.mark.parametrize("raw,expected", [("99999", C.N_MAX), ("1", C.N_MIN), ("150", 150), ("abc", None), ("", None)])
def test_clean_value_clamps_integers(raw, expected):
    assert P.clean_value(P.SETTING_SPECS["n_slider"], raw) == expected


def test_clean_value_clamps_and_rounds_the_correlation():
    spec = P.SETTING_SPECS["rho_slider"]
    assert P.clean_value(spec, "0.55") == 0.55 and P.clean_value(spec, "7") == C.RHO_MAX and P.clean_value(spec, "-1") == C.RHO_MIN
    assert P.clean_value(spec, "0.123456") == 0.12 and P.clean_value(spec, "x") is None


def test_clean_value_accepts_only_known_levels_and_clamps_the_seed():
    spec = P.SETTING_SPECS["level_select"]
    assert P.clean_value(spec, "0.999") == 0.999 and P.clean_value(spec, "0.5") is None and P.clean_value(spec, "") is None
    assert P.clean_value(P.SETTING_SPECS["seed_input"], "-5") == 0 and P.clean_value(P.SETTING_SPECS["seed_input"], str(C.SEED_MAX + 1)) == C.SEED_MAX
