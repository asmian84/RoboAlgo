"""
tests/test_auto_trader/test_config_gate.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for auto_trader.config — the persisted-settings loader/saver and
its live-mode safety gate (assert_mode_safe / save()).

Every test monkeypatches the module-level `_CONFIG_FILE` path onto a pytest
`tmp_path`, so nothing here ever touches the real `auto_trader_config.json`.
"""
from __future__ import annotations

import pytest

from auto_trader import config
from .conftest import FakeBroker

ENV_VAR = "ROBOALGO_ALLOW_LIVE"


@pytest.fixture(autouse=True)
def _isolated_config_file(tmp_path, monkeypatch):
    """Redirect config._CONFIG_FILE to a throwaway path for every test here."""
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "auto_trader_config.json")
    # Start every test with the env var unset unless a test sets it itself.
    monkeypatch.delenv(ENV_VAR, raising=False)
    yield


def _cfg(**overrides) -> config.AutoTraderConfig:
    return config.AutoTraderConfig(**overrides)


class TestAssertModeSafeNeitherSwitch:
    def test_raises_when_broker_live_with_neither_switch(self):
        broker = FakeBroker(is_paper=False)
        cfg = _cfg(mode="paper", live_confirmed=False)
        with pytest.raises(config.AutoTraderConfigError):
            config.assert_mode_safe(broker, cfg)


class TestAssertModeSafeOneSwitchOnly:
    def test_raises_with_only_env_var_set(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "1")
        broker = FakeBroker(is_paper=False)
        cfg = _cfg(mode="live", live_confirmed=False)
        with pytest.raises(config.AutoTraderConfigError):
            config.assert_mode_safe(broker, cfg)

    def test_raises_with_only_live_confirmed_set(self):
        # env var deliberately left unset by the autouse fixture
        broker = FakeBroker(is_paper=False)
        cfg = _cfg(mode="live", live_confirmed=True)
        with pytest.raises(config.AutoTraderConfigError):
            config.assert_mode_safe(broker, cfg)


class TestAssertModeSafeBothSwitches:
    def test_passes_with_both_switches_set(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "1")
        broker = FakeBroker(is_paper=False)
        cfg = _cfg(mode="live", live_confirmed=True)
        # Must not raise.
        config.assert_mode_safe(broker, cfg)


class TestAssertModeSafePaperConfigLiveBroker:
    def test_raises_for_paper_mode_with_live_broker_even_with_both_switches(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "1")
        broker = FakeBroker(is_paper=False)
        cfg = _cfg(mode="paper", live_confirmed=True)
        with pytest.raises(config.AutoTraderConfigError):
            config.assert_mode_safe(broker, cfg)


class TestSaveRefusesLiveWithoutEnvVar:
    def test_save_refuses_live_mode_without_env_var(self):
        with pytest.raises(config.AutoTraderConfigError):
            config.save({"mode": "live"})
        # No file should have been written.
        assert not config._CONFIG_FILE.exists()

    def test_save_refuses_live_confirmed_without_env_var(self):
        with pytest.raises(config.AutoTraderConfigError):
            config.save({"live_confirmed": True})
        assert not config._CONFIG_FILE.exists()


class TestEnvVarMustBeExactLiteralOne:
    @pytest.mark.parametrize("bad_value", ["true", "yes", "0"])
    def test_non_literal_one_values_do_not_open_the_gate(self, monkeypatch, bad_value):
        monkeypatch.setenv(ENV_VAR, bad_value)
        broker = FakeBroker(is_paper=False)
        cfg = _cfg(mode="live", live_confirmed=True)
        with pytest.raises(config.AutoTraderConfigError):
            config.assert_mode_safe(broker, cfg)

    @pytest.mark.parametrize("bad_value", ["true", "yes", "0"])
    def test_non_literal_one_values_refuse_save(self, monkeypatch, bad_value):
        monkeypatch.setenv(ENV_VAR, bad_value)
        with pytest.raises(config.AutoTraderConfigError):
            config.save({"mode": "live"})
