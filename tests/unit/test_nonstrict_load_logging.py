"""Non-strict checkpoint loads must log dropped keys instead of hiding them.

YOLOX loads with strict=False. Shape mismatches raise regardless of
strictness, but *named* key mismatches (missing/unexpected) were silently
discarded: a partially matching checkpoint would "load" and then predict with
fresh-initialized tensors wherever keys were absent. `_load_state_dict_logged`
makes that visible with a warning while keeping healthy loads silent.
"""

import logging

import pytest

from libreyolo.models.yolox.model import LibreYOLOX

pytestmark = pytest.mark.unit

LOGGER_NAME = "libreyolo.models.base.model"


def _clean_state(size="s", nb_classes=3):
    return LibreYOLOX(size=size, nb_classes=nb_classes).model.state_dict()


class TestNonStrictLoadLogging:
    def test_healthy_load_is_silent(self, caplog):
        state = _clean_state()
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            LibreYOLOX(model_path=state, size="s", nb_classes=3)
        assert not [r for r in caplog.records if "Non-strict load" in r.message]

    def test_missing_and_unexpected_keys_warn(self, caplog):
        state = _clean_state()
        dropped = [k for k in state if k.startswith("head.")][:3]
        for key in dropped:
            del state[key]
        state["totally.bogus.weight"] = next(iter(state.values())).clone()

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            LibreYOLOX(model_path=state, size="s", nb_classes=3)

        warnings = [r for r in caplog.records if "Non-strict load" in r.message]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "3 missing key(s)" in msg
        assert "1 unexpected key(s)" in msg
        assert dropped[0] in msg
        assert "totally.bogus.weight" in msg
