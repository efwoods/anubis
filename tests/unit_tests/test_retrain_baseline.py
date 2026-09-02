"""Cover the baseline-retraining script's file-rewriting and prompt invariants.

The fit itself (matrix, IsolationForest, SHAP explainer, Tukey fence) is already
covered by ``test_style_features.py``. What is new and genuinely risky in
``scripts/retrain_chatgpt_baseline.py`` is the part that edits source and env files
in place, plus the frozen generation prompt — a silent failure in either produces a
baseline that looks fine and scores wrong.
"""

import json
import re
from pathlib import Path

import pytest

from scripts.retrain_chatgpt_baseline import (
    STYLE_SYSTEM_PROMPT,
    replace_assignment_value,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ENVIRONMENT_LINE_PATTERN = r"^BASELINE_RESPONSE_THRESHOLD=.*$"
CONTEXT_DEFAULT_PATTERN = (
    r"(baseline_response_threshold: float = field\(\s*\n\s*default=)[0-9.eE+-]+"
)


class TestReplaceAssignmentValue:
    """The threshold rewrite must land, or not happen at all — never half-happen."""

    def test_replaces_the_environment_value(self):
        text = "FOO=1\nBASELINE_RESPONSE_THRESHOLD=47.66322963655769\nBAR=2\n"
        updated = replace_assignment_value(
            text, ENVIRONMENT_LINE_PATTERN, "BASELINE_RESPONSE_THRESHOLD=12.5"
        )
        assert "BASELINE_RESPONSE_THRESHOLD=12.5" in updated
        assert "47.66322963655769" not in updated

    def test_preserves_surrounding_comments_and_line_count(self):
        text = (
            "# Stylistic + Knowledge profile thresholds\n"
            "# Recalibrated by data/build_baseline_features_arr.py.\n"
            "BASELINE_RESPONSE_THRESHOLD=47.66322963655769\n"
            "MIN_QUOTES_FOR_PROFILE=20\n"
        )
        updated = replace_assignment_value(
            text, ENVIRONMENT_LINE_PATTERN, "BASELINE_RESPONSE_THRESHOLD=12.5"
        )
        assert "# Stylistic + Knowledge profile thresholds" in updated
        assert "MIN_QUOTES_FOR_PROFILE=20" in updated
        assert updated.count("\n") == text.count("\n")

    def test_is_idempotent(self):
        text = "BASELINE_RESPONSE_THRESHOLD=47.66322963655769\n"
        once = replace_assignment_value(
            text, ENVIRONMENT_LINE_PATTERN, "BASELINE_RESPONSE_THRESHOLD=12.5"
        )
        twice = replace_assignment_value(
            once, ENVIRONMENT_LINE_PATTERN, "BASELINE_RESPONSE_THRESHOLD=12.5"
        )
        assert once == twice

    def test_raises_rather_than_silently_no_opping(self):
        """A missing declaration must fail loudly.

        A no-op here would leave the threshold disagreeing with the artifacts it was
        calibrated against — miscalibrated verdicts with no error anywhere.
        """
        with pytest.raises(ValueError, match="No declaration matched"):
            replace_assignment_value(
                "SOMETHING_ELSE=1\n",
                ENVIRONMENT_LINE_PATTERN,
                "BASELINE_RESPONSE_THRESHOLD=12.5",
            )

    def test_raises_when_the_anchor_is_ambiguous(self):
        """Two matches means the anchor cannot identify the authoritative one."""
        text = (
            "BASELINE_RESPONSE_THRESHOLD=1.0\n"
            "BASELINE_RESPONSE_THRESHOLD=2.0\n"
        )
        with pytest.raises(ValueError, match="expected exactly one"):
            replace_assignment_value(
                text, ENVIRONMENT_LINE_PATTERN, "BASELINE_RESPONSE_THRESHOLD=12.5"
            )

    def test_rewrites_the_live_context_default(self):
        """The multi-line dataclass anchor must still match the real context.py."""
        context_text = (
            REPOSITORY_ROOT / "src" / "anubis" / "utils" / "context.py"
        ).read_text(encoding="utf-8")
        updated = replace_assignment_value(
            context_text, CONTEXT_DEFAULT_PATTERN, r"\g<1>12.5"
        )
        match = re.search(
            r"baseline_response_threshold: float = field\(\s*\n\s*default=([0-9.eE+-]+)",
            updated,
        )
        assert match is not None
        assert match.group(1) == "12.5"

    def test_rewrites_the_live_environment_files(self):
        """Both env files must still carry exactly one declaration to rewrite."""
        for environment_file_name in (".env", ".env.dev"):
            environment_file_path = REPOSITORY_ROOT / environment_file_name
            if not environment_file_path.exists():
                continue
            updated = replace_assignment_value(
                environment_file_path.read_text(encoding="utf-8"),
                ENVIRONMENT_LINE_PATTERN,
                "BASELINE_RESPONSE_THRESHOLD=12.5",
            )
            assert "BASELINE_RESPONSE_THRESHOLD=12.5" in updated


class TestFrozenGenerationPrompt:
    """The corpus must differ across retrains ONLY by the model that answered."""

    def test_prompt_matches_the_system_message_recorded_in_the_corpus(self):
        """Editing STYLE_SYSTEM_PROMPT silently destroys cross-retrain comparability.

        Every baseline corpus is generated under this exact text, so the shipped
        corpus records it. If the two diverge, a future retrain would move the
        threshold for a reason unrelated to the model upgrade — and nobody could
        attribute the shift.
        """
        from data.build_baseline_features_arr import BASELINE_CORPUS_PATH

        if not BASELINE_CORPUS_PATH.exists():
            pytest.skip(f"{BASELINE_CORPUS_PATH.name} is not present")

        with BASELINE_CORPUS_PATH.open("r", encoding="utf-8") as handle:
            first_conversation = json.loads(handle.readline())

        system_messages = [
            message["content"]
            for message in first_conversation["messages"]
            if message["role"] == "system"
        ]
        assert system_messages, "corpus conversation carries no system message"
        assert system_messages[0] == STYLE_SYSTEM_PROMPT


class TestStoreCacheKeys:
    """The purge must target every artifact the runtime caches."""

    def test_covers_every_cached_baseline_artifact(self):
        """A key missing here is an artifact that stays stale after a retrain.

        The runtime self-heals only on feature-vector WIDTH change, so a
        model-upgrade retrain (which keeps the width) relies entirely on this list
        being complete.
        """
        from scripts.retrain_chatgpt_baseline import BASELINE_STORE_CACHE_KEYS

        assert set(BASELINE_STORE_CACHE_KEYS) == {
            "baseline_features_arr_list_str",
            "baseline_features_model_b64_pkl",
            "baseline_features_explainer_b64_pkl",
            "baseline_key_phrase_profile",
        }

    def test_every_key_is_read_by_the_runtime(self):
        """Each purged key must actually appear in the code paths that read it."""
        from scripts.retrain_chatgpt_baseline import BASELINE_STORE_CACHE_KEYS

        runtime_sources = "\n".join(
            (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
            for relative_path in (
                "src/anubis/graph.py",
                "src/anubis/utils/utility.py",
            )
        )
        for cache_key in BASELINE_STORE_CACHE_KEYS:
            assert cache_key in runtime_sources, f"{cache_key} is read nowhere"
