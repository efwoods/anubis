"""Cover the baseline-retraining script's file-rewriting, provenance, and prompt invariants.

The fit itself (matrix, IsolationForest, SHAP explainer, Tukey fence) is covered by
``test_style_features.py``. What is genuinely risky in
``scripts/retrain_chatgpt_baseline.py`` is the part that edits source and env files
in place, the provenance sidecar that every later boot trusts, the store publish,
and the frozen generation prompt — a silent failure in any of them produces a
baseline that looks fine and scores wrong.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts.retrain_chatgpt_baseline import (
    BASELINE_ARTIFACT_STORE_KEYS,
    BASELINE_PROVENANCE_STORE_KEY,
    BASELINE_RETRAIN_LOCK_STORE_KEY,
    BASELINE_STORE_CACHE_KEYS,
    MODEL_CONFIGURATION_ANCHORS,
    STYLE_SYSTEM_PROMPT,
    ModelConfiguration,
    build_baseline_provenance,
    compute_style_system_prompt_sha256,
    format_cost_for_environment_file,
    publish_baseline_to_store,
    read_baseline_provenance,
    replace_assignment_value,
    write_baseline_provenance,
    write_model_configuration_to_environment_files,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ENVIRONMENT_LINE_PATTERN = r"^BASELINE_RESPONSE_THRESHOLD=.*$"
CONTEXT_DEFAULT_PATTERN = (
    r"(baseline_response_threshold: float = field\(\s*\n\s*default=)[0-9.eE+-]+"
)

# A realistic slice of an env file: commented-out alternatives ABOVE the live
# declaration (exactly as .env has them), a quoted MODEL, and the two look-alike
# names that a sloppy ``MODEL`` anchor would clobber.
SYNTHETIC_ENVIRONMENT_TEXT = (
    "# MODEL_PROVIDER=TOGETHER\n"
    "MODEL_PROVIDER=OPEN_AI\n"
    "# MODEL=meta-llama/Meta-Llama-3-8B-Instruct-Lite\n"
    "# MODEL=\"gpt-5-nano\"\n"
    'MODEL="gpt-5.4-nano"\n'
    "MODEL_PROMPT_COST=0.0000002\n"
    "MODEL_COMPLETION_COST=0.00000125\n"
    "MODEL_TOKEN_LIMIT=400000\n"
    "BASELINE_RESPONSE_THRESHOLD=47.66322963655769\n"
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


class TestModelConfigurationRewrite:
    """A model switch rewrites exactly the five live lines, nothing around them."""

    def test_each_anchor_matches_exactly_once_on_synthetic_text(self):
        for name, (pattern, _) in MODEL_CONFIGURATION_ANCHORS.items():
            matches = re.findall(pattern, SYNTHETIC_ENVIRONMENT_TEXT, flags=re.MULTILINE)
            assert len(matches) == 1, f"{name} matched {matches}"

    def test_rewrite_preserves_quoting_and_leaves_decoys_alone(self, tmp_path):
        environment_file = tmp_path / ".env"
        environment_file.write_text(SYNTHETIC_ENVIRONMENT_TEXT, encoding="utf-8")
        configuration = ModelConfiguration(
            model="gpt-5.6-luna",
            model_provider="OPEN_AI",
            model_prompt_cost="0.0000002",
            model_completion_cost="0.0000012",
        )
        write_model_configuration_to_environment_files(configuration, [environment_file])
        updated = environment_file.read_text(encoding="utf-8")
        assert 'MODEL="gpt-5.6-luna"\n' in updated
        assert "MODEL_COMPLETION_COST=0.0000012\n" in updated
        assert "MODEL_PROMPT_COST=0.0000002\n" in updated
        # Untouched: the commented alternatives, the look-alike names, the threshold.
        assert "# MODEL=meta-llama/Meta-Llama-3-8B-Instruct-Lite\n" in updated
        assert '# MODEL="gpt-5-nano"\n' in updated
        assert "MODEL_TOKEN_LIMIT=400000\n" in updated
        assert "BASELINE_RESPONSE_THRESHOLD=47.66322963655769\n" in updated
        assert updated.count("\n") == SYNTHETIC_ENVIRONMENT_TEXT.count("\n")

    def test_omitted_fields_are_not_rewritten(self, tmp_path):
        environment_file = tmp_path / ".env"
        environment_file.write_text(SYNTHETIC_ENVIRONMENT_TEXT, encoding="utf-8")
        write_model_configuration_to_environment_files(
            ModelConfiguration(model="gpt-5.6-luna"), [environment_file]
        )
        updated = environment_file.read_text(encoding="utf-8")
        assert "MODEL_PROVIDER=OPEN_AI\n" in updated
        assert "MODEL_COMPLETION_COST=0.00000125\n" in updated

    def test_each_anchor_matches_exactly_once_in_the_live_environment_files(self):
        for environment_file_name in (".env", ".env.dev"):
            environment_file_path = REPOSITORY_ROOT / environment_file_name
            if not environment_file_path.exists():
                continue
            text = environment_file_path.read_text(encoding="utf-8")
            for name, (pattern, _) in MODEL_CONFIGURATION_ANCHORS.items():
                matches = re.findall(pattern, text, flags=re.MULTILINE)
                assert len(matches) == 1, f"{environment_file_name}: {name} matched {matches}"

    def test_validate_rejects_bad_costs_and_limits(self):
        with pytest.raises(SystemExit):
            ModelConfiguration(model_prompt_cost="not-a-number").validate()
        with pytest.raises(SystemExit):
            ModelConfiguration(model_completion_cost="-1").validate()
        with pytest.raises(SystemExit):
            ModelConfiguration(model_token_limit="0").validate()
        with pytest.raises(SystemExit):
            ModelConfiguration(model="   ").validate()
        ModelConfiguration(model="gpt-5.6-luna", model_prompt_cost="0.0000002").validate()

    def test_cost_is_written_as_a_plain_decimal_never_scientific(self):
        assert format_cost_for_environment_file(2e-07) == "0.0000002"
        assert format_cost_for_environment_file(0.0000012) == "0.0000012"
        assert format_cost_for_environment_file(0.0) == "0"

    def test_merged_over_fills_only_the_unset_fields(self):
        explicit = ModelConfiguration(model="gpt-5.6-luna")
        fallback = ModelConfiguration(
            model="gpt-5.4-nano", model_provider="OPEN_AI", model_prompt_cost="0.0000002"
        )
        merged = explicit.merged_over(fallback)
        assert merged.model == "gpt-5.6-luna"
        assert merged.model_provider == "OPEN_AI"
        assert merged.model_prompt_cost == "0.0000002"


class _FakeBuildResult:
    baseline_response_threshold = 51.25
    row_count = 161
    feature_width = 28
    key_phrase_count = 40


class TestBaselineProvenance:
    """The sidecar is what every later boot trusts; it must round-trip exactly."""

    def test_write_and_read_round_trip_with_prompt_fingerprint(self, tmp_path):
        provenance = build_baseline_provenance(
            ModelConfiguration(
                model="gpt-5.6-luna",
                model_provider="OPEN_AI",
                model_prompt_cost="0.0000002",
                model_completion_cost="0.0000012",
            ),
            question_count=161,
            conversation_count=161,
            build_result=_FakeBuildResult(),
        )
        sidecar = tmp_path / "corpus.meta.json"
        write_baseline_provenance(provenance, sidecar)
        read_back = read_baseline_provenance(sidecar)
        assert read_back == provenance
        assert read_back["model"] == "gpt-5.6-luna"
        assert read_back["baseline_response_threshold"] == 51.25
        assert read_back["style_system_prompt_sha256"] == hashlib.sha256(
            STYLE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()
        assert read_back["style_system_prompt_sha256"] == compute_style_system_prompt_sha256()

    def test_configuration_from_provenance_never_touches_the_token_limit(self):
        configuration = ModelConfiguration.from_provenance(
            {"model": "gpt-5.6-luna", "model_provider": "OPEN_AI", "model_prompt_cost": "0.0000002"}
        )
        assert configuration.model == "gpt-5.6-luna"
        assert configuration.model_token_limit is None
        assert ("MODEL_TOKEN_LIMIT", None) not in configuration.provided_items()

    def test_read_rejects_a_sidecar_without_a_model(self, tmp_path):
        sidecar = tmp_path / "corpus.meta.json"
        sidecar.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        with pytest.raises(SystemExit):
            read_baseline_provenance(sidecar)

    def test_tracked_sidecar_matches_the_feature_vector_contract(self):
        """A committed sidecar from a different vector version is a stale baseline."""
        from src.anubis.utils.dataset.style_features import (
            BASELINE_PROVENANCE_PATH,
            FEATURE_NAMES,
            STYLE_FEATURE_VECTOR_VERSION,
        )

        sidecar = REPOSITORY_ROOT / BASELINE_PROVENANCE_PATH
        if not sidecar.exists():
            pytest.skip(f"{sidecar.name} is not present (no retrain has run yet)")
        provenance = read_baseline_provenance(sidecar)
        assert provenance["style_feature_vector_version"] == STYLE_FEATURE_VECTOR_VERSION
        assert provenance["feature_width"] == len(FEATURE_NAMES)
        assert provenance["style_system_prompt_sha256"] == compute_style_system_prompt_sha256()

    def test_environment_model_matches_tracked_baseline(self):
        """The local tripwire for "bumped MODEL, forgot to retrain".

        Env files are untracked, so this skips in CI; locally, where the bump is
        made, a MODEL that differs from the committed sidecar fails here before
        the deployment silently scores against the previous model's cloud.
        """
        from dotenv import dotenv_values

        from src.anubis.utils.dataset.style_features import BASELINE_PROVENANCE_PATH

        sidecar = REPOSITORY_ROOT / BASELINE_PROVENANCE_PATH
        environment_file = REPOSITORY_ROOT / ".env.dev"
        if not sidecar.exists() or not environment_file.exists():
            pytest.skip("sidecar or .env.dev absent")
        environment = dotenv_values(environment_file)
        configured_model = (environment.get("MODEL") or "").strip()
        if not configured_model:
            pytest.skip("MODEL not set in .env.dev")
        provenance = read_baseline_provenance(sidecar)
        assert provenance["model"] == configured_model, (
            f".env.dev MODEL={configured_model!r} but the committed baseline was built "
            f"from {provenance['model']!r}; run scripts/retrain_chatgpt_baseline.py"
        )
        environment_threshold = environment.get("BASELINE_RESPONSE_THRESHOLD")
        if environment_threshold:
            assert float(environment_threshold) == provenance["baseline_response_threshold"]


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

    def test_legacy_generator_re_exports_the_same_frozen_prompt(self):
        from data.chat_gpt_baseline_dataset_creation import (
            STYLE_SYSTEM_PROMPT as legacy_prompt,
        )

        assert legacy_prompt == STYLE_SYSTEM_PROMPT


class _RecordingCursor:
    def __init__(self, statements):
        self.statements = statements
        self.rowcount = 1

    def execute(self, statement, parameters=None):
        self.statements.append((statement, parameters))

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        return False


class _RecordingConnection:
    def __init__(self):
        self.statements = []
        self.transactions_opened = 0

    def cursor(self):
        return _RecordingCursor(self.statements)

    def transaction(self):
        self.transactions_opened += 1
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        return False


class TestStoreCacheKeys:
    """The publish must cover every row the runtime reads."""

    def test_covers_every_cached_baseline_artifact(self):
        """A key missing here is an artifact that stays stale after a retrain.

        The runtime self-heals only on feature-vector WIDTH change, so a
        model-upgrade retrain (which keeps the width) relies entirely on this list
        being complete.
        """
        assert set(BASELINE_ARTIFACT_STORE_KEYS) == {
            "baseline_features_arr_list_str",
            "baseline_features_model_b64_pkl",
            "baseline_features_explainer_b64_pkl",
            "baseline_key_phrase_profile",
        }
        assert set(BASELINE_STORE_CACHE_KEYS) == set(BASELINE_ARTIFACT_STORE_KEYS) | {
            BASELINE_PROVENANCE_STORE_KEY
        }

    def test_every_key_is_read_by_the_runtime(self):
        """Each published key must actually appear in the code paths that read it."""
        runtime_sources = "\n".join(
            (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
            for relative_path in (
                "src/anubis/graph.py",
                "src/anubis/utils/utility.py",
                "src/anubis/utils/dataset/baseline_provenance.py",
            )
        )
        for cache_key in BASELINE_STORE_CACHE_KEYS + [BASELINE_RETRAIN_LOCK_STORE_KEY]:
            assert cache_key in runtime_sources, f"{cache_key} is read nowhere"

    def test_publish_upserts_all_five_rows_and_releases_the_lock_in_one_transaction(
        self, monkeypatch
    ):
        import scripts.retrain_chatgpt_baseline as retrain_module

        connection = _RecordingConnection()
        monkeypatch.setattr(retrain_module, "_connect_to_store", lambda context: connection)
        monkeypatch.setattr(
            retrain_module,
            "read_artifact_store_values_from_disk",
            lambda: {key: f"<{key}>" for key in BASELINE_ARTIFACT_STORE_KEYS},
        )
        provenance = {"model": "gpt-5.6-luna", "baseline_response_threshold": 51.25}

        publish_baseline_to_store(context=object(), provenance=provenance)

        assert connection.transactions_opened == 1
        upserts = [
            parameters
            for statement, parameters in connection.statements
            if statement == retrain_module.SQL_UPSERT_STORE_ROW
        ]
        assert [parameters[0] for parameters in upserts] == BASELINE_STORE_CACHE_KEYS
        for prefix, key, envelope in upserts:
            assert prefix == key
            assert set(json.loads(envelope)) == {"value"}
        provenance_envelope = json.loads(upserts[-1][2])["value"]
        assert json.loads(provenance_envelope) == provenance
        deletes = [
            parameters
            for statement, parameters in connection.statements
            if statement == retrain_module.SQL_DELETE_STORE_ROW
        ]
        assert deletes == [(BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY)]
