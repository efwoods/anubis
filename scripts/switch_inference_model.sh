#!/usr/bin/env bash
# scripts/switch_inference_model.sh
#
# Switch the inference model and rebuild the unmodified-inference-model style
# baseline in ONE command, running scripts/retrain_chatgpt_baseline.py inside the
# API container. The container is the right place because its interpreter carries
# the PINNED scikit-learn (the baseline pickles are not portable across releases)
# and its network resolves ASYNC_POSTGRES_STORE_URI (the committed URI names
# host.docker.internal, which does not resolve on the host).
#
# Usage:
#   ./scripts/switch_inference_model.sh --model gpt-5.6-luna \
#       --model-provider OPEN_AI \
#       --model-prompt-cost 0.0000002 --model-completion-cost 0.0000012
#
#   # Retrain against whatever MODEL the container already holds:
#   ./scripts/switch_inference_model.sh
#
#   # Sync this checkout (env files, threshold, store) from the committed sidecar,
#   # with no model calls — for a second checkout that pulled a retrain:
#   ./scripts/switch_inference_model.sh --configuration-only
#
#   # Also recreate the API container afterwards so it loads the rewritten env
#   # (compose reads env_file only when a container is CREATED):
#   ./scripts/switch_inference_model.sh --recreate --model gpt-5.6-luna ...
#
# Every argument other than --recreate is passed straight through to
# scripts/retrain_chatgpt_baseline.py (run it with --help for the full list).
#
# Environment overrides:
#   API_CONTAINER      container to run in (default: anubis-dev-langgraph-api-dev-1)
#   ENVIRONMENT_FILE   env file the script loads and checks (default: .env.dev; the
#                      rewrites always target BOTH .env and .env.dev)
#   COMPOSE_SERVICE    compose service to recreate with --recreate
#                      (default: langgraph-api-dev)
#   COMPOSE_FILE_ARGS  extra compose args, e.g. "-f docker-compose-prod.yml"
#
# Prod example (run from the prod checkout after pulling the committed retrain):
#   API_CONTAINER=anubis-langgraph-api-prod-1 ENVIRONMENT_FILE=.env \
#   COMPOSE_SERVICE=langgraph-api-prod COMPOSE_FILE_ARGS="-f docker-compose-prod.yml" \
#   ./scripts/switch_inference_model.sh --recreate --configuration-only
set -euo pipefail

cd "$(dirname "$0")/.."

API_CONTAINER="${API_CONTAINER:-anubis-dev-langgraph-api-dev-1}"
ENVIRONMENT_FILE="${ENVIRONMENT_FILE:-.env.dev}"
COMPOSE_SERVICE="${COMPOSE_SERVICE:-langgraph-api-dev}"
COMPOSE_FILE_ARGS="${COMPOSE_FILE_ARGS:-}"

recreate_after=false
passthrough_arguments=()
for argument in "$@"; do
    case "$argument" in
        --recreate) recreate_after=true ;;
        *) passthrough_arguments+=("$argument") ;;
    esac
done

if ! docker inspect --format '{{.State.Running}}' "$API_CONTAINER" 2>/dev/null | grep -q true; then
    echo "Container $API_CONTAINER is not running. Start the API (docker compose --env-file .env.dev up -d) or set API_CONTAINER." >&2
    exit 1
fi

echo "Running scripts/retrain_chatgpt_baseline.py in $API_CONTAINER (environment file $ENVIRONMENT_FILE)"
docker exec "$API_CONTAINER" python scripts/retrain_chatgpt_baseline.py \
    --environment-file "$ENVIRONMENT_FILE" "${passthrough_arguments[@]}"

# The container runs as root on a bind mount of this checkout, so every file the
# script (re)creates would otherwise come back root-owned and unreadable to the
# host user (and to git). Hand them back to the invoking user.
docker exec "$API_CONTAINER" chown "$(id -u):$(id -g)" \
    data/unmodified_inference_model_baseline_corpus.jsonl \
    data/unmodified_inference_model_baseline_corpus.meta.json \
    src/anubis/utils/dataset/baseline_features_arr.npy \
    src/anubis/utils/dataset/baseline_features_model_b64.pkl \
    src/anubis/utils/dataset/baseline_features_explainer_b64.pkl \
    src/anubis/utils/dataset/baseline_key_phrases.json \
    src/anubis/utils/context.py .env .env.dev 2>/dev/null || true

if [ "$recreate_after" = true ]; then
    echo "Recreating $COMPOSE_SERVICE so it loads the rewritten env"
    # shellcheck disable=SC2086 — COMPOSE_FILE_ARGS is intentionally word-split.
    docker compose $COMPOSE_FILE_ARGS --env-file "$ENVIRONMENT_FILE" up -d --force-recreate --no-deps "$COMPOSE_SERVICE"
else
    echo
    echo "Now recreate the API container so it loads the rewritten env:"
    echo "  make recreate_dev_api    (or re-run with --recreate)"
fi
