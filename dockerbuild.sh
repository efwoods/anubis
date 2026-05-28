#!/usr/bin/env bash
set -euo pipefail

# Base image: FFmpeg + full pip install (keeps uv for the layer below).
docker build -t anubis-base:latest -f Dockerfile.anubis.base .

# Runtime image: refresh source on top of base, then strip uv/pip in final layers.
docker build -t evdev3/anubis-langgraph-api:latest .
