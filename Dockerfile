FROM anubis-base:latest

# -- Add full source (replaces the stub left by the base image) --
ADD . /deps/anubis

# -- Install only net-new / changed deps; uv skips already-satisfied packages --
#    No --no-deps: full resolution is required so uv can diff against the
#    already-installed base layer and fetch only what's missing.

RUN for dep in /deps/*; do \
        if [ -d "$dep" ]; then \
            echo "Installing $dep"; \
            (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir \
                --link-mode=copy -c /api/constraints.txt -e .); \
        fi; \
    done

    
ENV LANGGRAPH_STORE='{"index": {"dims": 640, "embed": "huggingface:microsoft/harrier-oss-v1-270m", "fields": ["document.kwargs.page_content"]}}'
ENV LANGGRAPH_HTTP='{"app": "/deps/anubis/src/api/webapp.py:app"}'
ENV LANGSERVE_GRAPHS='{"Anubis": "/deps/anubis/src/anubis/graph.py:graph"}'

# -- Ensure user deps didn't inadvertently overwrite langgraph-api --
# The install above resolves the whole dependency graph, and langgraph-cli's
# "inmem" extra declares `langgraph-api>=0.5.35`, so the resolver is free to
# fetch the PUBLIC langgraph-api from PyPI and write it over the licensed build
# this image ships (installed editable from /api). The two carry the same
# version number and not the same code: the licensed langgraph_runtime_postgres
# under /storage imports names the public build never defines
# (PREFER_GRPC_CHECKPOINTER), so the server dies at boot with an ImportError
# before a single request is served. /api/forbidden.txt states the same rule
# from the image's own side: "Block user overrides of the base api".
#
# Reinstalling /api editable puts the licensed package back in front of
# site-packages; --no-deps so restoring it moves nothing else. This must stay
# ahead of the layers that strip uv and pip, which is the only window left for
# repairing an install.
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --

# NOTE: no `playwright install` here — the Playwright-managed Chromium
# download lacks the system shared libraries on wolfi (and `--with-deps`
# is apt-based, unavailable under apk). The base image instead installs
# the wolfi `chromium` apk package and sets
# BROWSER_CHROMIUM_EXECUTABLE_PATH so the browser tools launch that binary
# (see Dockerfile.anubis.base and src/anubis/utils/tools/browser/).

# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

WORKDIR /deps/anubis