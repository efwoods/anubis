FROM anubis-base:latest

# -- Force uv to COPY files instead of hardlinking. The deps below are installed
#    on top of the base image, whose package files live in a lower overlayfs
#    layer. uv's default hardlink link-mode truncates those lower-layer source
#    files to 0 bytes when it reinstalls a package, leaving valid .pyc shadowed
#    by an empty .py (e.g. langchain_core/load/load.py → "cannot import name
#    'Reviver'"). Copy mode writes real files and avoids the corruption.
ENV UV_LINK_MODE=copy

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
# RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license \
#     && touch /api/langgraph_api/__init__.py \
#              /api/langgraph_runtime/__init__.py \
#              /api/langgraph_license/__init__.py
# RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --

# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

WORKDIR /deps/anubis