"""Ensure the NLTK corpora the stylometric feature extractor needs exist locally.

The mirror of :mod:`src.anubis.utils.huggingface_prefetch` for NLTK data. Every
reply scored with ``include_quality_metrics`` is tokenized and part-of-speech
tagged by :func:`src.anubis.utils.dataset.style_features.extract_style_features`,
which needs the four corpora listed below. Downloading them lazily on the first
request made that request sit for minutes behind the download while the
streaming endpoint emitted nothing but keepalive frames, so the download is
pulled forward to process startup (and baked into the image; see
``Dockerfile.anubis.base``).
"""

from __future__ import annotations

import logging
import socket
import threading

logger = logging.getLogger(__name__)

# Each entry is (package name passed to ``nltk.download``, locator passed to
# ``nltk.data.find``). The two differ — ``punkt`` downloads as a package but
# resolves under ``tokenizers/`` — so both spellings are kept explicitly rather
# than derived. This tuple is the single source of truth: the runtime extractor
# path and the startup prefetch both read it.
NLTK_RESOURCES: tuple[tuple[str, str], ...] = (
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ("stopwords", "corpora/stopwords"),
)

# ``nltk.download`` exposes no timeout parameter, and the ``urllib`` transport
# underneath inherits the process-wide socket default — which is "block
# forever". The four corpora above total roughly 20 MB, so a throttled or
# unreachable CDN wedges the caller indefinitely rather than failing. Bounding
# the socket default for the duration of the download converts that indefinite
# hang into a bounded, retryable failure. The value is per-socket-operation, not
# for the whole transfer, so it only has to exceed the longest expected stall
# between packets. Promote to GlobalContext if a deployment ever needs to tune
# this.
_NLTK_DOWNLOAD_SOCKET_TIMEOUT_SECONDS = 60.0

_prefetch_lock = threading.Lock()
_prefetch_done = False


def _missing_resource_package_names() -> list[str]:
    """Return the package names from :data:`NLTK_RESOURCES` not resolvable on disk."""
    import nltk

    missing: list[str] = []
    for package_name, locator in NLTK_RESOURCES:
        try:
            nltk.data.find(locator)
        except LookupError:
            missing.append(package_name)
    return missing


def ensure_nltk_corpora_cached() -> None:
    """Download any of :data:`NLTK_RESOURCES` missing from the local NLTK data path.

    Safe to call repeatedly and from multiple threads: the lock keeps concurrent
    callers from each starting their own copy of the same download. Unlike the
    Hugging Face prefetch this does NOT latch as done after a failed attempt —
    a transient network failure at startup must not permanently disable the
    retry that the extractor path depends on, so the completion flag is set only
    once every resource actually resolves.
    """
    global _prefetch_done
    with _prefetch_lock:
        if _prefetch_done:
            return

        import nltk

        missing = _missing_resource_package_names()
        if missing:
            previous_socket_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(_NLTK_DOWNLOAD_SOCKET_TIMEOUT_SECONDS)
            try:
                for package_name in missing:
                    logger.info("Downloading missing NLTK corpus: %s", package_name)
                    if not nltk.download(package_name, quiet=True):
                        logger.warning(
                            "NLTK corpus download reported failure: %s", package_name
                        )
            except Exception:
                # Best-effort: a failed download leaves the corpus missing, which
                # the next call retries. Raising here would take down startup over
                # metadata enrichment.
                logger.warning(
                    "NLTK corpus download raised; stylometric features will retry "
                    "on the next call.",
                    exc_info=True,
                )
            finally:
                socket.setdefaulttimeout(previous_socket_timeout)

        _prefetch_done = not _missing_resource_package_names()
