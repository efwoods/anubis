"""Keep invented captions and silent clips out of live-voice transcripts.

Whisper-style speech models were trained on subtitled video, so a clip that
holds no speech (a cough, a keyboard click, the trailing silence the browser
records after a person stops talking) is often "transcribed" as a memorised
caption: the Korean news sign-off ``MBC 뉴스 이덕영입니다``, ``Thank you for
watching``, ``Subtitles by the Amara.org community`` and similar. In voice mode
that text was shown as what the person said and answered by the avatar.

Two guards run on the live-voice paths (``/transcribe`` and the speaker-labelled
``diarize=true`` turn), never on uploaded media:

* :func:`measure_peak_volume_db` reads the clip's loudest sample with ffmpeg's
  ``volumedetect`` filter. A clip whose peak stays below the configured floor
  holds no speech and is never sent to the speech model.
* :func:`is_known_hallucination` recognises the memorised captions (and the
  advert-style ones: "Learn more at www.example.com") so a transcript made only
  of them is dropped after the call.
* :func:`keep_confident_segments` reads whisper's per-segment confidence
  (``no_speech_prob``, ``avg_logprob``, ``compression_ratio``) from a
  ``verbose_json`` response and keeps only the segments that hold real speech.
"""

from __future__ import annotations

import logging
import re
import subprocess
import unicodedata

logger = logging.getLogger(__name__)

# Captions speech models produce for silence or noise. Compared after
# normalisation (case-folded, punctuation and whitespace removed), so
# "Thank you for watching!" and "thank you for watching" are the same entry.
KNOWN_HALLUCINATED_CAPTIONS: frozenset[str] = frozenset(
    {
        # Korean broadcast sign-offs
        "MBC 뉴스 이덕영입니다",
        "MBC 뉴스",
        "MBC 뉴스 김성현입니다",
        "MBC 뉴스 이재민입니다",
        "MBC 뉴스 박진준입니다",
        "MBC 뉴스 정영훈입니다",
        "MBC 뉴스 이지선입니다",
        "MBC 뉴스 김지경입니다",
        "MBC 뉴스 임소정입니다",
        "MBC 뉴스 조의명입니다",
        "MBC 뉴스 정동욱입니다",
        "MBC 뉴스 이덕영",
        "KBS 뉴스",
        "SBS 뉴스",
        "뉴스 이덕영입니다",
        "시청해주셔서 감사합니다",
        "시청해 주셔서 감사합니다",
        "구독과 좋아요 부탁드립니다",
        # English subtitle credits and sign-offs
        "Thank you for watching",
        "Thanks for watching",
        "Thank you for watching!",
        "Thank you so much for watching",
        "Thanks for watching and see you next time",
        "Please subscribe",
        "Please like and subscribe",
        "Subscribe to my channel",
        "See you in the next video",
        "Subtitles by the Amara.org community",
        "Subtitles by Amara.org",
        "Subtitles created by the Amara.org community",
        "Transcribed by",
        "Transcription by CastingWords",
        "Copyright WDR",
        "Thank you for joining us",
        "Thanks for joining us",
        "Thank you for listening",
        "Thanks for listening",
        "We'll see you next time",
        "We'll see you next time. Bye for now.",
        "See you next time",
        "See you next time. Bye for now.",
        "Bye for now",
        "That's all for today",
        "Stay tuned",
        "You",
        # Japanese
        "ご視聴ありがとうございました",
        "ご視聴ありがとうございます",
        "字幕",
        "字幕視聴ありがとうございました",
        # Chinese
        "字幕由Amara.org社区提供",
        "由Amara.org社区提供字幕",
        "谢谢观看",
        "謝謝觀看",
        "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
        # European
        "Sous-titres réalisés par la communauté d'Amara.org",
        "Sous-titrage Société Radio-Canada",
        "Sous-titrage ST' 501",
        "Untertitel im Auftrag des ZDF",
        "Untertitel von Stephanie Geiges",
        "Untertitelung des ZDF",
        "Untertitelung aufgrund der Amara.org-Community",
        "Sottotitoli creati dalla comunità Amara.org",
        "Sottotitoli e revisione a cura di QTSS",
        "Legendas pela comunidade Amara.org",
        "Subtítulos realizados por la comunidad de Amara.org",
        "Subtítulos por la comunidad de Amara.org",
        "Tack för att du tittade",
        "Tekstet av Nicolai Winther",
        "Продолжение следует",
        "Субтитры сделал DimaTorzok",
        "Субтитры создавал DimaTorzok",
        "Редактор субтитров А.Семкин Корректор А.Егорова",
        # Arabic / Turkish / Dutch / Polish
        "اشتركوا في القناة",
        "İzlediğiniz için teşekkürler",
        "Altyazı M.K.",
        "Ondertiteld door de Amara.org gemeenschap",
        "Napisy stworzone przez społeczność Amara.org",
    }
)

_NORMALISE_PATTERN = re.compile(r"[\W_]+", re.UNICODE)

# Advert and credit captions the models invent around web addresses and
# broadcast sign-offs. Nobody dictates "Learn more at www.example.com" to
# their avatar; the model is remembering a video's end card.
_ADVERT_CAPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:www\.|https?://)\S+", re.IGNORECASE),
    re.compile(r"\.(?:com|org|net|co\.uk|kr|jp)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:to\s+)?learn\s+more\b", re.IGNORECASE),
    re.compile(r"\bto\s+learn\s+more\s*[.!]?\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*(?:please\s+)?(?:see|read)\s+the\s+(?:complete|full)\s+disclaimer\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+)?visit\b.*\b(?:for|to)\s+(?:more|learn)\b", re.IGNORECASE
    ),
    re.compile(r"^\s*for\s+more\s+information\b", re.IGNORECASE),
    re.compile(
        r"^\s*(?:subtitles?|captions?|transcripts?)\s+(?:by|provided|created|made)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:don't\s+forget\s+to\s+)?(?:like|subscribe|hit\s+the\s+bell)\b.*\b(?:subscribe|channel|bell|notifications?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*copyright\b|©", re.IGNORECASE),
)


def normalise_caption(text: str) -> str:
    """Lower-case the text and strip every non-letter, non-digit character."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return _NORMALISE_PATTERN.sub("", folded)


_NORMALISED_HALLUCINATIONS: frozenset[str] = frozenset(
    normalise_caption(caption) for caption in KNOWN_HALLUCINATED_CAPTIONS
)


def is_known_hallucination(text: str) -> bool:
    """Report whether the whole text is one memorised caption, possibly repeated.

    Speech models sometimes emit the same caption several times in a row for a
    longer silent clip ("MBC 뉴스 이덕영입니다. MBC 뉴스 이덕영입니다."), so a
    transcript is also a hallucination when every sentence of the transcript is
    a known caption.
    """
    normalised = normalise_caption(text)
    if not normalised:
        return True
    if normalised in _NORMALISED_HALLUCINATIONS:
        return True
    if any(pattern.search(text or "") for pattern in _ADVERT_CAPTION_PATTERNS):
        return True
    sentences = [
        piece for piece in re.split(r"[.!?。！？\n]+", text or "") if piece.strip()
    ]
    if len(sentences) > 1 and all(
        normalise_caption(sentence) in _NORMALISED_HALLUCINATIONS
        for sentence in sentences
    ):
        return True
    return False


def _segment_value(segment: object, name: str, default: float) -> float:
    if isinstance(segment, dict):
        value = segment.get(name, default)
    else:
        value = getattr(segment, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment_text(segment: object) -> str:
    if isinstance(segment, dict):
        return str(segment.get("text") or "")
    return str(getattr(segment, "text", "") or "")


def keep_confident_segments(
    segments: list[object] | None,
    *,
    no_speech_probability_max: float,
    average_logprob_min: float,
    compression_ratio_max: float,
) -> tuple[str, int]:
    """Join the whisper segments that hold real speech; return (text, dropped_count).

    Whisper's ``verbose_json`` response carries, per segment, the probability
    that the segment is not speech, the mean token log-probability, and the
    gzip compression ratio of the text (a repeated caption compresses very
    well). A segment is dropped when the segment is probably not speech, the
    model was unsure of the words, the text repeats, or the text is one of the
    memorised captions. A threshold of 0 or below (or 0 or above for the
    log-probability) disables that check.
    """
    kept: list[str] = []
    dropped = 0
    for segment in segments or []:
        text = _segment_text(segment).strip()
        if not text:
            continue
        no_speech = _segment_value(segment, "no_speech_prob", 0.0)
        logprob = _segment_value(segment, "avg_logprob", 0.0)
        compression = _segment_value(segment, "compression_ratio", 1.0)
        reasons: list[str] = []
        if no_speech_probability_max > 0.0 and no_speech >= no_speech_probability_max:
            reasons.append(f"no_speech_prob {no_speech:.2f}")
        if average_logprob_min < 0.0 and logprob <= average_logprob_min:
            reasons.append(f"avg_logprob {logprob:.2f}")
        if compression_ratio_max > 0.0 and compression >= compression_ratio_max:
            reasons.append(f"compression_ratio {compression:.2f}")
        if is_known_hallucination(text):
            reasons.append("memorised caption")
        if reasons:
            dropped += 1
            logger.info(
                "Dropped a live-voice segment (%s): %r", ", ".join(reasons), text
            )
            continue
        kept.append(text)
    return " ".join(kept).strip(), dropped


def drop_hallucinated_text(text: str, *, description: str = "transcript") -> str:
    """Return the text, or an empty string when the text is a memorised caption."""
    if is_known_hallucination(text):
        if (text or "").strip():
            logger.info(
                "Dropped a hallucinated %s from the speech model: %r", description, text
            )
        return ""
    return text


_VOLUME_PATTERN = re.compile(r"(max|mean)_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def measure_peak_volume_db(
    audio_path: str, ffmpeg_executable: str
) -> tuple[float, float] | None:
    """Return ``(max_volume_db, mean_volume_db)`` of the clip, or None on failure.

    Uses ffmpeg's ``volumedetect`` filter, so no audio library is decoded in
    Python. Values are in dBFS; digital silence reports ``-91.0`` dB. A failure
    to measure returns None and the caller proceeds as if the clip were loud,
    so a broken measurement never silences a real utterance.
    """
    command = [
        ffmpeg_executable,
        "-nostdin",
        "-hide_banner",
        "-i",
        audio_path,
        "-af",
        "volumedetect",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("Could not measure the clip volume", exc_info=True)
        return None
    output = completed.stderr.decode("utf-8", "replace")
    found = {name: float(value) for name, value in _VOLUME_PATTERN.findall(output)}
    if "max" not in found:
        return None
    return found["max"], found.get("mean", found["max"])


def clip_is_silent(
    audio_path: str,
    ffmpeg_executable: str,
    *,
    silence_max_volume_db: float | None,
) -> bool:
    """Report whether the clip's loudest sample is below the configured floor.

    ``silence_max_volume_db`` of None (or a non-negative value) disables the
    gate: nothing recorded through a microphone is louder than 0 dBFS.
    """
    if silence_max_volume_db is None or float(silence_max_volume_db) >= 0.0:
        return False
    measured = measure_peak_volume_db(audio_path, ffmpeg_executable)
    if measured is None:
        return False
    peak_db, mean_db = measured
    silent = peak_db < float(silence_max_volume_db)
    if silent:
        logger.info(
            "Skipping a silent live-voice clip (peak %.1f dB, mean %.1f dB, floor %.1f dB)",
            peak_db,
            mean_db,
            float(silence_max_volume_db),
        )
    return silent
