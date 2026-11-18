"""Speech helpers that sit beside the OpenAI speech path in ``utility.py``.

``published_transcripts`` reads a speaker-labelled transcript that the source
of a recording publishes, so the recording is never sent to a paid diarizer.
``diarization_providers`` sends a recording to a hosted diarizer other than
OpenAI's and returns the response in the shape ``transcribe_audio_diarize``
returns.
"""
