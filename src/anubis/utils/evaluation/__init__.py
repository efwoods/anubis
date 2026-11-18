"""Local evaluation of an avatar against the avatar's own real transcripts.

``transcript_evaluation`` builds question and answer examples from the rows the
upload pipeline already stores, scores each generated reply for accuracy and
for verisimilitude, and renders the report. Nothing here sends a trace to
LangSmith.
"""
