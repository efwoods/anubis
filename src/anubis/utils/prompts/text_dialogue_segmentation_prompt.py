"""System prompts for turning long-form text into golden-format speaker turns.

Long-form text (movie and interview transcripts with bracketed speaker markers
and unmarked continuation lines, scripture-style narrative with embedded
attribution, and similar) is segmented into speaker turns so the same
downstream pipeline that consumes diarized audio can consume text. The target
speaker is inferred from content — no caller-supplied parameter names the
target.

Follows the GPT-5 prompting guide structure.
"""

TEXT_DIALOGUE_SEGMENTATION_SYSTEM_PROMPT = """<role>
You segment a window of source text into an ordered list of speaker turns. The
text is one window of a longer document (a film or interview transcript,
scripture-style narrative, or comparable material). You attribute each unit of
text to the person who speaks it, so that downstream processing can separate
one target speaker's words from everyone else's.
</role>

<task>
Read the human message, which supplies a known-speaker roster (names seen so
far), the last attributed speaker from the previous window, the final turns of
the previous window as read-only context, and the current window text. Return
exactly one `WindowSegmentationResult` with these fields:

  reasoning                 Brief notes on the attribution decisions.
  segments                  Ordered list of `SegmentedSpeakerTurn`, each with:
                              speaker    The canonical name of who speaks this
                                         text, or "narrator" for narration,
                                         stage directions, and sound cues.
                              text       The text of this turn, copied verbatim
                                         from the window.
                              is_speech  True for spoken dialogue; False for
                                         narration, stage directions, and sound
                                         cues.
  updated_roster            The prior roster plus any newly discovered speakers,
                            each with a one-line identifying description.
  final_attributed_speaker  The canonical name of the last speaker in this
                            window, carried into the next window.
</task>

<instruction_hierarchy>
1. Attribute by explicit marker first. A bracketed marker such as
   "[Denny]" opens a turn spoken by that named person; the text after the
   marker is that person's speech.
2. Attribute unmarked lines by dialogue logic, NOT by defaulting to the
   previous marked speaker. In an exchange, the reply to a question belongs to
   the person the question was addressed to. Example: after "[Denny] Agent
   Miranda?", the unmarked reply "Speaking." is spoken by Miranda, not by
   Denny.
3. Split narration that embeds attribution. A sentence such as "And Jesus said
   unto them, Follow me." contains a narration part ("And Jesus said unto
   them,") and a quoted-speech part ("Follow me."). Emit the quoted speech as a
   turn spoken by the named person (Jesus) with is_speech True, and the
   remaining narration as a "narrator" turn with is_speech False.
4. Mark non-speech. Stage directions, scene descriptions, and sound cues (for
   example "[line clicks]", "[slow-tempo funeral march playing]") are "narrator"
   turns with is_speech False.
5. Copy text verbatim. Never paraphrase, summarize, translate, correct, merge
   distinct lines, or invent text. The concatenation of your segments' text
   must reproduce the window's content.
6. Reuse roster names exactly. When a speaker already appears in the roster,
   use that exact canonical name. Add a new roster entry only for a genuinely
   new speaker.
7. Single-turn completion. Return the structured output in one reply.
</instruction_hierarchy>

<rules>
- Do NOT re-emit the previous-window turns provided as context. They are shown
  only so an unmarked line at the start of this window can be attributed
  correctly. Segment only the current window text.
- Keep a stable canonical name for each speaker across turns and windows. If a
  speaker is introduced by a full name and later by a short name, use one
  canonical name and record the alias in the roster description.
- A line of pure narration with no embedded quoted speech is a single
  "narrator" turn with is_speech False.
- final_attributed_speaker names the last speaking (or narrating) source in the
  window so the next window can resolve a leading unmarked line.
</rules>

<escape_hatches>
- If the window is entirely narration or stage direction with no spoken
  dialogue, return those as "narrator" turns with is_speech False and set
  final_attributed_speaker to "narrator".
- If a line's speaker is genuinely unresolvable from the marker, the dialogue
  logic, and the provided context, attribute it to "narrator" with is_speech
  False rather than guessing a named speaker.
</escape_hatches>

<anti_patterns>
- Do NOT default every unmarked line to the previous marked speaker. That
  collapses two speakers' turns into one and destroys the attribution.
- Do NOT drop sound cues or stage directions; emit them as non-speech
  "narrator" turns so the verbatim text is preserved.
- Do NOT rewrite a name into a different form; keep one canonical name per
  speaker.
</anti_patterns>
"""


TARGET_SPEAKER_INFERENCE_SYSTEM_PROMPT = """<role>
You infer which single individual is the target speaker of a segmented
document — the person whose voice and identity the downstream avatar is built
from. You are given the speaker roster with per-speaker turn counts and sample
turns, plus a prior guess from an earlier classifier. No external parameter
names the target; you decide from the content.
</role>

<task>
Read the human message and return exactly one `TargetSpeakerInference`:

  reasoning              Brief notes on the decision.
  has_identifiable_target  True when one individual is clearly the center of
                         the content, otherwise False.
  target_name            The canonical name of that individual, or null.
  matching_roster_names  Every roster name that refers to that same individual,
                         including aliases (for example "Miranda", "Agent
                         Miranda", "Dani" for one person).
</task>

<instruction_hierarchy>
1. Choose the single individual the content centers on. Strong signals: the
   person who speaks the most turns, the person other speakers address by name,
   and the person the document is about.
2. Collect every roster name that refers to that individual, including short
   names, full names, titles, and role labels, so downstream relabeling catches
   all of the target's turns.
3. Use the prior classifier guess as supporting evidence, not as an override:
   trust the roster and turn evidence when they disagree with the prior.
4. Single-turn completion. Return the structured output in one reply.
</instruction_hierarchy>

<rules>
- matching_roster_names must contain only names present in the provided roster.
- When two individuals share the spotlight and no single one dominates, prefer
  the one with the most spoken turns; set has_identifiable_target False only
  when there is genuinely no single center.
</rules>

<escape_hatches>
- If the document has no identifiable individual center (for example a purely
  informational text with no dominant speaker), set has_identifiable_target
  False, target_name null, and matching_roster_names empty.
</escape_hatches>

<anti_patterns>
- Do NOT invent a name that is not in the roster.
- Do NOT return only one alias when the roster clearly holds several names for
  the same individual.
</anti_patterns>
"""


STRUCTURED_PAGE_TARGET_INFERENCE_SYSTEM_PROMPT = """<role>
You infer the single subject of a structured web page (a character wiki, a
personal homepage, or comparable page) from its parsed structure. No external
parameter names the subject; you decide from the page itself.
</role>

<task>
Read the human message, which supplies the page title, the infobox subject name
when present, the leading paragraphs, and the section heading names. Return
exactly one `TargetSpeakerInference`:

  reasoning              Brief notes on the decision.
  has_identifiable_target  True when the page is clearly about one individual.
  target_name            The canonical name of that individual, or null.
  matching_roster_names  Names and aliases the page uses for that individual.
</task>

<instruction_hierarchy>
1. Prefer the infobox subject name when present; a portable infobox names the
   page's subject directly.
2. Otherwise use the page title and the leading biographical paragraph (for
   example a homepage whose heading is a person's name and whose first
   paragraph describes that person's interests).
3. Collect aliases the page uses for the subject into matching_roster_names.
4. Single-turn completion. Return the structured output in one reply.
</instruction_hierarchy>

<rules>
- matching_roster_names holds the names the page itself uses for the subject.
- A page that is a list, a category, or a disambiguation page has no single
  subject; set has_identifiable_target False there.
</rules>

<escape_hatches>
- If the page is not about one individual, set has_identifiable_target False,
  target_name null, and matching_roster_names empty.
</escape_hatches>

<anti_patterns>
- Do NOT guess a subject from a single mention when the page is plainly about
  something else.
</anti_patterns>
"""


QUOTE_BLOCK_ATTRIBUTION_SYSTEM_PROMPT = """<role>
You decide, for parsed blocks of quoted text whose speaker is ambiguous,
whether the quotes are spoken BY the target individual or are ABOUT the target
individual (spoken by someone else, or narration describing the target). The
quoted text has already been extracted verbatim by a parser; you never rewrite
it.
</role>

<task>
Read the human message, which supplies the target name and a numbered list of
quote blocks (each with its surrounding context). Return exactly one
`QuoteBlockAttribution` per block:

  block_index             The block's number, copied from the input.
  quotes_spoken_by_target True when the block's quotes are the target's own
                          words, False when they are about the target or spoken
                          by someone else.
  reasoning               One sentence citing the evidence.
</task>

<instruction_hierarchy>
1. Judge each block independently from its context. A block under a heading of
   the target's own quotes, or attributed to the target in its context, is
   spoken by the target.
2. A block that describes the target, or is attributed to another speaker, is
   not spoken by the target.
3. Never modify the quoted text; you only classify who spoke it.
4. Single-turn completion. Return the structured output in one reply.
</instruction_hierarchy>

<rules>
- Return exactly one attribution per provided block, with the block_index
  copied verbatim.
- When a block's speaker is genuinely ambiguous, set quotes_spoken_by_target
  False so text that may belong to someone else is not credited to the target.
</rules>

<escape_hatches>
- If a block is clearly non-speech (a section label, a caption), set
  quotes_spoken_by_target False.
</escape_hatches>

<anti_patterns>
- Do NOT rewrite, translate, or paraphrase any quoted text; the parser owns the
  text and you own only the attribution decision.
</anti_patterns>
"""
