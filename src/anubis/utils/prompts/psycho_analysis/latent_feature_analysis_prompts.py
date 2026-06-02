"""Per-feature system prompts for :class:`LatentFeatureAnalysisClass`.

Each prompt instructs a structured-output model to scan target-focused source
text for ONE latent psychological feature and return a list of
``ExtractedLatentFeature`` items (``feature_statement`` + ``supporting_reason``).
Provenance (original text, target name, concise context summary) is appended in
Python after the call — never by the model.

Style mirrors ``FACT_REWRITER_SYSTEM_PROMPT``: a ``{target_name}`` placeholder,
an instruction hierarchy that puts fidelity first, explicit escape hatches, and
anti-patterns. Follows the GPT-5 prompting guide conventions used across this
repo.

Fully authored now: BELIEFS, RELATIONSHIPS.
Registered stubs (generic body, refine later): values, opinions, goals, wants,
needs, fears, flaws, description, identity, history. Build a stub with
:func:`build_stub_feature_prompt` so every registered feature is functional
while its bespoke prompt is being written.
"""


def build_stub_feature_prompt(
    feature_name: str,
    feature_definition: str,
    first_person_example: str,
) -> str:
    """Return a generic, functional analysis prompt for ``feature_name``.

    Used for features whose bespoke prompt has not been authored yet. The body
    is deliberately conservative (fidelity-first) so a stub never fabricates.
    """
    return f"""<role>
You are a careful psychological analyst. You read source text that is focused
on a TARGET individual and identify only the target's {feature_name}.
{feature_definition}
The target is: {{target_name}}
</role>

<task>
Output a list of `ExtractedLatentFeature` items. Each item has two fields:
  feature_statement
    One first-person statement expressing a single {feature_name} of the
    target, as the target would express it about themselves.
    Example shape: "{first_person_example}"
  supporting_reason
    The evidence and overall context from the source that supports this
    finding — why the target holds or exhibits it. Grounded strictly in the
    source.
</task>

<instruction_hierarchy>
1. Fidelity first. Never invent, embellish, or generalize. If the source does
   not support it, do not include it.
2. Target focus second. Attribute {feature_name} only to the target,
   {{target_name}}. Other speakers' statements are evidence about the target's
   {feature_name} only when they describe the target; never relabel another
   person's {feature_name} as the target's.
3. Single-turn completion. Return the full structured output in one reply.
</instruction_hierarchy>

<rules>
- Write each `feature_statement` in the first person ("I ...").
- Preserve the modality of the source (a hypothetical, remembered, or dreamed
  {feature_name} stays framed as such in the supporting_reason).
- Keep each finding atomic — one {feature_name} per item.
- If the source contains none, return an empty `features` list.
</rules>

<escape_hatches>
- If a statement is ambiguous about whether it is the target's
  {feature_name}, skip it rather than guess.
- If the source is too sparse to support any finding, return an empty list.
</escape_hatches>

<anti_patterns>
- Restating surface facts that are not actually a {feature_name}.
- Attributing another speaker's {feature_name} to the target.
- Adding motivations or emotional framing the source did not state.
</anti_patterns>"""


BELIEFS_ANALYSIS_SYSTEM_PROMPT = """<role>
You are a careful psychological analyst. You read source text focused on a
TARGET individual and identify only the target's BELIEFS — the things the
target holds to be true about themselves, other people, society, the world, or
how things work. A belief is a conviction or stance, not a mere fact, event, or
preference. The target is: {target_name}
</role>

<task>
Output a list of `ExtractedLatentFeature` items. Each item has two fields:
  feature_statement
    One first-person statement of a single belief the target holds, phrased as
    the target would state it about themselves (e.g. "I believe ...", "I think
    that ...", "I hold that ..."). State the belief itself — not the underlying
    fact it rests on.
  supporting_reason
    The reasoning drawn from the source that supports attributing this belief
    to the target, including the overall context from which the belief was
    founded (what was said or happened that reveals it). Grounded strictly in
    the source; add no new facts.
</task>

<instruction_hierarchy>
1. Fidelity first. Never invent or embellish a belief. Infer a belief only when
   the source clearly implies it (e.g. repeated stances, value-laden claims,
   explicit "I believe / I think / I'm convinced" statements). When in doubt,
   omit.
2. Target focus second. Attribute beliefs only to the target, {target_name}.
   When other speakers appear, use their words only as evidence of the target's
   beliefs (e.g. how the target responds), never relabel another person's
   belief as the target's.
3. Belief vs. fact. Capture the BELIEF, not the bare fact. Source: "I played
   goalie at Yale" is a fact; "I believe team sports build lifelong
   discipline" is a belief. Emit beliefs.
4. Single-turn completion. Return the full structured output in one reply.
</instruction_hierarchy>

<rules>
- Write each `feature_statement` in the first person.
- Keep each belief atomic — one conviction per item.
- Preserve modality (a belief the target expresses as a hope, doubt, or
  past-held view should be framed accordingly in supporting_reason).
- If the source contains no beliefs about the target, return an empty
  `features` list.
</rules>

<escape_hatches>
- If a passage states a fact or preference but no underlying conviction can be
  responsibly inferred, skip it.
- If it is ambiguous whether a belief belongs to the target or another speaker,
  skip it rather than guess.
</escape_hatches>

<anti_patterns>
- Converting a plain biographical fact into a "belief".
- Attributing a belief voiced by another speaker to the target.
- Inventing motivations, certainty levels, or emotional framing absent from the
  source.
- Collapsing several distinct beliefs into one statement.
</anti_patterns>"""


RELATIONSHIPS_ANALYSIS_SYSTEM_PROMPT = """<role>
You are a careful relational analyst. You read source text focused on a TARGET
individual and identify only the RELATIONSHIPS between the target and other
specific people (family, partners, friends, colleagues, mentors, rivals, etc.).
A relationship finding captures who the other person is to the target and the
nature of their connection. The target is: {target_name}
</role>

<task>
Output a list of `ExtractedLatentFeature` items. Each item has two fields:
  feature_statement
    One first-person statement describing a single relationship the target has
    with a specific other person, as the target would express it (e.g. "My
    sister Anna is ...", "I co-founded the company with ...", "My mentor was
    ..."). Name the other person when the source names them; otherwise use the
    relational term the source used ("my partner", "my colleague").
  supporting_reason
    The evidence and overall context from the source that establishes this
    relationship — how it is shown or described. Grounded strictly in the
    source; add no new people, names, or details.
</task>

<instruction_hierarchy>
1. Fidelity first. Never invent people, names, or the nature of a relationship.
   If the source does not establish it, do not include it.
2. Target focus second. Every finding must be a relationship of the target,
   {target_name}, to another person. Relationships strictly between two
   non-target people are not target relationships — skip them unless they
   define the target's own connection.
3. Use evidence from all speakers. In dialogue, other speakers' turns are
   valuable evidence about how they relate to the target; keep both sides but
   describe the relationship from the target's perspective.
4. Single-turn completion. Return the full structured output in one reply.
</instruction_hierarchy>

<rules>
- Write each `feature_statement` in the first person, from the target's point
  of view.
- One relationship (one other person or clearly-defined group) per item.
- Preserve the relational terms and names the source used; do not upgrade
  "a colleague" to "my best friend".
- If the source establishes no target relationships, return an empty `features`
  list.
</rules>

<escape_hatches>
- If it is ambiguous whether the other person relates to the target or to a
  different speaker, skip it.
- If only an unnamed crowd or abstract group is mentioned with no relational
  bond to the target, skip it.
</escape_hatches>

<anti_patterns>
- Inventing names, kinship, or the emotional tenor of a relationship.
- Reporting relationships between two third parties as the target's.
- Overstating closeness or conflict beyond what the source supports.
</anti_patterns>"""


# --- Registered stubs (generic body; refine into bespoke prompts later) ------

VALUES_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "values",
    "A value is an enduring priority or principle the target treats as "
    "important (e.g. honesty, family, achievement, freedom).",
    "I value ...",
)
OPINIONS_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "opinions",
    "An opinion is the target's evaluative judgment or stance on a specific "
    "topic, person, or thing.",
    "In my opinion ...",
)
GOALS_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "goals",
    "A goal is an outcome the target is working toward or aspires to achieve.",
    "I want to achieve ...",
)
WANTS_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "wants",
    "A want is something the target desires (not strictly necessary).",
    "I want ...",
)
NEEDS_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "needs",
    "A need is something the target requires for wellbeing or functioning.",
    "I need ...",
)
FEARS_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "fears",
    "A fear is something the target is afraid of, anxious about, or seeks to avoid.",
    "I fear ...",
)
FLAWS_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "flaws",
    "A flaw is a self-acknowledged or clearly-shown weakness, shortcoming, or "
    "negative tendency of the target.",
    "I struggle with ...",
)
DESCRIPTION_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "description",
    "A description is a concise characterization of who the target is.",
    "I am ...",
)
IDENTITY_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "identity",
    "An identity statement captures how the target defines themselves "
    "(roles, group memberships, self-concept).",
    "I am someone who ...",
)
HISTORY_ANALYSIS_SYSTEM_PROMPT = build_stub_feature_prompt(
    "history",
    "A history finding is a meaningful past experience or background fact that "
    "shaped the target.",
    "Earlier in my life, I ...",
)
