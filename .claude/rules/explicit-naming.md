---
description: Require explicit names in code, comments, and prompts; never use the pronoun it or other stand-ins
---

# Explicit naming

Name every subject. Do not use the pronoun "it" or other stand-in words in identifiers, comments, docstrings, commit messages, or prompt text.

Ambiguous references and abbreviations degrade how reliably a model follows an instruction. Repeat the noun every time.

## Identifiers

Spell names out fully so the name documents the meaning:

- Write `deep_agent_config`, not `da_config`
- Write `configuration`, not `cfg`
- Write `response`, not `resp`

Do not use acronyms or ad-hoc abbreviations. Standard loop indices (`i`, `j`) are allowed.

## Comments, prose, and prompts

Do not replace a named subject with: it, its, itself, this, that, these, those, they, them, their, the former, the latter, the value, the result, the thing.

Repeat the noun: `upload_payload`, `transcript_document`, `GlobalContext`.

```python
# Forbidden
processed = transform(data)  # handle it, then return the result

# Required
transcript_document = transform_upload_payload(upload_payload)
# Parse the upload payload, then return the transcript document.
```

The explicit-naming rule also applies to Anubis prompt text (`system_prompts.py`): instructions to a model must name the subject on every reference.
