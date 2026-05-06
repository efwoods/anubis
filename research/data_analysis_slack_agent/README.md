# Research: Slack-Native Data Analysis Agent for Anubis

Research bundle for building a personal + company "state-of-the-business" deep agent on top of [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/data-analysis), reachable from Slack, that summarizes activity across nine data sources (sleep, BoA, Google Meets, Slack, git, Cursor, ChatGPT, Azure, AWS) into an After-Action Report.

> Research only. Nothing in `src/` is modified. All code below is illustrative.

## Read in this order

1. [`00_overview.md`](00_overview.md) — goal, why deep agents, runtime topology, mapping to existing anubis architecture, roadmap.
2. [`01_data_source_matrix.md`](01_data_source_matrix.md) — per-source connector recipe (lib, auth, schema, frequency).
3. [`02_context_additions.md`](02_context_additions.md) — exact `GlobalContext` field additions following `.cursorrules`, plus matching `.env` keys.
4. [`03_slack_interface.md`](03_slack_interface.md) — Bolt-Python entrypoints, FastAPI mount, `slack_send_message` tool.
5. [`04_aar_prompting.md`](04_aar_prompting.md) — system-prompt structure for the After-Action Report and the matching Pydantic schema.
6. [`05_proposed_layout.md`](05_proposed_layout.md) — file tree the implementation should land at.

## Reference code (illustrative skeletons)

- [`reference_code/data_analysis_graph.py`](reference_code/data_analysis_graph.py) — `create_deep_agent(...)` wiring.
- [`reference_code/slack_bolt_app.py`](reference_code/slack_bolt_app.py) — Slack adapter (Socket Mode + FastAPI).
- [`reference_code/connectors/aws_cost.py`](reference_code/connectors/aws_cost.py) — AWS Cost Explorer connector pattern.
- [`reference_code/connectors/git_local.py`](reference_code/connectors/git_local.py) — Local git log connector pattern.
- [`reference_code/connectors/oura_sleep.py`](reference_code/connectors/oura_sleep.py) — Oura sleep connector pattern.

## Skills (progressive-disclosure SOPs)

- [`skills/aar/SKILL.md`](skills/aar/SKILL.md) — the AAR SOP.
- [`skills/sleep_analysis/SKILL.md`](skills/sleep_analysis/SKILL.md) — sleep scoring SOP.
- [`skills/cloud_cost/SKILL.md`](skills/cloud_cost/SKILL.md) — cloud + LLM cost SOP.
