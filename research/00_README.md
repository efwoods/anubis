# Anubis — Scaling Research

This folder contains the research produced for the question:

> "I need a plan to scale this API. I want N standalone single-avatar
> agents that run on the AWS Free Tier (1 GB RAM; we're currently at
> 1.3 GB). Then I need a plan to scale the API on Azure (Postgres + Redis +
> AKS), with cost projections that drop into the Kickstarter budget."

Read in order:

| # | File | What it answers |
| --- | --- | --- |
| 01 | [`01_current_state_audit.md`](./01_current_state_audit.md) | What's *actually* running today, where the 1.3 GB lives, what's persistent. |
| 02 | [`02_phase1_aws_free_tier_single_avatar.md`](./02_phase1_aws_free_tier_single_avatar.md) | AWS Free Tier specs (Track A vs the new July-2025 \$200-credit Track B), storage limits, sealed-VM vs shared-DB topologies, and a memory diet to actually fit on 1 GB. |
| 03 | [`03_phase2_azure_kubernetes_scale_plan.md`](./03_phase2_azure_kubernetes_scale_plan.md) | Azure AKS + Postgres Flexible Server + Cache for Redis sizing, LangSmith Agent Server / Control Plane / Self-Hosted choices, migration plan. |
| 04 | [`04_token_workload_cost_model.md`](./04_token_workload_cost_model.md) | Token math from the user's stated workload (72 tok/msg × 10 msg × 30 days), naive vs agent-amplified cost, fleet projections at 1 / 100 / 1 000 / 10 000 avatars. |
| 05 | [`05_kickstarter_budget_lineitems.md`](./05_kickstarter_budget_lineitems.md) | Concrete spreadsheet rows for the Kickstarter budget at three raise scenarios (\$2.5K / \$8K / \$25K). |

---

## Executive summary (TL;DR)

### The 1 GB problem is solvable, but only by moving the embedding model out of process

The container holds a **270 M-param embedding model** (`microsoft/harrier-oss-v1-270m`,
~540 MB FP16) and a **82 M-param sentiment classifier** (~328 MB FP32) inside
Python. Together they are ~70 % of the 1.3 GB resident.

To fit on 1 GB:

1. **Swap the in-process embedder for an API call** —
   `text-embedding-3-small` at \$0.02/M tokens. Frees ~540 MB.
2. **Cache the sentiment pipeline** as a module-level singleton, or feature-
   flag it off for the lite image. Frees ~328 MB.
3. **Split the Docker image** into `anubis-agent` (runtime only, ~600 MB
   resident) and `anubis-ingest` (pipeline only, runs on demand).

After this, a `t3.micro` (1 GB RAM) holds the Anubis runtime with
~250 MB headroom. See §2.3 of the Phase-1 doc.

### Phase 1 topology — single VM fleet, shared DB, free for 6+ months

```
1 × t3.small Track-B VM
  ├── docker compose: N × anubis-agent containers (one per avatar)
  └── shared redis:7-alpine
            │
            └─→ 1 × RDS db.t3.micro (Postgres 16 + pgvector, 20 GB free)
            └─→ 1 × S3 bucket (5 GB free)
            └─→ OpenAI (embeddings + LLM) + LangSmith Plus (\$40/mo)
```

Free Tier hard ceilings (per AWS account):

- EC2 hours: **750/mo** (Track A) or **\$200 of credits** (Track B, post-2025-07-15)
- EBS attached storage: **30 GB total**
- S3 Standard: **5 GB**
- RDS db.t3.micro: **20 GB SSD**

That account holds **~50 – 80 fully-built avatars** in Postgres, **~16
avatars' worth of media** in S3, and **~5 simultaneously-running avatar
containers** on a single t3.small.

### Phase 2 topology — Azure / AKS / LangSmith Agent Server

Funded by the Kickstarter. Use **LangSmith Cloud Plus** as the control-plane
(\$39/seat/mo, includes 10 K traces and 1 dev deployment), our own Azure
Postgres + Redis for data sovereignty.

Cluster sized for ≤ 100 avatars: **~\$215/mo Azure + \$28/mo OpenAI = ~\$243/mo total**.
Cluster sized for ≤ 1 000 avatars: **~\$585/mo Azure + \$301/mo OpenAI = ~\$886/mo total**.

Self-hosting the **full LangSmith Control Plane** in Azure adds ~\$1 100/mo
of infra (16 vCPU / 64 GB AKS + 8 GB Postgres + 4 GB Redis + 32 GB
ClickHouse) and requires Enterprise license — recommended only after annual
revenue clears \$50K.

### Workload economics

User's stated baseline: 72 tok/msg × 10 msg/conv × 1 conv/day × 30 days.
After agent amplification (system prompt, identity context, `think` +
`respond` passes, retrieval), the realistic cost on **gpt-4o-mini** is:

| Scale | Inference | Embeddings | LangSmith overage | **Total LLM \$/mo** |
| --- | --- | --- | --- | --- |
| 1 avatar | \$0.27 | \$0.01 | \$0 | **\$0.28** |
| 100 | \$27 | \$1 | \$0 | **\$28** |
| 1 000 | \$270 | \$6 | \$25 | **\$301** |
| 10 000 | \$2 700 | \$60 | \$295 | **\$3 055** |

Per-avatar all-in (infra + LLM + observability), at **1 000 active backers**,
is roughly **\$0.82/avatar/month**. At 100 backers it's **~\$2.50/avatar/month**.

### Kickstarter budget at a glance

Recommended raise: **\$8 000** for a 12-month runway covering Phase 1 (free)
+ early Phase 2 (≤ 100 avatars) + LangSmith Plus + Cursor Pro + OpenAI burn
+ legal/design + ~30 % contingency. See `05_kickstarter_budget_lineitems.md`
for the spreadsheet rows.

---

## Code-side prerequisites (must be done before either phase ships)

Per the workspace `.cursorrules`, anything that becomes an env var has to
be added uppercase to `.env` / `.env.dev`, mirrored lowercase as a
`field(default=…, metadata={"description": …})` on `GlobalContext` in
`src/anubis/utils/context.py`, and read at the top of the function before
being passed through nested calls.

These are the variables this research recommends introducing (not added —
this is research-only):

```
EMBEDDING_PROVIDER=                # "huggingface" | "openai"
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMS=1536
ENABLE_SENTIMENT_CLASSIFIER=false
AVATAR_RUNTIME_MODE=lite           # "lite" (Free Tier) | "full" (Phase 2)
LANGSMITH_DEPLOYMENT_MODE=cloud    # "cloud" | "hybrid" | "self_hosted"
TOKEN_BUDGET_PER_AVATAR_MONTH=     # int; for the rate-limit guard
```

Wiring these up unlocks both the memory diet (§2.3) and the per-avatar
budget guards (§4.7).

---

## Outstanding follow-ups

1. **Confirm AWS account Track** — is it pre- or post-July-2025? This
   changes Phase 1 from "12-month free" to "6-month \$200-credit."
2. **Re-read the actual Kickstarter sheet** — the link
   ([sheet](https://docs.google.com/spreadsheets/u/0/d/1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA/htmlview?pli=1#gid=0))
   requires Google sign-in. The line items in `05_kickstarter_budget_lineitems.md`
   are structured to drop in but should be reconciled against any
   already-committed labels.
3. **Decide LangSmith Cloud vs Hybrid** for Phase 2 — Cloud is cheaper to
   operate; Hybrid keeps avatar data in our Azure subscription. The
   research recommends **starting Cloud** and converting to Hybrid only
   if a backer specifically requires it.
4. **Validate the \$60 Cursor figure** — Cursor Pro is \$20/mo; the \$60
   number maps to either Cursor Teams (\$40) + Pro (\$20), or three Pro
   seats. The budget sheet should clarify which.
5. **Resolve the Azure Cache for Redis retirement (Sep 30, 2028)** — fine
   for the Kickstarter year, but the Phase 2 plan should document the
   migration path to Azure Managed Redis or in-cluster `redis:7-alpine`.

---

## Source links used

- [AWS Free Plan / \$200 credits announcement (Jul 2025)](https://aws.amazon.com/about-aws/whats-new/2025/07/aws-free-tier-credits-month-free-plan)
- [AWS Free Tier plan choice docs](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html)
- [EC2 Free Tier usage tracking](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html)
- [Aurora PostgreSQL added to Free Tier (Mar 2026)](https://aws.amazon.com/about-aws/whats-new/2026/03/amazon-aurora-postgresql-aws-free-tier/)
- [Azure Postgres Flexible Server pricing](https://azure.microsoft.com/pricing/details/postgresql/flexible-server/)
- [Azure Cache for Redis pricing](https://azure.microsoft.com/en-us/pricing/details/cache)
- [Azure Kubernetes Service pricing](https://azure.microsoft.com/en-us/pricing/details/kubernetes-service)
- [AKS Free / Standard / Premium tiers](https://learn.microsoft.com/en-us/azure/aks/free-standard-pricing-tiers)
- [LangSmith Agent Server architecture](https://docs.langchain.com/langsmith/agent-server)
- [LangSmith control plane](https://docs.langchain.com/langsmith/control-plane)
- [LangSmith standalone server](https://docs.langchain.com/langsmith/deploy-standalone-server)
- [LangSmith Kubernetes self-host requirements](https://docs.langchain.com/langsmith/kubernetes)
- [LangSmith plans and pricing](https://www.langchain.com/langgraph-platform-pricing)
- [OpenAI API pricing (gpt-4o-mini, embeddings, transcription)](https://openai.com/api/pricing/)
- [Together AI pricing](https://together.ai/pricing/)
- [Cursor pricing](https://www.cursor.com/pricing)
- [microsoft/harrier-oss-v1-270m model card](https://huggingface.co/microsoft/harrier-oss-v1-270M)
- [j-hartmann/emotion-english-distilroberta-base model card](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
