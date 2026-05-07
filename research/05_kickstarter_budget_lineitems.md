# 05 — Kickstarter Budget Line-Items (12-Month Runway)

> Goal: produce concrete, defensible rows that drop into the Kickstarter
> budget spreadsheet at
> [docs.google.com/spreadsheets/.../1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA](https://docs.google.com/spreadsheets/u/0/d/1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA/htmlview?pli=1#gid=0).
>
> The shared sheet requires Google sign-in (we couldn't read it directly).
> The structure below is opinionated — copy the rows you want and rename
> categories to match the existing sheet's headers.

The math here uses the numbers established in:

- [`02_phase1_aws_free_tier_single_avatar.md`](./02_phase1_aws_free_tier_single_avatar.md) — Phase 1 hosting
- [`03_phase2_azure_kubernetes_scale_plan.md`](./03_phase2_azure_kubernetes_scale_plan.md) — Phase 2 hosting
- [`04_token_workload_cost_model.md`](./04_token_workload_cost_model.md) — token / model spend

---

## 5.1 Recurring SaaS / dev tooling (always-on, every month)

| Line item | Vendor | Plan | Monthly | 12-month |
| --- | --- | --- | --- | --- |
| LangChain / LangSmith Plus | LangChain Inc. | Plus, 1 seat (10 K traces, 500 Fleet runs, 1 dev deployment) | **\$40** | **\$480** |
| Cursor Pro (founder seat) | Cursor | Pro | \$20 | \$240 |
| Cursor Teams (collaborator seat) | Cursor | Teams | \$40 | \$480 |
| **Cursor sub-total (matches stated \$60 budget)** | | | **\$60** | **\$720** |
| OpenAI API (model + STT + embeddings, 100 avatars) | OpenAI | PAYG | **\$30** | \$360 |
| (alt at 1 000 avatars) | OpenAI | PAYG | \$300 | \$3 600 |
| Domain + DNS | Cloudflare / Route 53 | Standard | \$2 | \$24 |
| Stripe / payment fees (≈ 2.9 % + \$0.30 per pledge) | Stripe | PAYG | varies | varies |
| GitHub Pro / Actions minutes | GitHub | Pro + paid Actions | \$10 | \$120 |
| Sentry or equivalent error tracking (optional) | Sentry | Team | \$26 | \$312 |
| **Recurring SaaS subtotal (Phase-1 traffic)** | | | **\$168** | **\$2 016** |

---

## 5.2 Phase 1 hosting — AWS Free Tier (months 1–6)

| Line item | Provider | Spec | Monthly | Cumulative (6 mo) |
| --- | --- | --- | --- | --- |
| EC2 fleet host (t3.small Track-B) | AWS | 2 vCPU / 2 GB | \$0 (covered by \$200 credit) | \$0 |
| RDS db.t3.micro Postgres + pgvector | AWS | 1 vCPU / 1 GB / 20 GB | \$0 (free 12 mo) | \$0 |
| S3 Standard (5 GB) | AWS | 5 GB + 20 K GET + 1 GB egress | \$0 | \$0 |
| Public IPv4 hours (post-2024 charge) | AWS | 1 EIP × 730 h | \$3.60 | \$22 |
| CloudWatch metrics & logs (above free) | AWS | 5 GB logs + 10 metrics | \$0 | \$0 |
| **Phase-1 hosting subtotal** | | | **\$3.60** | **\$22** |

> Realistic worst-case: even if we blow past free quotas in month 4, the
> \$200 credit absorbs another ~\$30/mo for two months before any out-of-
> pocket cost.

---

## 5.3 Phase 2 hosting — Azure Production (months 4–12 onward)

We assume Phase 2 stands up in **month 4** in parallel with Phase 1 still
serving backers. Months 4–6 run *both* environments at once (negligible
since Phase 1 is free); month 7 onwards is Azure-only.

### 5.3.1 Early production (≤ 100 avatars) — months 4–9

| Line item | Provider | Spec | Monthly |
| --- | --- | --- | --- |
| AKS control plane (Free tier, 1 cluster) | Azure | no SLA | \$0 |
| AKS node pool — 2 × Standard_B2s | Azure | 2 vCPU / 4 GB each | \$60 |
| AKS node pool — 1 × Standard_B2ms (ingest + listener) | Azure | 2 vCPU / 8 GB | \$60 |
| Azure Database for PostgreSQL — B2s + 64 GB | Azure | 2 vCore / 4 GiB | \$40 |
| Azure Cache for Redis — Standard C1 | Azure | 1 GB HA | \$70 |
| (alt) in-cluster `redis:7-alpine` | self-hosted | shared pod | \$0 |
| Standard SSD persistent volumes (100 GB) | Azure | 100 GiB | \$5.80 |
| Standard Load Balancer | Azure | 1 LB | \$18 |
| Azure Container Registry (Basic) | Azure | 10 GB included | \$5 |
| Azure Blob Storage (50 GB hot) | Azure | replaces S3 | \$1 |
| Log Analytics workspace (15 GB ingested) | Azure | per-GB | \$15 |
| Egress to internet (after 100 GB free) | Azure | est. 50 GB / mo | \$0 |
| **Subtotal (managed Redis path)** | | | **\$214.80** |
| **Subtotal (in-cluster Redis path)** | | | **\$144.80** |

### 5.3.2 Scale (≤ 1 000 avatars) — months 9–12

| Item | Spec | Monthly |
| --- | --- | --- |
| AKS Standard tier control plane (99.95 % SLA) | | \$73 |
| AKS node pool — 3 × Standard_D2s_v3 | 2 vCPU / 8 GB each | \$210 |
| Postgres — D2ds_v5 + 100 GB | 2 vCore / 8 GiB | \$152 |
| Redis Standard C1 (1 GB HA) | | \$70 |
| ACR Standard | geo-replication | \$20 |
| LB + egress + log analytics | | \$60 |
| **Phase-2 scale subtotal** | | **\$585** |

### 5.3.3 LLM API spend at the same scale points

(From `04_token_workload_cost_model.md`)

| Avatars | OpenAI inference (gpt-4o-mini) | Embeddings | LangSmith trace overage | **LLM monthly** |
| --- | --- | --- | --- | --- |
| 100 | \$27 | \$1 | \$0 | **\$28** |
| 1 000 | \$270 | \$6 | \$25 | **\$301** |
| 10 000 | \$2 700 | \$60 | \$295 | **\$3 055** |

---

## 5.4 One-time setup line-items

| Line item | Cost | Purpose |
| --- | --- | --- |
| Initial avatar onboarding STT (per backer × 1h source) | \$2 | First-time transcription / diarization |
| Initial avatar embedding pass | \$0.15 | One-time `text-embedding-3-small` ingest |
| Apple / Google developer accounts (if mobile launches) | \$99 + \$25 | One-time |
| Trademark search / filing | \$250 – \$1 500 | One-time |
| Legal review of Terms / Privacy | \$500 – \$2 000 | One-time |
| Designer (logo + landing) | \$500 – \$3 000 | One-time |

---

## 5.5 Runway scenarios — what does \$X of Kickstarter funding buy?

Assumes founder draws \$0 salary (or that's a separate line). Numbers below
are pure infra + tooling + LLM at three scale points.

### Scenario A — \$2 500 raise (low-end, "MVP demo only")

```
Phase 1 hosting          : 6 × $3.60   =     $22
SaaS (LangSmith + Cursor): 12 × $100    =  $1 200
OpenAI @ 100 avatars     :  8 × $30     =    $240
Domain / GitHub / misc   :              =    $200
One-time legal + design  :              =    $500
                                       --------
                                          $2 162  (16 % buffer)
```

### Scenario B — \$8 000 raise (recommended target, "real Phase 2 launch")

```
Phase 1 hosting          : 4 × $3.60    =     $14
Phase 2 hosting (early)  : 8 × $215     =  $1 720
SaaS (LangSmith + Cursor): 12 × $100    =  $1 200
OpenAI @ 100 avatars     : 12 × $30     =    $360
LangSmith trace overage  :              =     $50
One-time legal + design  :              =  $1 500
Domain / GitHub / Sentry / misc        =    $500
Reserve / contingency    :              =  $2 656
                                       --------
                                          $8 000
```

### Scenario C — \$25 000 raise ("scale to 1 000 avatars and keep them happy")

```
Phase 1 hosting          : 4 × $3.60    =     $14
Phase 2 hosting (scale)  : 9 × $585     =  $5 265
SaaS (LangSmith + Cursor): 12 × $100    =  $1 200
OpenAI @ 1 000 avatars   : 12 × $300    =  $3 600
LangSmith trace overage  : 12 × $25     =    $300
One-time legal + design  :              =  $3 000
Domain / GitHub / Sentry / misc        =  $1 500
Founder honoraria / contractor budget  = $5 000
Reserve / contingency    :              =  $5 121
                                       --------
                                         $25 000
```

---

## 5.6 Direct mapping to the Kickstarter sheet

Suggested category labels (rename to match what the sheet already uses):

1. **Software & Tools** — LangChain (\$40 × 12), Cursor (\$60 × 12), GitHub
   (\$10 × 12), Sentry (optional).
2. **Cloud Infrastructure (AWS Phase 1)** — \$22 (6 mo).
3. **Cloud Infrastructure (Azure Phase 2)** — \$214 × N (early) or \$585 × N
   (scale).
4. **AI Model Usage (OpenAI)** — \$30 / \$300 / \$3 055 per month depending
   on tier.
5. **Per-Avatar Onboarding** — \$2.15 × backer count (one-time STT + embed).
6. **Legal / Trademark / Compliance** — line at \$500 – \$2 000.
7. **Design & Marketing** — \$500 – \$3 000.
8. **Reserve / Contingency** — at least 15 % of total.
9. **Stripe / Kickstarter platform fees** — Kickstarter takes 5 % + Stripe
   ~3–5 %; subtract this from the gross raise before allocating to lines
   above.

---

## 5.7 Suggested copy for the Kickstarter funding-use section

> Funds raised pay for the infrastructure that keeps each backer's avatar
> alive: hosting on AWS during the early-access window (covered by AWS Free
> Tier credits — your contribution funds nothing wasted on overhead),
> migration to a scalable Azure / Kubernetes cluster as the community grows
> (Postgres, Redis, and our AI agent server), and the OpenAI inference and
> embedding tokens that let your avatar actually think and remember. We use
> LangChain's LangSmith for tracing every conversation so we can debug
> issues for backers, and Cursor Pro to stay productive while shipping
> features.
>
> Concretely, every \$10 of pledge funds approximately one backer's avatar
> for one full year of typical use (one conversation/day, ten messages/
> conversation), including their share of hosting, model inference,
> embeddings, and observability. Heavier usage (multiple conversations/day)
> scales linearly with the same per-message cost.

