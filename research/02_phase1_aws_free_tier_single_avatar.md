# 02 — Phase 1: Single-Avatar VMs on AWS Free Tier

> Goal: deploy **N independent avatar containers** — one Anubis graph per
> Kickstarter-backer avatar — on **free** AWS infrastructure, with each VM
> staying inside the 1 GB RAM ceiling that the new AWS Free Plan permits.

This phase is the runway plan. It exists to keep **per-avatar marginal hosting
cost ≈ \$0** while we collect Kickstarter funds and validate retention.
Phase 2 (Azure / Kubernetes / LangSmith Control Plane) is funded by
Kickstarter and starts in parallel — see
[`03_phase2_azure_kubernetes_scale_plan.md`](./03_phase2_azure_kubernetes_scale_plan.md).

---

## 2.1 What the AWS Free Tier actually gives us in 2026

AWS replaced the legacy 12-month tier on **July 15, 2025**. There are now two
distinct tracks — confirm which one applies to your account.

### Track A — Accounts created **before** July 15, 2025 (legacy "12-month free tier")

Effectively unlimited duration so long as you stay under each service's
monthly free allotment. Relevant items:

| Service | Free Allotment (per month, 12 months from signup) | Notes |
| --- | --- | --- |
| EC2 `t2.micro` **or** `t3.micro` | 750 hours / month | 750 hours = 1 instance running 24×7. Either family, not both simultaneously. |
| EBS gp2/gp3/standard/st1/sc1 | 30 GB total | Per account, *not* per instance. |
| EBS snapshots | 1 GB | |
| EBS I/O | 2,000,000 I/Os | |
| S3 Standard | 5 GB | + 20,000 GET, 2,000 PUT, 1 GB egress. |
| Public IPv4 egress | 100 GB / month | All-services aggregate. |
| Lambda | 1 M requests + 400 K GB-sec | Always-free, not 12-month. |
| CloudWatch | 10 custom metrics, 5 GB logs | Always-free. |
| RDS db.t3/t4g.micro | 750 hours, 20 GB SSD | 12-month, useful as a managed alternative. |
| Aurora PostgreSQL Serverless | Free tier (added Mar 2026) | Capped capacity; verify quota in your region. |

### Track B — Accounts created **on or after** July 15, 2025 (new "Free Plan")

- **\$100 in credits** at signup, **+\$100 more** earned by activating key
  services → up to **\$200 total**.
- **6-month** Free Plan duration **OR** until credits are exhausted, whichever
  comes first.
- Eligible EC2 families expanded to: `t3.micro`, `t3.small`, `t4g.micro`,
  `t4g.small`, `c7i-flex.large`, `m7i-flex.large`. The credits cover whichever
  you choose.
- 30+ "always-free" services remain free outside this window.
- Unused credits remain applicable up to 12 months from signup.

> **Recommendation:** if the project's AWS account is post-July-2025, plan on
> burning the \$200 credit on `t3.small` (2 GB RAM) so we get headroom while
> we trim the image, then either roll over to a paid `t3.micro` or migrate the
> avatar VMs to Azure once Phase 2 is live.

### Storage ceiling for "data" on Free Tier

Combining the relevant lines:

| Bucket | Free | Practical use |
| --- | --- | --- |
| EBS attached to EC2 | **30 GB total per account** | OS (~5 GB), Docker layers (~3 GB), HF cache (~1.5 GB), per-VM working set (~5 GB). On a single VM there's plenty of room; across N VMs the 30 GB cap binds at roughly **5–8 micro VMs**. |
| S3 Standard | **5 GB** | Avatar reference media, generated audio clips, profile artifacts. With per-avatar media at ~300 MB this saturates at ~16 avatars. |
| Postgres (RDS db.t3.micro) | **20 GB SSD** | At ~250–400 MB per fully-built avatar (§1.3) → **50–80 avatars** of vector + memory data fit on a single shared Postgres free instance. |
| Aurora Postgres Serverless | New 2026 free tier (capped) | Could replace RDS once we confirm quota; supports multi-AZ. |

**Implication:** the binding free-tier constraint is **EBS (30 GB) and S3
(5 GB)**, *not* compute hours. A topology that puts the avatar containers on
EC2 but offloads all bulk data to S3 + RDS is the right shape.

---

## 2.2 Architectural choice: in-VM DB vs external DB

The user's question — "run the graph in the container *or* use the container
and connect to an external database" — is the central architectural call.
Both are viable on Free Tier; the trade-offs are clear.

### Option A — "Sealed avatar": Postgres + Redis inside the VM

```
+---------------------- t3.micro 1 GB ---------------------+
|  langgraph-api (Anubis)   ~700 MB                        |
|  Redis (alpine, no replication)                ~30 MB    |
|  Postgres + pgvector                           ~250 MB   |
|  /var/lib/postgresql/data on EBS gp3 (~5–10 GB / VM)     |
+----------------------------------------------------------+
```

Pros:
- One VM = one avatar = one zero-coordination unit. Trivially demoable.
- Zero outbound egress to a DB; latency is sub-millisecond.
- Fits perfectly into Track-B \$100 credit math: each avatar consumes only
  its own VM hours, no shared bill.

Cons / hard problems:
- **It does not fit in 1 GB RAM** with the embedding + sentiment model
  in-process. We must either (a) move embedding out of process, or (b) move
  to `t3.small` / `t4g.small` (2 GB) and pay ≈ \$15/mo per VM after credits.
- 30 GB EBS account-wide cap caps us at roughly **5 sealed VMs** before
  paying for storage.
- Backups, schema migrations, and embedding-index rebuilds are now
  N-fold operational work.
- pgvector queries compete with the model for CPU on a 1 vCPU box.

### Option B — "Shared brain": graph in VM, DB external (RECOMMENDED)

```
+-------- t3.micro 1 GB (per avatar) --------+        +-------- RDS db.t3.micro --------+
|  langgraph-api (Anubis)                    | -----> |  postgres + pgvector             |
|  thin Redis (alpine) for queue/pubsub      |        |  schema-per-avatar OR row-level  |
|  no DB on local disk                       |        |  isolation by assistant_id        |
+--------------------------------------------+        +----------------------------------+
                                                              ^
                                                              | (or one shared
                                                              |  RDS Aurora Serverless
                                                              |  on the new free tier)
+-------- S3 bucket: anubis-avatars ---------+
|  /{user_id}/{avatar_id}/media/...          |
|  /{user_id}/{avatar_id}/profiles/...       |
+--------------------------------------------+
```

Pros:
- Fits cleanly in Free Tier: 1 RDS db.t3.micro (20 GB) handles **50–80
  avatars' worth of vectors + memory** before storage becomes the constraint.
- Schema migrations and backups become a single-server problem.
- Avatar VMs become *stateless*; we can recycle them or migrate to
  Azure (Phase 2) without data movement — only DSN flip.
- Aligns with the LangSmith Agent Server reference architecture
  (`docs.langchain.com/langsmith/agent-server`), which assumes Postgres +
  Redis are shared, durable services and treats containers as stateless.

Cons:
- Adds 5–30 ms RTT per `BaseStore` / checkpoint write.
- All avatars share a 1 vCPU / 1 GB Postgres — once we exceed ~50 active
  conversations *concurrently*, this becomes the bottleneck and we need to
  scale the DB up (paid) or split tenants.
- Shared Redis means any avatar's pubsub traffic competes for the 1 free
  ElastiCache micro (or we run `redis:7-alpine` on a tiny EC2 — which uses
  the 750 hours).

### Recommendation

**Option B** for production demo. Use **Option A** only for the "kiosk
demo VM" (e.g. event/booth use case where we hand someone a single self-
contained image). The same Docker image supports both modes — the only
difference is whether `VECTORSTORE_POSTGRES_URI` and `ASYNC_POSTGRES_STORE_URI`
point at `localhost` or at the shared RDS endpoint.

---

## 2.3 Memory diet to actually fit on 1 GB

These are concrete, ranked changes. Each is independent; they compose.

### Diet step 1 — Pull the embedding model out of the Python process (mandatory)

The 270 M-param `microsoft/harrier-oss-v1-270m` is the single largest RAM
consumer (~540 MB even at FP16). Two viable options:

**1a. Swap to a managed embedding API** (cheapest on RAM, costs cents):
- `text-embedding-3-small` (OpenAI): **\$0.02 / 1 M tokens**, 1536 dim.
- Together AI hosts smaller embedders too. Cost analysis is in
  [`04_token_workload_cost_model.md`](./04_token_workload_cost_model.md).
- **Tradeoff:** vector dimension changes from 640 → 1536. We'd need to
  re-embed existing data and update `LANGGRAPH_STORE` config and the
  `dims` column in pgvector.

**1b. Run the embedder in a sidecar container** on the *same* VM (or on a
single shared "embedder" VM):
- HuggingFace Text Embeddings Inference (TEI) or `sentence-transformers`
  served via a thin FastAPI process. Memory still consumed but isolated
  from the agent process; we can OOM-restart it without taking down the
  agent.
- On Free Tier this only helps if we move the embedder to a *separate*
  shared VM (one embedder serving many avatar VMs). With 750 free hours
  this is one always-on micro for the embedder + N micros for the agents,
  but they share the 750-hour budget — so realistically we can afford
  exactly **one** continuously-running VM under Track A.

**1c. Switch to a smaller local model** — e.g. `all-MiniLM-L6-v2`
(80 MB, 384 dim). Significant retrieval-quality regression vs harrier-oss
(MTEB v2 of 66.5 vs ~58 for MiniLM), but trivial to deploy.

**Recommendation:** 1a (OpenAI `text-embedding-3-small`) for production
single-avatar VMs. Keep harrier-oss as the local default for our laptops /
training corpus pipeline (`build_profile.py`, `build_knowledge_profile.py`).

Add an env var to `context.py` (per `.cursorrules`):

```
EMBEDDING_PROVIDER=    # "huggingface" | "openai"
EMBEDDING_MODEL=       # e.g. "text-embedding-3-small"
EMBEDDING_DIMS=        # e.g. 1536
```

…and route `LANGGRAPH_STORE` and `vector_store_graph` accordingly.

### Diet step 2 — Cache the sentiment classifier (mandatory)

In `src/anubis/graph.py:respond()`:

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
sentiment = classifier(avatar_response.content)
```

This re-loads ~328 MB on every response. Two fixes:

- **Module-level singleton** (lazy + cached) — saves repeated loads but the
  RAM still sits at ~328 MB resident.
- **Drop the local model entirely** and call OpenAI / Together for
  classification, or remove the feature for the Free Tier build (the
  emotion is a "nice to have" annotation in `response_metadata`).

**Recommendation:** make sentiment opt-in via env var
`ENABLE_SENTIMENT_CLASSIFIER=false`. The Free Tier image disables it; the
Azure Phase-2 image enables it (running on a beefier node).

### Diet step 3 — Move heavy media processing off the avatar VM

`yt_dlp`, `moviepy`, `bert_score`, `spacy` are pulled in for ingest pipelines
(`process_media_graph`, `build_profile.py`). They contribute ~200 MB import-
time RAM but are only needed when the user uploads new media.

Split the image into two:

- **`anubis-agent`** — runtime path only. No `moviepy`, `yt_dlp`, `bert_score`,
  `spacy`. ~600 MB resident.
- **`anubis-ingest`** — invoked on-demand (Lambda, ECS task, or Azure
  Container Instance) to run the upload pipeline. Spawn → process → die.

`pyproject.toml` already separates `[project.optional-dependencies] dev`. Add
two more groups: `agent` and `ingest`, and use them as build-time `--extra`
selectors.

### Diet step 4 — Postgres tuning if we keep Option A

If we ship "sealed" kiosk VMs, set:

```
shared_buffers = 64MB
work_mem = 4MB
maintenance_work_mem = 32MB
effective_cache_size = 256MB
max_connections = 25
```

This drops Postgres steady-state from ~250 MB to ~120 MB, leaving more for
Python.

### Resulting memory budget on a t3.micro (1 GB)

| Component | Stripped image RAM | Notes |
| --- | --- | --- |
| OS + Docker + ssh | 100 MB | Amazon Linux 2023 |
| `anubis-agent` Python process | 350 MB | no transformers pipelines, embedding via API |
| Redis 7-alpine | 30 MB | |
| LangSmith tracer | 30 MB | |
| OS file cache headroom | 250 MB | |
| Spike headroom for tool calls | 240 MB | |
| **Total** | **~750 MB** | fits comfortably on 1 GB |

---

## 2.4 How many avatars actually fit per Free Tier account

Combining §2.1 + §2.3:

| Constraint | Hard ceiling |
| --- | --- |
| EC2 hours (Track A: 750/mo) | **1** continuously-running avatar VM. To run N avatars cheaply on a single account, multi-tenant them on one VM. |
| EC2 hours (Track B: \$100 credit) | t3.micro is ~\$0.0104/hr → 750 hrs = \$7.80. \$100 covers ~12,800 hrs ≈ **17 t3.micros** simultaneously for 1 month, or **1 VM for ~13 months**. |
| EBS 30 GB | ~5–8 sealed VMs at 5 GB OS+work each, or 1 ingest VM + many small agents. |
| S3 5 GB | ~16 avatars at 300 MB media each. |
| RDS db.t3.micro 20 GB | **50–80 avatars** of vector + memory rows. |

**The cleanest topology that maximises avatars-per-account:**

```
1 × t3.small VM (Track B) running N agent containers on the same Docker host
  ├── docker compose: anubis-agent x N (one per avatar, ~600 MB each)
  ├── docker compose: redis (shared)
  └── outbound to:
        ├── 1 × RDS db.t3.micro Postgres+pgvector (shared, 20 GB)
        ├── S3: anubis-avatars (5 GB)
        └── OpenAI / Together / LangSmith
```

This gives us **~5 concurrent avatars on a single 2 GB VM** under the new
Free Plan, all sharing one Postgres and one S3 bucket, and zero state on the
avatar containers.

**For the Kickstarter** — backers paying for "their own avatar" map cleanly
to "one container in this fleet." When demand exceeds ~5 simultaneous active
conversations, we either (a) launch a second t3.small (covered by remaining
\$100 credit), or (b) cut over the entire fleet into the Azure Phase-2
cluster — DSN flip only.

---

## 2.5 Concrete Phase-1 deployment recipe

1. **Account setup**
   - Confirm whether the AWS account is pre- or post-July-2025.
   - Activate budget alarms at \$1, \$10, \$25.
   - Tag every resource with `Project=anubis, Phase=1`.

2. **Image split**
   - `Dockerfile.anubis.agent` (no moviepy/yt_dlp/bert_score/spacy/streamlit)
   - `Dockerfile.anubis.ingest` (full set; only run on demand)
   - Both inherit from the existing `anubis-base` to keep base layer cached.

3. **Env-var additions** (per `.cursorrules`, also wire into `context.py`):
   ```
   EMBEDDING_PROVIDER=openai
   EMBEDDING_MODEL=text-embedding-3-small
   EMBEDDING_DIMS=1536
   ENABLE_SENTIMENT_CLASSIFIER=false
   AVATAR_RUNTIME_MODE=lite          # lite | full
   ```

4. **Provision once per AWS account**
   - 1 × VPC default
   - 1 × t3.small (Track B) or t3.micro (Track A) — `anubis-fleet-host-1`
   - 1 × RDS db.t3.micro Postgres 16 + pgvector — `anubis-shared-db`
     (alternatively try the new Aurora PostgreSQL Serverless free tier)
   - 1 × S3 bucket `anubis-avatars-prod` with public read disabled,
     CloudFront in front for the avatar reference images
   - 1 × Route 53 hosted zone (~\$0.50/mo, not free) — only if needed
   - Security groups: VM → RDS:5432, VM → 0.0.0.0:443

5. **Per-avatar provisioning step (automated)**
   - `assistant_id` is created in Postgres
   - Container started with `-e ASSISTANT_ID=… -e USER_ID=…`
   - S3 prefix created: `s3://anubis-avatars-prod/{user_id}/{assistant_id}/`
   - LangSmith project name set to the deployment name (matches the
     pattern Control Plane uses for Phase 2)

6. **Observability** — keep the existing Prometheus + Grafana stack but
   move Grafana off the agent host (CloudWatch can do it for free if budget
   is tight).

7. **Disaster cap**
   - Configure RDS automated daily snapshots (free up to 100 % of provisioned
     storage = 20 GB).
   - S3 versioning **off** (saves storage); rely on Postgres for source of
     truth.

---

## 2.6 What this phase *does not* solve

- It is single-tenant from a billing standpoint — one avatar's runaway
  token usage hits one shared OpenAI key.
- There is no fleet-wide control plane: deploying a new revision means SSH
  + `docker compose pull && up`. (LangSmith Control Plane in Phase 2 fixes
  this.)
- No SLA. AWS Free Plan VMs in Track B run on standard infrastructure, but
  RDS db.t3.micro is single-AZ and not HA.

These are acceptable for the Kickstarter demo + early-backer rollout. The
moment we have paying revenue or a single backer pays for "premium hosting,"
we cut over to Phase 2.
