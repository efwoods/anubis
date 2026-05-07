# 03 — Phase 2: Azure / Kubernetes Scale Plan with LangSmith

> Goal: replace the AWS Free Tier topology with a horizontally scalable,
> SLA-bearing deployment on **Azure Kubernetes Service (AKS)**, **Azure
> Database for PostgreSQL Flexible Server**, and **Azure Cache for Redis**,
> using **LangSmith Agent Server + Control Plane** as the deployment substrate.

This phase is what the Kickstarter funds. Phase 1 keeps backers' avatars
running while Phase 2 is built; Phase 2 absorbs them via DSN flip.

---

## 3.1 Why this stack maps cleanly to Anubis

The LangSmith Agent Server architecture
([docs.langchain.com/langsmith/agent-server](https://docs.langchain.com/langsmith/agent-server))
is structured exactly the way our graph already runs:

| LangSmith concept | Anubis component today |
| --- | --- |
| Graph | `src/anubis/graph.py:graph` (compiled `MessagesState` workflow) |
| Assistant | one row in our `assistant_state` keyed by `assistant_id` |
| Thread | LangGraph thread checkpoint (`thread_id` from `RunnableConfig`) |
| Run | one invocation of `/message` |
| Cron job | not used yet — future fit for memory consolidation |
| Postgres-backed persistence | already using `langgraph-checkpoint-postgres` and `BaseStore` over Postgres |
| Redis pubsub for streaming | already wired in `docker-compose-prod.yml` |
| Stateless API + queue worker containers | this is exactly `langgraph-api` |

Because we already speak the LangGraph runtime contract, deploying via
LangSmith is **almost zero-code-change**: it is mostly a rehoming of the
container + database + Redis to managed services and turning on the
control-plane "listener" inside our cluster.

The Control Plane
([docs.langchain.com/langsmith/control-plane](https://docs.langchain.com/langsmith/control-plane))
gives us, out of the box:

- A UI to create / update / version each avatar deployment.
- Per-deployment Postgres provisioned and managed automatically (in Cloud
  mode) — we'd skip this and bring our own Azure Postgres in Hybrid /
  Self-Hosted mode.
- LangSmith tracing project auto-created per deployment.
- Built-in CPU, memory, request latency, queue depth, and replica count
  metrics.
- Autoscaling between `Development` (1 CPU / 1 GB / 1 replica) and
  `Production` (2 CPU / 2 GB / up to 10 replicas).

---

## 3.2 Three deployment models — pick one

LangSmith offers three self-hosting flavors. We pick based on cost vs
control-plane convenience.

### Model A — LangSmith **Cloud** + Azure managed services for our DB/Redis

- LangSmith manages the control plane and even the agent-server
  containers; we just push our graph repo.
- Cheapest for ops; we still run our **own** Postgres + Redis if we want
  data sovereignty (Hybrid).
- Plus plan: \$39/seat/month, includes 1 free dev-sized agent deployment
  + 10 K traces/month + 500 Fleet runs/month.
- **Recommended starting point.** Move to Model B if we hit
  data-residency or pricing pain.

### Model B — LangSmith **Hybrid** (control plane in their cloud, data plane in our AKS)

- Best of both: deployment UI + autoscaling logic stay managed; our
  Postgres + Redis + container runtime stay in our Azure subscription.
- Listener runs in our cluster and polls the control plane.
- Same Plus / Enterprise plan structure.

### Model C — LangSmith **Self-Hosted Lite / Standalone Agent Server**

- Just the Agent Server runtime, no control plane UI.
- Minimum: **4 vCPU, 16 GB RAM, plus Postgres and Redis**.
- Cheapest infra footprint at "Lite" tier; manual deployment via Helm.
- Use if Kickstarter wants to claim "fully self-sovereign infra."

### Model D — LangSmith **Full Self-Hosted Control Plane + Data Plane**

- Required minimum cluster: **16 vCPU / 64 GB RAM**.
- Datastores (Postgres ≥ 8 GB / 2 vCPU, Redis ≥ 4 GB / 2 vCPU,
  ClickHouse ≥ 32 GB / 8 vCPU). Strongly recommended: managed externals.
- Enterprise license required.
- Estimated baseline cost: **\$1,150 – \$1,800 / month** in Azure infra
  before agent traffic.
- **Skip until annual Anubis revenue ≥ \$50K.**

> **Decision for the Kickstarter:** Model A (LangSmith Cloud / Plus) for the
> first 12 months, with our Postgres + Redis hosted in Azure to keep avatar
> data in our subscription. Move to Model B once we hit either ~5 deployed
> agents *or* a customer specifically requesting in-tenant deployment.

---

## 3.3 Azure cost model — line-item breakdown

All prices below are East US (May 2026 list pricing; pay-as-you-go;
unreserved). Reservations cut 30–60 %.

### 3.3.1 AKS — Azure Kubernetes Service

| Item | Spec | Price (PAYG) |
| --- | --- | --- |
| Control plane (Free tier — 1 cluster / region / sub) | up to ~10 nodes practical, no SLA | **\$0** |
| Standard tier control plane (with 99.95 % SLA) | required for prod | ~\$0.10/cluster-hour ≈ **\$73/mo** |
| Node pool — Standard_B2s (2 vCPU / 4 GB) | burstable, default for early prod | \$0.041/hr × 730 h ≈ **\$30/node/mo** |
| Node pool — Standard_D2s_v3 (2 vCPU / 8 GB) | balanced | \$0.096/hr ≈ **\$70/node/mo** |
| Node pool — Standard_D4s_v3 (4 vCPU / 16 GB) | for LangSmith heavy nodes | \$0.192/hr ≈ **\$140/node/mo** |
| Standard SSD persistent volumes | per agent | \$0.058/GB-month |
| Egress within Azure | free | \$0 |
| Egress to internet (after 100 GB free) | per GB | \$0.087/GB |
| Public Load Balancer (Standard) | 1 per cluster | ~\$18/mo |

**Recommended early prod cluster:**

- Free-tier control plane (no SLA; OK for Kickstarter rollout).
- 2 × `Standard_B2s` nodes for agent pods → **\$60/mo**.
- 1 × `Standard_B2ms` node (2 vCPU / 8 GB) for ingest workloads + LangSmith
  listener → **\$60/mo**.
- 100 GB Standard SSD → **\$5.80/mo**.
- 1 LB → **\$18/mo**.

**AKS subtotal: ~\$144/mo** (free control plane), ~\$217/mo (standard SLA).

### 3.3.2 Azure Database for PostgreSQL — Flexible Server

LangSmith Agent Server requires a single durable Postgres for assistants,
threads, runs, checkpoints, and store data.

| SKU | vCore / RAM | Compute | + Storage 64 GB | + Backup (≤ 100 % free) | Monthly |
| --- | --- | --- | --- | --- | --- |
| **B1ms** burstable | 1 / 2 GiB | \$12.41 | \$7.36 | \$0 | **~\$20** |
| **B2s** burstable | 2 / 4 GiB | ~\$33 | \$7.36 | \$0 | **~\$40** |
| **D2ds_v5** (general purpose) | 2 / 8 GiB | ~\$140 | \$11.50 (100 GB) | \$0 | **~\$152** |
| **D4ds_v5** | 4 / 16 GiB | ~\$280 | \$11.50 | \$0 | **~\$292** |

Free tier (Track A, new Azure customer): 750 hrs B1ms + 32 GB storage for
12 months. We will burn this first.

**Recommended early prod:** **B2s + 64 GB** → **~\$40/mo**, sufficient for
~50–100 active avatars (see §3.4).

Storage: \$0.115/GiB-month, with included IOPS up to 3 000 (sufficient for
our load). Backup is free up to 100 % of provisioned storage; geo-redundant
backup costs 2× extra.

### 3.3.3 Azure Cache for Redis

LangSmith uses Redis only for ephemeral pubsub/queue signaling — no user
data persists there.

> **Important:** Microsoft has announced retirement of Azure Cache for Redis
> (the original C0/C1 SKUs) on **September 30, 2028**, with Enterprise Redis
> retiring **March 30, 2027**. We should plan to migrate to Azure Managed
> Redis or self-host `redis:7-alpine` in AKS before then.

| Tier | RAM | Monthly |
| --- | --- | --- |
| Basic C0 | 250 MB | **\$16.06** (no SLA, no replication — dev only) |
| Basic C1 | 1 GB | **~\$70** (dev only) |
| Standard C1 | 1 GB | **~\$70** (HA, primary/replica) |
| Standard C2 | 6 GB | **~\$135** |

LangSmith standalone agent server only needs ~256 MB Redis for our scale;
the prod recommendation in Microsoft's own docs is **Standard tier C1 = \$70/mo**
to get a 99.9 % SLA. Self-hosting `redis:7-alpine` in the AKS cluster
(~30 MB RAM, free) is also valid for early prod and saves the \$70.

### 3.3.4 Other Azure line items

| Item | Use | Cost |
| --- | --- | --- |
| Azure Container Registry (Basic) | host `anubis-agent` and `anubis-ingest` images | ~\$5/mo |
| Azure Storage (Blob, hot tier) | replaces S3; user uploads + reference media | \$0.0184/GB-mo + transactions; for 100 avatars × 500 MB ≈ 50 GB ≈ \$1/mo |
| Application Gateway / Front Door | TLS termination, WAF | \$25–\$50/mo if used; AKS LB alone is fine to start |
| Log Analytics workspace | container logs + AKS diagnostics | \$2.30/GB ingested; budget \$15/mo at start |
| Azure Monitor for containers | included with AKS Standard tier | \$0 incremental |

### 3.3.5 LangSmith Cloud (SaaS) line items

| Item | Cost |
| --- | --- |
| Plus plan, 1 seat | \$39/mo |
| Additional seats | \$39/seat/mo |
| Included | 10 000 traces / mo, 500 Fleet runs / mo, 1 free dev-sized agent deployment, unlimited Fleet agents |
| Trace overage | \$0.50 per 1 000 traces |
| Fleet run overage | \$0.05 per run |
| Additional dev deployments | \$0.005/deployment-run + \$0.0007/min uptime |

The user's stated **\$40/mo for LangChain** maps to a single LangSmith Plus
seat (\$39 rounded). Once the team grows, plan on \$39 × seats.

---

## 3.4 Sizing the cluster against the workload

Workload assumptions (validated in `04_token_workload_cost_model.md`):

- 30-day window, ~30 conversations/avatar/month, 10 messages/conversation
  → 300 messages/avatar/month → ~10 messages/avatar/day.
- Each message triggers `think` + `respond` (= 2 model passes) + occasional
  tool call → call this ~3 LLM-pass-equivalents.
- p95 wall time per `/message`: 5–8 s (network + 2 model passes).
- Assume backers spread their use evenly: ~10 messages × 1/86400 s
  ≈ 1.16e-4 messages/sec/avatar continuous load.
- For **100 active avatars:** ~0.012 messages/sec, but **bursty**: peak
  prime-time concurrency ~5–10× average.
- Sustainable concurrency target: ≥ 10 simultaneous `/message` runs at
  100 avatars; ≥ 80 simultaneous at 1 000 avatars.

LangSmith default: each queue worker handles `N_JOBS_PER_WORKER=10`
runs concurrently. So one 2-CPU 4-GB pod easily covers 100 avatars.

| Avatar count | Pods | Postgres SKU | Redis SKU | AKS nodes (B-series) | Indicative Azure / mo |
| --- | --- | --- | --- | --- | --- |
| 1 – 50 | 1 × Dev (1 CPU / 1 GB) | B1ms | C0 or in-cluster | 2 × B2s | **\$120** |
| 51 – 200 | 2 × Prod (2 CPU / 2 GB) | B2s | C1 Standard | 3 × B2s | **\$220** |
| 201 – 1 000 | 3–5 × Prod (autoscale to 10) | D2ds_v5 | C1 Standard | 3–5 × D2s_v3 | **\$520** |
| 1 001 – 5 000 | 6–10 × Prod | D4ds_v5 + read replica | C2 Standard | 5 × D4s_v3 | **\$1 600** |

These are infrastructure-only numbers; LLM costs are
in [`04_token_workload_cost_model.md`](./04_token_workload_cost_model.md).

---

## 3.5 Migration plan — Phase 1 → Phase 2

1. Export all `assistant_state` rows + memory store rows from the AWS RDS
   db.t3.micro using `pg_dump`.
2. Provision the Azure Postgres B2s + 64 GB; restore.
3. Provision the AKS cluster (free control plane + 2 B2s nodes, in-cluster
   Redis to start).
4. Push `anubis-agent:phase2` image to Azure Container Registry.
5. Deploy via LangSmith Cloud Plus:
   - Connect the GitHub repo as a control-plane Integration.
   - Create a `Production` deployment per avatar **OR** a single
     multi-tenant deployment that selects assistant by `assistant_id` (we
     already do the latter — see `extract_user_id_assistant_id`).
   - Bind it to the Azure Postgres + Redis URIs.
6. Flip DNS: the avatar VMs in AWS forward `/message/...` requests to the
   AKS ingress for a 7-day soak, then we shut the AWS VMs down.
7. Drop the AWS RDS instance after a final snapshot.

The migration is data-driven and can be done one avatar at a time — useful
for staged Kickstarter backer onboarding.

---

## 3.6 What we still owe ourselves before Phase 2 ships

- A "lite" Docker image (per §2.3) so the same artifact runs on Free Tier
  EC2 *or* on AKS.
- An ingest job runner (Azure Container Apps Jobs is the cheapest fit:
  pay-per-second on demand, no always-on cost).
- Schema migrations runnable via `alembic` — currently we depend on the
  bundled `langgraph-checkpoint-postgres` migrations only.
- A `terraform/azure/` module mirroring what LangChain ships at
  `docs.langchain.com/langsmith/kubernetes` (they provide a reference
  Terraform module for AKS + AzureRM Postgres + Redis).
