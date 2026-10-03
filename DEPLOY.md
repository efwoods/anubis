# Deploying Anubis to AWS EC2

Written 2026-10-02. Region: `us-east-1` (chosen; change the region and re-read the price lists if the deployment goes elsewhere).

Costs below exclude data transfer out, burstable-CPU surplus credit charges, and API CPU use under load. Those three values are **not measured**.

## 1. Measured resource use

Measured on the local host on 2026-10-02 against the running prod stack.

| Component | Measured value | Source |
|---|---|---|
| API container `anubis-langgraph-api-prod-1` | 1.668 GiB in use; 2.606 GiB peak (2,798,063,616 B, includes page cache) | `docker stats`; cgroup `memory.peak` |
| API container average CPU (idle, 1,318 s uptime) | 50.254 s ÷ 1,318 s = 0.0381 cores | cgroup `cpu.stat` `usage_usec` |
| Postgres 16 (`pgvector` 0.8.2, `ltree` 1.2, `btree_gin` 1.3) | 291.7 MiB RAM; database `postgres` = 24,390,204,439 B, shared by dev and prod | `docker stats`; `pg_database_size` |
| Redis | 1,286,440 B used; 1,344,408 B peak; 0 keys | `redis-cli info memory`; `redis-cli dbsize` |
| Grafana + Prometheus + `phone_worker` | 233.6 + 40.9 + 18.73 = 293.23 MiB | `docker stats` |
| API image `evdev3/anubis-langgraph-api:latest` | 14.6 GB | `docker images` |
| Hugging Face cache `~/.cache/huggingface` | 11,756,179,369 B | `du -sb` |
| NLTK cache `~/.cache/nltk_data` | 73,725,591 B | `du -sb` |

Idle peak RAM for the whole prod stack: 2.606 GiB + (291.7 + 233.6 + 40.9 + 18.73 + 4.492) MiB = 3.182 GiB.

**Not measured:**
- Peak RAM and CPU under request load. To measure, sample `docker stats --no-stream` every second while sending a batch of `/message` requests plus one audio upload, then read `memory.peak` and `cpu.stat` from `/sys/fs/cgroup/system.slice/docker-<container id>.scope/`.
- The prod-only share of the 24,390,204,439 B Postgres database.

## 2. AWS prices

Sources: AWS public price lists for `us-east-1`, on-demand, Linux, 730 hours per month.
- EC2 and EBS: `b0.p.awsstatic.com` meteredUnitMaps, published 2026-09-25.
- RDS: `AmazonRDS` offer, published 2026-10-01.
- ElastiCache: `AmazonElastiCache` offer, published 2026-09-14.
- Public IPv4: `AmazonVPC` offer.

| Item | Rate | Per month |
|---|---|---|
| EC2 t3.medium (2 vCPU, 4 GiB) | $0.0416/h | $30.368 |
| EC2 t3.large (2 vCPU, 8 GiB) | $0.0832/h | $60.736 |
| EC2 m7i.large (2 vCPU, 8 GiB, not burstable) | $0.1008/h | $73.584 |
| EC2 t3.xlarge (4 vCPU, 16 GiB) | $0.1664/h | $121.472 |
| EBS gp3 storage | $0.08/GB-month | — |
| Public IPv4 address | $0.005/h | $3.65 |
| RDS PostgreSQL db.t4g.small (2 vCPU, 2 GiB), Single-AZ | $0.032/h | $23.36 |
| RDS gp3 storage | $0.115/GB-month | — |
| RDS backup storage | $0.095/GB-month | — |
| ElastiCache cache.t4g.micro, Redis engine (0.5 GiB) | $0.016/h | $11.68 |
| ElastiCache cache.t4g.micro, Valkey engine (0.5 GiB) | $0.0128/h | $9.344 |

## 3. Deployment options and monthly cost

### Option A: everything on one EC2 instance (Docker Compose)

Disk needed (measured): 14.6 + 11.756 + 0.074 + 24.39 = 50.82 GB. Chosen EBS size: 60 GiB × $0.08 = $4.80.

| Instance | Arithmetic | Per month |
|---|---|---|
| t3.large | $60.736 + $4.80 + $3.65 | **$69.186** |
| t3.medium | $30.368 + $4.80 + $3.65 | **$38.818** |

The t3.medium leaves 4 − 3.182 = 0.818 GiB above the measured idle peak. Whether 0.818 GiB of headroom survives load is not measured.

### Option B: EC2 for the API, RDS for Postgres, ElastiCache for Redis

Chosen sizes: 30 GiB EC2 disk ($2.40), 30 GiB RDS storage (30 × $0.115 = $3.45).

$60.736 (t3.large) + $2.40 + $3.65 + $23.36 + $3.45 + $11.68 = **$105.276/month**

Verify before choosing Option B:
- Which `pgvector`, `ltree`, and `btree_gin` versions RDS PostgreSQL 16 offers.
- Whether `langgraph-api` works against the Valkey engine (Valkey is $2.336/month cheaper than Redis).

### Option C: Option B with Redis kept on the instance

Redis holds 1,344,408 B at peak, so a managed cache adds cost without a measured need.

$105.276 − $11.68 = **$93.596/month**

## 4. Automatic deploy on push to `main`

### Why not a self-hosted GitHub runner

`efwoods/anubis` is a **public** repository. A self-hosted runner on a public repository can be made to run code from outside contributors on the same machine that holds `.env`.

Instead, a GitHub-hosted job signs in to AWS through OIDC (no stored AWS keys) and uses AWS Systems Manager (SSM) Run Command to execute a deploy script on the instance. No SSH port is open and no image registry is needed. The image is built on the instance, the same way `dockerbuild.sh` builds the image today.

### One-time setup

1. **EC2 instance**
   - Launch the instance with an IAM instance profile that has `AmazonSSMManagedInstanceCore`.
   - Install Docker and the Docker Compose plugin.
   - `git clone https://github.com/efwoods/anubis.git /opt/anubis`
   - Copy `.env` to `/opt/anubis/.env`, and copy `~/.cache/huggingface` and `~/.cache/nltk_data` to the same paths on the instance.
   - Run `./dockerbuild.sh` once, then `docker compose -f docker-compose-prod.yml up -d`.
2. **Database wiring**
   - Option A: add a `pgvector/pgvector:pg16` Postgres service with a named volume to `docker-compose-prod.yml`, and point `POSTGRES_URI` at the Postgres service. Today prod reaches Postgres through `host.docker.internal:5432`.
   - Options B and C: point `POSTGRES_URI` (and, for Option B, `REDIS_URI`) at the RDS and ElastiCache endpoints.
3. **AWS IAM**
   - Add `token.actions.githubusercontent.com` as an OIDC identity provider.
   - Create a deploy role whose trust policy accepts only `repo:efwoods/anubis:ref:refs/heads/main`.
   - Grant the deploy role `ssm:SendCommand` (scoped to the instance and the `AWS-RunShellScript` document) and `ssm:GetCommandInvocation`.
4. **GitHub repository secrets**
   - `AWS_DEPLOY_ROLE_ARN`
   - `EC2_INSTANCE_ID`

### `scripts/deploy.sh` (runs on the instance)

```bash
#!/usr/bin/env bash
set -euo pipefail
target_commit="$1"
cd /opt/anubis
previous_commit=$(git rev-parse HEAD)
git fetch origin main
git checkout --force "$target_commit"
if git diff --name-only "$previous_commit" "$target_commit" | grep -qE '^(Dockerfile\.anubis\.base|pyproject\.toml|uv\.lock)$'; then
    docker build -t anubis-base:latest -f Dockerfile.anubis.base .
fi
docker build -t evdev3/anubis-langgraph-api:latest .
docker compose -f docker-compose-prod.yml up -d --remove-orphans
```

### `.github/workflows/deploy-production.yml`

```yaml
name: deploy-production
on:
  push:
    branches: [main]
concurrency:
  group: deploy-production
  cancel-in-progress: false
permissions:
  id-token: write
  contents: read
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_DEPLOY_ROLE_ARN }}
          aws-region: us-east-1
      - name: Run deploy script on the EC2 instance
        run: |
          command_id=$(aws ssm send-command \
            --instance-ids "${{ secrets.EC2_INSTANCE_ID }}" \
            --document-name AWS-RunShellScript \
            --parameters 'commands=["/opt/anubis/scripts/deploy.sh ${{ github.sha }}"],executionTimeout=["3600"]' \
            --query Command.CommandId --output text)
          while true; do
            command_status=$(aws ssm get-command-invocation --command-id "$command_id" \
              --instance-id "${{ secrets.EC2_INSTANCE_ID }}" --query Status --output text)
            case "$command_status" in
              Pending|InProgress|Delayed) sleep 15 ;;
              Success) exit 0 ;;
              *) aws ssm get-command-invocation --command-id "$command_id" \
                   --instance-id "${{ secrets.EC2_INSTANCE_ID }}" --query StandardErrorContent --output text; exit 1 ;;
            esac
          done
```

## 5. Warnings before enabling auto-deploy

- **Live Stripe writes on every deploy.** Every `docker compose -f docker-compose-prod.yml up` runs `stripe-provision --live`, which writes to the live Stripe account. Re-running `stripe-provision` changes nothing, so the writes are safe, but the writes happen on every deploy.
- **Downtime.** `dcrp.sh` runs `down` before `up`, so every `dcrp.sh` run takes the API offline. `scripts/deploy.sh` uses `up -d`, which recreates only containers whose image or configuration changed.
- **Branch.** The integration branch is `test`. Only pushes and merges into `main` trigger a deploy.
