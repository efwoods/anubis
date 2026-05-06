---
name: cloud-cost
description: Cloud + LLM cost analysis SOP. Use whenever the prompt mentions cost, spend, burn, AWS, Azure, OpenAI, ChatGPT, runway, or budget.
license: MIT
metadata:
  author: anubis
  version: "0.1"
allowed-tools: read_file, write_file, execute, ingest_aws_cost, ingest_azure_cost, ingest_chatgpt_usage
---

# Cloud Cost SOP

## When to use

Prompts mentioning cost, spend, burn, runway, budget, AWS, Azure, OpenAI, ChatGPT, or any service-name regex.

## What to produce

A cost section that fills `CompanyOverview.burn_rate_usd_30d` and feeds `what_worked` / `what_did_not_work` with deltas vs prior window.

## Steps

### 1. Ingest three sources in parallel

- `ingest_aws_cost(days=window_days*2)` so we have current + prior window.
- `ingest_azure_cost(days=window_days*2)`.
- `ingest_chatgpt_usage(days=window_days*2)`.

### 2. Normalize to a common shape

After loading, melt to `[date, source, sub_dim, cost_usd]`:

- AWS: `sub_dim = service + " / " + usage_type`
- Azure: `sub_dim = service_name + " / " + resource_group`
- ChatGPT: `sub_dim = model`

### 3. Compute KPIs

- `daily_total_usd[t]` summed across sources.
- `top3_drivers` = top 3 `sub_dim` by total cost in the window.
- `delta_pct` per `sub_dim` vs prior window, sorted.
- 30-day burn = sum of last 30 daily totals (clip the join window).

### 4. Excellence rules

- Any `sub_dim` with `delta_pct >= +25%` AND absolute increase >= $50 -> goes under `what_did_not_work` with a one-line optimization suggestion.
- Any `sub_dim` with `delta_pct <= -10%` AND absolute decrease >= $20 -> goes under `what_worked`.
- If `chatgpt` cost > 30% of total spend, add a `new_standard`: "Move <topic> off premium model to gpt-mini by EOW".
- If AWS NAT Gateway or Data Transfer is in top3 drivers, add a `new_standard` to investigate VPC endpoints.

### 5. Output chart

Plot to `/artifacts/<run>/cost.png`:

- Stacked area chart (date, cost_usd) split by source.
- Right-aligned legend, dollar y-axis.

## Notes

- AWS Cost Explorer is `us-east-1` only. Override `aws_default_region` before client init.
- Azure Cost Management call must use `client_credentials` flow — service principal, not user OAuth.
- OpenAI admin endpoint is `/v1/organization/usage/completions` and `/v1/organization/costs`. Both require an `sk-admin-...` key, distinct from regular API keys.
