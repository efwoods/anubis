# METERING
I need to estimate all costs:
 - inference
 - image to text
 - transcription (audio to text)


I also need to estimate costs with respect to space (storage in the database)

I need to be able to simply increase the cost estimation:
For example I will be using adapters in the future and I will need to scale to add these costs to the cost structure although I do not currently have an endpoint to create, save, attach, and infer adapters.  

# Pricing Estimation: 
## https://docs.together.ai/docs/fine-tuning/pricing

<!-- import os

from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

estimate = client.fine_tuning.estimate_price(
    training_file="file-abc123",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    n_epochs=3,
    training_method={"method": "sft"},
    training_type={"type": "Lora", "lora_r": 8},
)

print(estimate) -->

## Llama 4 dedicated endpoint 0.398 per minute with 15 minute auto shutdown
<!-- from together import Together

client = Together()

response = client.endpoints.create(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP4",
    display_name="e.woods.business@icloud.com: meta-llama - Llama-4-Maverick-17B-128E-Instruct-FP4",
    hardware="2x_nvidia_b200_180gb_sxm",
    min_replicas=1,
    max_replicas=1
)
print(response) -->


# resources to implement usage based billing per the metered usage past a subscription-tier threshold use. 
 - legacy stripe usage based billing: https://docs.stripe.com/billing/subscriptions/usage-based
 - pay as you go: https://docs.metronome.com/guides/pricing-packaging/billing-model-guides/pay-as-you-go
 - metronome hybrid: https://docs.metronome.com/guides/pricing-packaging/billing-model-guides/hybrid-business-models

# Evaluate cost model:
cost per upload per media type
cost per message
architectural costs
- scalable Architecture: 

## Create scalable architecture for multiple simultaneous requests and users and higher volume of data:

### Structure Graph:
<!-- https://docs.langchain.com/langsmith/agent-server -->

### Quantity Table of Resources
<!-- https://docs.langchain.com/langsmith/control-plane -->


I need to create a metering endpoint to identify the accrued token and storage usage. I will need to extend to three tiers with pay-as-you go usage past a predetermined allotment per tier. There needs to be a free trial. Different tiers allow different amounts of different capabilities such as uploading different forms of media and training and using adapters. The metering needs to account for tokens that are generated per message endpoint, how much storage is used when the messages are stored with the checkpointer in the database and how much data is stored in the database with respect to documents as well as the cost of transcription of those documents. The architecture is out of scope for this feature request. The stripe api and mcp server has been connected. There is an attempt to use the stripe sdk currently within this application. I will need to be able to tune the allotment per-tier token usage as well as per tier document uploads. Please follow the instructions as listed in _FEATURE.MD within this repository.


I need cost estimation. please use the legacy stripe api and evaluate all costs with respect to how much storage is required per document and per checkpointed message as well as inference costs and transcription costs in the update_avatar_identity_with_media pipeline as well as the message endpoints. I need to have three tiers of subscription (entry, pro, enterprise) with a free
  trial. I will need to be able to simply return the cost with respect to used space in the database and the monetary costs that are incurred from each message or upload. I will need to be able to extend this simply when new features are added. For example I will be changing the base model in the future and adding adapters (training adapters) this will add additional costs. I will
  need to limit the amount of tokens a user has available to use per tier and allow for pay-per-usage after an allotment. Different tiers have different available capabilities (n uploads, n messages, n tokens) which are configurable per tier at a later point in time. Entry level may be free. basemodel inference with a trained adapter will cost about $19000 for 24/7 service and will
  require $9 per $2000 approximately to cover the costs. I will need to know the costs required for scalable infrastructure as per the agent server and control plane in the future which will cost $500/month at minimum to scale and serve multiple people concurrently and house their data. Architecture is out of scope for this feature request. The thesis of this request are the
  metering via stripe legacy endpoints of the actual inference costs, awareness of data usage with respect to space accumulated per checkpointed message and document indexed, the ability to simple extend the metering when new features are added that accrue costs, the three subscription tiers, configurable usage per tier, and pay-per-usage past a threshold token count
  per-user-per-teir-per-period (anonymous users included; they are individually verifiable). I am referencing @_FEATURE.MD