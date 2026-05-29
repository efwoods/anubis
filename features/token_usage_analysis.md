I need to collect the following information after every api request of the /message endpoints:

- aggregate model token use, cost, and latency

total tokens 

completion tokens

prompt tokens

and token cost

and response time (latency)



for the following inference types (total per request):

image model usage (total)

model with structured output (total)

inference model (every other init_model that does not have response_format or is not init_image_description_model)



and for each model that is used

for gpt-nano-5.4 model for example:

total tokens 

completion tokens

prompt tokens

and token cost

and response time (latency)



and I need the aggregate of 

total tokens 

completion tokens

prompt tokens

and token cost

and response time (latency)

for the entire request



so in a single request I will know

how much  each model cost in total with respect to time and money and how much each inference type cost with respect to time and money and how much the entire request cost with respect to time and money. 



Important!

Note how I am using AsyncLlamaAPIClientWrapper (this is free inference but I am calculating the cost if this cost would have been used. Don't use this now because I need real scaling capability). 



I need to log all these metrics to the api_metrics table in the pg database and visualize these metrics in grafana. 

