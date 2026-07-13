

# testing
I need to test:
sign up
cancel subscription
manage subscription endpoint
verify subscription status
change subscription status

using the /message endpoints should test 72 tokens for each scenario

using the /update_avatar_identity_with_media endpoint and the following test (text) files /home/user/gh/anubis-project/wt/f-metering/data/shivon_zilis/test_tokens_1_tokens.md
/home/user/gh/anubis-project/wt/f-metering/data/shivon_zilis/test_tokens_2_tokens.md

test audio: /home/user/gh/anubis-project/wt/f-metering/data/test_data_avatar_evan_woods/reference_audio.wav

test video: /home/user/gh/anubis-project/wt/f-metering/data/test_data_avatar_evan_woods/1minuteFounderVideo_EvanWoods_NeuralNexus.mp4

test image: /home/user/gh/anubis-project/wt/f-metering/data/test_data_avatar_evan_woods/_test_image_token_usage_54_5kb.jpg

scenarios:
I need to know the token count for the above files, test above and below metered usage allotment with both scenarios pay-per-usage enabled and disabled for all three tiers and the free trial active for pro.


# Expected behavior and testing
--- 
free trial (pro)

free trial ending with payment (continue)

free trial ending without payment (switch to free)

beyond usage limit pro without payment (limit usage):
    - tokens
    - document uploads

beyond usage limit pro with payment (pay-per-use token then reset at the end of the month):
    - tokens
    - document uploads


beyond usage limit free without payment (limit usage):
    - tokens
    - document uploads

beyond usage limit free-tier (pay-per-use token beyond usage then reset at the end of the month) with payment:
    - tokens
    - document uploads


beyond usage limit premium-tier (pay-per-use token beyond usage then reset at the end of the month) with payment:
    - tokens
    - document uploads

beyond usage limit premium-tier (limit token usage unless pay-per-usage is enabled then reset at the end of the month) with payment:
    - tokens
    - document uploads
    - adapter training
    - adapter inference

----
# to be built (and verify if already created)
There needs to be rate limiting to prevent too many tokens being used per period that is set.
The manage subscription dashboard needs to show:
monthly:
    token allotment
    token usage
    document token usage
    adapter training token usage
    adapter inference token usage
current tier
switch tier
cancel subscription
enable/disable pay-per-usage tokens
manage payment information


# BUG: .env.dev environment is still using this url. I believe this is not for test and there is no user for e.woods.business@icloud.com although there is a test user subscription

/manage_subscription
{
  "url": "https://billing.stripe.com/p/login/eVq28s6XA53C5XpdqH1oI00",
  "message": "Follow this link to manage your subscription."
}

/verify_subscription_status
{
  "status": "active",
  "subscription_id": "sub_1TrgF6Limk9GVblr24vL1APN",
  "customer_id": "cus_UrP2LkbgqxE8aO",
  "email": "e.woods.business@icloud.com",
  "tier": "premium"
}

# Clearly there is meant to be a customer billing portal for all the above features:
https://dashboard.stripe.com/acct_1RyHMQLimk9GVblr/test/settings/billing/portal
customers should be able to see their token allotment and usage
 
Known bug: there are no names explictly for per usage (token usage should be clearly named as a product, currently there is an allotment of up to 5 million and 10 million but this is not clearly labeled as to WHAT this is).

Known bug: the current billing portal is a production link only not a test portal. The portal should be the standard customer user experience. 


Clearly:
I need the metering endpoints to declare and monitor per-token usage as well as subscriptions and signing up for the pro tier should start a free trial if the account has not already used the free trial; the end of the free trial without the activation of payment information (total token usage and allotment) verify subscription status should show token usage for all products and trial status (used, active, etc.)
  allows for free tier usage. There needs to be a visible way for a user to see their token usage and allotment. use Stripe MCP servers please for reference documentation.