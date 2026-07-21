main page with the following information and features (in this example, the page displays that a user has canceled a subscription already): 

Current subscription
Cancels Aug 13

Neural Nexus Free Tier
Price varies with usage
Your service will end on August 13, 2026.
Don't cancel subscription

Payment method
Visa •••• 4242
Expires 04/2042
Default
Add payment method

Billing information
Name
FIRSTNAME LASTNAME
Billing address
ROAD
CITY, STATE ZIP US
Phone number
(AREA) PHONE-NUMBER

Update information
Invoice history
Jul 13, 2026
$0.00
Paid

Neural Nexus Free Tier


# Missing Features
- usage allotment viewing for token usage allotment for messaging-inference-tokens/document-upload-tokens/adapter-training-tokens/adapter-inference-tokens
- switching tiers (token usage allotment for messaging-inference-tokens/document-upload-tokens/adapter-training-tokens/adapter-inference-tokens should rollover for the month; if paid for premium then down grades to free then unused token allotment continues; visual bar to show token allotment and usage)
- pay-per-use after allotment enable/disable button


# ULTIMATE REQUIREMENTS: 
I need a dashboard with the capabilities of the current base customer portal with the Missing Features. Features (see all invoices; change cancel subscription; update payment information; update billing information; see current subscription status; login with email verification; subscribe;)

There needs to be test and live environments for the customer portal.

This is the reference documentation for the Customer Portal:
https://docs.stripe.com/payments/advanced/build-subscriptions


# API USAGE REQUIREMENTS:
I need the messaging tokens metered (per-user allotment per-tier per month and per-user usage per-tier per-month and rate limited per-period (tokens/second for example in alignment with the following rate limits: https://developers.openai.com/api/docs/guides/rate-limits))

I need the document upload tokens metered (per-user allotment per-tier per month and per-user usage per-tier per-month and rate limited per-period (tokens/second for example in alignment with the following rate limits: https://developers.openai.com/api/docs/guides/rate-limits))

I need the adapter training tokens metered (per-user allotment per-tier per month and per-user usage per-tier per-month and rate limited per-period (tokens/second for example in alignment with the following rate limits: https://developers.openai.com/api/docs/guides/rate-limits))

I need adapter inference tokens metered (per-user allotment per-tier per month and per-user usage per-tier per-month and rate limited per-period (tokens/second for example in alignment with the following rate limits: https://developers.openai.com/api/docs/guides/rate-limits))

This usage needs to be reported with every /message endpoint call (for messaging tokens/adapter inference)
This usage needs to be reported with every adapter training job (to be created)
This usage needs to be reported with every update_avatar_identity_with_media call (for adapter training tokens only)

There needs to be a subscribe api endpoint call
there needs to be a cancel subscription api endpoint call
there needs to be a modify subscription api endpoint call (enable/disable pay-per-usage is a parameter)
there needs to be a verify subscription api endpoint call (also shows all token allotment and current usage to date for the period)

The period is an environment variable to be set for token allotment and usage
The there are rate limit environment variables to be set for messaging requests and update_avatar_identity_with_media requests

## Rate limits need to be in alignment with the following rate limits:
https://developers.openai.com/api/docs/guides/rate-limits


# Bottom Line
I need all metering now with api endpoints to show usage vs allotment in an api request. There needs to be limitation if pay-per-use is disabled and usage moves beyond allotment for each of the limits.

There are three tiers free, pro, premium and a free-trial with pro. 


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


# Test Situations:
Free tier:
messaging: 
 - under usage allotment messaging succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, messaging does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for free-tier

Pro tier:
messaging: 
 - under usage allotment messaging succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, messaging does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for pro-tier
Document token uploads:
 - under usage allotment upload succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, document upload does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for pro-tier

Pro tier (Free-Trial subscription is free; free usage to point of pay-per-usage for tokens for the period of the free-trial):
messaging: 
 - under usage allotment messaging succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, messaging does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for pro-tier
Document token uploads:
 - under usage allotment upload succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, document upload does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for pro-tier


Premium Tier:
messaging: 
 - under usage allotment messaging succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, messaging does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for premium-tier

Document token uploads:
 - under usage allotment upload succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, document upload does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for premium-tier

Adapter Training token allotment: 
 - under usage allotment adapter training succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, adapter training does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for premium-tier

Adapter Inference:
 - under usage allotment adapter inference messaging succeeds, metering increments
 - over usage allotment, pay-per-usage disabled, adapter inference messaging does not succeed
 - over usage allotment, pay-per-usage enabled, metering increments, pay-per-usage is charged per rate for premium-tier


# Additional situations to test:

## Usage
Make certain usage resets at the end of the period; 
Make certain usage for messaging tokens, adapter tokens, adapter training, and document uploads from premium is retained when downgrading to free or pro tier.
Make certain usage for messaging tokens, adapter tokens, adapter training, and document uploads from premium is cleared when upgrading from free or pro tier to the next tier. (moving up to a new tier allows for the usage to be reset for the usage period) 

## Free Trial
make certain pro with free trial moves to free tier if the trial ends without payment information
make certain pro with free trial continues to pro tier if the trial ends with payment information

# IMMEDIATE NEED:
I need the basic customer portal for test environment and live environments now with subscription tiers and free trial in addition to the above metering awareness and usage of this api. I will be building a customer portal to extend the capability of the current customer portal with usage-based awareness. The api needs to be functional and metered now and users need to be able to subscribe now. SCAN AND REWORK THIS CODEBASE TO FIT THESE IMMEDIATE NEEDS.

# Repository of the customer portal:
/home/user/gh/anubis-project/anubis-customer-portal
The customer portal uses a client-server architecture and will be hosted on Vercel:

# IMPORTANT:
READ THE INSTRUCTIONS DEFINING THE CUSTOMER PORTAL THAT WILL BE CREATED SO AS THE CUSTOMER PORTAL AND THIS REPOSITORY ARE IN ALIGNMENT:
/home/user/gh/anubis-project/anubis-customer-portal/_FEATURE.md

