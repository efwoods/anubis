I need to make 3 tiers of subscriptions with Stripe. 
There is a free tier, pro, and premium
there is a free trial for pro
pro allows you to upload n documents
each tier has an allotment of tokens (total tokens), and an allotment of uploads (aggregates to total tokens) per tier.

# Free tier allows for sending and receiving messages only, and the creation of avatars. there is a set amount of allotted tokens to be used per user per month. pay-per-use past this allotment.

# Pro allows for sending and receiving messages, and the updating_avatar_identity_with_media with an allotment of n documents, images, audio, video, urls, text (all quantified into tokens and charged against token use). pay-per-use past this allotment is allowed for both messaging and document uploads. (all document processing is estimated with respect to cost using the diarization and model with structured output cost and charged against the total allotment for this tier) token usage past the set allotment per-user-per-month is charged as usage based. There is a flat monthly fee after the trial period for this tier. otherwise, the user may continue to message under the free tier.

# Premium allows for all capability of the previous tiers, with the addition of training adapters for model fine-tuning (allows for an improvement of quality). There is an allotment of tokens, and support for training adapters. cost of training adapters is monitored. Each user may make n adapters. creation beyond this is pay-per-use. There is a flat rate for this tier monthly with the addition of an allotment of tokens with pay-per-use past this monthly allotment and the training of n adapters with pay-per-use of training of each adapter after this allotment (monthly). The document upload restrictions from pro are applied here with a larger number of uploads allowed. 

There is an allotment of tokens per user month per tier, uploads per user month (such that the uploads distill to tokens and do not overbudget), and adapters trained per month (such that the training of adapters do not go over budget). Adapter inference is charged (token usage) at a different rate than standard inference with a fallback to a non-attached adapter or pay-per-use past this model's inference usage. free tier can message, pro tier can upload documents, premium can train adapters and all tiers inherit the capabilities of previous tiers. 

I need to reserve a set amount of tokens for document uploads and adapter training specifically for each allotment as rational per these specifications (messaging tokens usage allotment, document upload allotment, adapter training allotment, and adapter inference allotment per user per month all with pay-per-use past a specific allotment and all per tier)

[python sdk documentation:](https://docs.stripe.com/api?lang=python)
[subscriptions](https://docs.stripe.com/api/subscriptions)
[metering](https://docs.stripe.com/api/billing/meter)

STRIPE_PAYMENT_URL="https://buy.stripe.com/cNi4gA2Hk3Zy2Ld86n1oI03"
STRIPE_MANAGE_SUBSCRIPTION_URL="https://billing.stripe.com/p/login/eVq28s6XA53C5XpdqH1oI00"


stripe payment link (needs to have all three tiers and usage-based-monitoring as per the following: (messaging tokens usage allotment, document upload allotment, adapter training allotment, and adapter inference allotment per user per month all with pay-per-use past a specific allotment and all per tier))
Skip to content
Payment Link
Neural Nexus API Pro Subscription with Trial
for
$20.00 USD / month
Copy and share to start accepting payments with this link.
https://buy.stripe.com/cNi4gA2Hk3Zy2Ld86n1oI03
URL parameters
Buy button
More options
Overview
Products
	
Name
	
Quantity
	
Adjustable Quantity
	
Neural Nexus API Pro Subscription with Trial
$20.00 USD / month
Product tax code: General - Electronically Supplied Services
Tax included in price: Yes
	
1
	
No
Payment methods
Manage
No payment information collected at checkout. Setup subscription email reminders so Stripe can automatically ask your customer to add their payment information before the trial ends.
If your customer chooses an option that would require a payment (e.g. a cross-sell), the following payment methods is available at checkout:
Card
Apple Pay
Klarna
Link
Cash App Pay
Amazon Pay
Details
Status
Active
Date created
Apr 9, 7:15 PM
Limited use
No
Allow promotion codes
No
Collect addresses
Billing
Collect phone numbers
No
Collect full names
Yes
Collect business names
No
Allow business customers to provide tax IDs
No
Free trial
30 days
After trial ends
Pause subscription if no payment method is provided
Collect tax automatically
Yes
Collect terms of service agreement
No
Confirmation page
Default
Deactivated link page
Default
Call to action button
Subscribe
MetadataUse metadata to store custom additional information. View docs
Edit metadata
No metadata
Events
A payment link with ID plink_1TKRnBLimk9GVblrGtWquA6q was created
	
	
4/9/26, 7:15:21 PM
Logs
200 OK
	
POST /v1/payment_links
	
4/9/26, 7:15:21 PM
Tips for using your link
Increase conversion
Let customers check out faster by enabling Link.
Turn on Link
Fight climate change
Fight climate change by contributing a fraction of your revenue.
Turn on Stripe Climate
Boost your sales
Increase revenue by upselling to a longer billing period
Add upsells
Increase conversion
Win more international customers by turning on relevant, popular payment methods.
Manage payment methods
Customize your brand
Match the look and feel of your brand by adding your logo, fonts, and colors.
Brand settings
Add your domain
Use your own domain for your payment links.
Custom domain settings
Get notified
Get an email after every successful payment by managing your settings.
Manage communication preferences
Preview
Edit
buy.stripe.com
Use your domain

Navigated to Payment Links – Afterlife Systems Inc – Stripe
Afterlife Systems Inc

    Home
    Balances
    Transactions
    Customers
    Product catalog

Shortcuts

    Link
    Usage-based billing
    Subscriptions
    Payment Links
    Billing overview

Products

    Treasury
    Payments
    Billing
    Reporting
    Apps
    More
        Link
        Profiles
        Tax
        Connect
        Identity
        Atlas
        Issuing
        Financial Connections
        Climate
        Workflows

Search
Developers
plink_1TKRnBLimk9GVb...
