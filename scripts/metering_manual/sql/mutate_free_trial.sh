# Set env.sh
# export NN_API="http://localhost:8123"       # dev API (NOT 8900 on this branch)
# export NN_API_KEY="sk-tDFV1JPbM-cTP7vWAYtzNdOP9DqSMglRTd3nLOkZWPk"  # raw API key for the test account
# export NN_AVATAR_ID="70fc7621-f590-4d71-8081-8786b8e7a810" # assistant_id from /create_avatar
# export NN_USER_ID="auth0|6a6177ccfd677c585b38d703"         # Auth0 user_id — for SQL meter injection
# export NN_CUSTOMER="cus_Uw4DtZm5rA9xTg"         # Stripe customer id — for SQL meter injection
# source ../env.sh


# end trial now → status flips trialing → active (webhook syncs Auth0)
python ../stripe_setup.py end-trial
