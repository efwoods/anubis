# --- shell session ---
export NN_API=http://localhost:8123
export NN_API_KEY='sk-rhZGqpwMTIARI_Lzm0GsPYNhdkL0GEiAj8FbvCMlP18'          # e.g. from rotate_api_key
export NN_AVATAR_ID=df95e4c2-0032-439a-be95-a857d0d39662
export NN_USER_ID='auth0|6a5fb6cd601f497f56f5aa37'   # match /verify
export NN_CUSTOMER_ID=cus_UvHevUUQ1nJnE1             # match /verify
export STRIPE_SECRET_KEY="$(grep '^STRIPE_SECRET_KEY=' .env.dev | cut -d= -f2- | tr -d '"')"



# tiny upload fixture (~few tokens)
mkdir -p /tmp/nn_metering
printf 'token\n' > /tmp/nn_metering/test_tokens_1_tokens.md
export NN_UPLOAD_FILE=/tmp/nn_metering/test_tokens_1_tokens.md



# --- shell session ---
# echo $NN_API
# echo $NN_API_KEY
# echo $NN_AVATAR_ID
# echo $NN_USER_ID
# echo $NN_CUSTOMER_ID
# echo $STRIPE_SECRET_KEY