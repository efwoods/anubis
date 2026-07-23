# start a trial (independently of meter injects)
python scripts/metering_manual/stripe_setup.py create --tier premium --trial-days 14
# or pro with default trial from signup; or:
python scripts/metering_manual/stripe_setup.py create --tier pro --trial-days 30

# end trial now → status flips trialing → active (webhook syncs Auth0)
python scripts/metering_manual/stripe_setup.py end-trial
# leave trial / drop to free
python scripts/metering_manual/stripe_setup.py cancel
# z
