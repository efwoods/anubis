# Copy to /etc/pgadmin/config_system.py (see pgadmin.evaluate_config).
# sudo mkdir -p /etc/pgadmin
# sudo cp deploy/pgadmin-config_system.py /etc/pgadmin/config_system.py
# sudo chmod 644 /etc/pgadmin/config_system.py
#
# Default MASTER_PASSWORD_REQUIRED=True forces master-password + OS secret storage;
# that flow often hangs in pgAdmin desktop on Linux when saving a server (SQLite server table stays empty).
MASTER_PASSWORD_REQUIRED = False
# If saving still hangs, uncomment (stores encryption key in-process only; less OS integration):
# USE_OS_SECRET_STORAGE = False
