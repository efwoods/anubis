# gcloud compute instances create nn-demo \
#   --machine-type=e2-micro \
#   --zone=us-central1-a \
#   --image-project=debian-cloud \
#   --boot-disk-size=100GB \
#   --tags=http-server,https-server \
#   --image-family=debian-12

# gcloud compute firewall-rules create allow-langgraph \
#   --allow tcp:8123 \
#   --target-tags=http-server

gcloud compute ssh neural-nexus-api-demo --zone=us-centra1-a

# On the VM:
# curl -fsSL https://get.docker.com | sh
# sudo usermod -aG docker $USER
# sudo apt-get install -y docker-compose-plugin
# newgrp docker

# (type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
# && sudo mkdir -p -m 755 /etc/apt/keyrings \
# && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
# && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
# && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
# && sudo apt update \
# && sudo apt install gh -y

