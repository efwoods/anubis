gcloud compute instances create anubis-demo \
  --machine-type=e2-micro \
  --zone=us-central1-a \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --tags=http-server,https-server


curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
sudo apt-get install -y docker-compose-plugin
newgrp docker

gcloud compute ssh anubis-demo --zone=us-centra1-a