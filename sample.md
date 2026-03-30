What more do I need to do to test thiis locally and deploy to the vm and test from cloudflare to the full-end to end lifecycle of an api call?


Unable to connect

Firefox can’t establish a connection to the server at localhost.

    The site could be temporarily unavailable or too busy. Try again in a few moments.
    If you are unable to load any pages, check your computer’s network connection.
    If your computer or network is protected by a firewall or proxy, make sure that Firefox is permitted to access the web.


DNS records:
		Type
	Name
	Content
	Proxy status
	TTL
	Actions
		A
	
api
	
172.206.31.150
	

Proxied
	
Auto
	
		A
	
neuralnexus.site
	
216.198.79.1
	

DNS only
	
Auto
	

Origin SSL/Encryption is set to full

Origin server certificates exist:
Origin Server

Manage origin certificates and configure how Cloudflare authenticates connections to your origin server.
Origin server SSL/TLS documentation
Origin Certificates

Generate a free Cloudflare-signed TLS certificate to install on your origin server. These certificates are only trusted by Cloudflare - if your origin receives traffic from outside the Cloudflare network, use a publicly trusted certificate instead.
Hosts
	Expires On
	 
*.neuralnexus.site, neuralnexus.site (2 hosts)
Mar 26, 2041
Download


AZURE LOAD BALANCER RULE CREATED FOR PORT 443
ACCEPT ALL SOURCE PORTS TO DESTINATION PORT 443 ON THE VM

the certificates are loaded on the local under ./certificates for the origin_cert.pem and the cloudflare_private.key both created from the cloudflare origin server 



volumes:
    langgraph-data:
        driver: local
        
services:
    nginx:
      image: nginx:latest
      ports:
        - "443:443"  # Only Nginx is exposed to the VM/Load Balancer
      volumes:
        - ./nginx.conf:/etc/nginx/nginx.conf
        - ./certificate:/etc/ssl/private/
      networks:
        - api-network
    langgraph-api:
        image: evdev3/anubis-langgraph-api:latest
        pull_policy: never
        dns:
            - 168.83.129.16
            - 8.8.8.8
        ports:
            - "8123:8000"
            - "5678:5678"
        healthcheck:
            test: python /api/healthcheck.py
            interval: 60s
            start_interval: 1s
            start_period: 10s
        env_file:
            - .env
        volumes:
            - .:/deps/anubis
            - ~/.cache/huggingface:/root/.cache/huggingface
        networks:
            - api-network

networks:
    api-network:
        driver: bridge

 