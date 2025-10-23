# on EC2
sudo apt-get update
sudo apt-get install docker.io -y
sudo usermod -aG docker $USER
newgrp docker
sudo chmod 777 /var/run/docker.sock
docker run -d --name sonar -p 9000:9000 sonarqube:lts-community
ec2-public-ip:9000
username admin
password admin

configure variables at Gitlab settings --> CICD --> Variables section
SONAR_HOST_URL (xxxxxxxxxx)
SONAR_TOKEN (xxxxxxxxxxx)
DOCKER_USERNAME (narian318)
DOCKER_PASSWORD (xxxxxxxxxxxx)    # either docker hub password or PAT Token

Install Gitlab Runner on aws EC2 by following commands in this way at Gitlab settings --> CICD --> Runners section 
New Project Runner --> Enable shared Runners
# on EC2
sudo vi nn-gitlab-runner
# Download the binary for your system
sudo curl -L --output /usr/local/bin/gitlab-runner https://gitlab-runner-downloads.s3.amazonaws.com/latest/binaries/gitlab-runner-linux-amd64
# Give it permission to execute
sudo chmod +x /usr/local/bin/gitlab-runner
# Create a GitLab Runner user
sudo useradd --comment 'GitLab Runner' --create-home gitlab-runner --shell /bin/bash
# Install and run as a service
sudo gitlab-runner install --user=gitlab-runner --working-directory=/home/gitlab-runner
sudo gitlab-runner start

sudo chmod +x nn-gitlab-runner
./nn-gitlab-runner
sudo gitlab-runner start
sudo gitlab-runner register --url https://gitlab.com/ --registration-token <tokenxxxxxxxxxxxx> --> description(youtube), tags (naren, youtube), Executor (shell)
sudo gitlab-runner start
sudo gitlab-runner run





