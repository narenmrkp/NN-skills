create Ec2(t2.medium,ubuntu 22.04) with needed ports opened(80,443,3000,9000)
  # on EC2 (docker setup)
sudo apt-get update
sudo apt install docker.io -y
sudo usermod -aG docker ubuntu
newgrp docker
sudo chmod 777 /var/run/docker.sock
docker run -d --name sonar -p 9000:9000 sonarqube:lts-community
access sonar ec2-public-ip:9000 (admin, admin)
sonar dashboard --> Manually --> Project Name(Netflix), main branch --> setup --> with Github actions --> follow instructions --> continue --> select others(Js, Ts, Python, GO, php)
Github Actions --> secrets --> SONAR_TOKEN (xxxxxx), SONAR_HOST_URL(https://xxxxxx:9000) --> add secrets  [token, url comes from above sonar instructions
Github Actions --> secrets --> DOCKERHUB_USERNAME (narian318), DOCKERHUB_TOKEN (passwordxxxx or PAT Tokenxxxx)
create file sonar-project.properties --> sonar.projectKey=Netflix
next create file .github/workflows/build.yml --> copy & paste script from above sonar dashboard instructions, remain docker build & deploy code lines we need to add to this file later
create TMDB API key --> login --> settings --> API --> create (developer) --> fill the details --> get the aPi key for use in build.yml file
  # Runner setup on EC2
Github Repo settings --> Actions --> Runners --> self-hosted runner --> Linux(x64) --> copy & paste commands one by one on EC2
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.310.2.tar.gz -L https://github.com/actions/runner/releases/download/v2.310.2/actions-runner-linux-x64-2.310.2.tar.gz
echo "fbxxxxxxxxxxxxxxxxxxxxxxxx  actions-runner-linux-x64-2.310.2.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.310.2.tar.gz
./config.sh --url https://github.com/Aj7Ay/Netflix-clone --token A2xxxxxxxxxxxxxxxxxxxxxxxxx
Runner name (self-hosted or any name aws-netflix...etc) --> labels(any name) --> ./run.sh
Run the trigger for pipeline in Github --> access App Ec2-instance-ip:8081
