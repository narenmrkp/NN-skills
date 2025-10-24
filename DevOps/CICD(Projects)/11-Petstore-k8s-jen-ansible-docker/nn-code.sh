create EC2(t2.large, ubuntu 22.04) --> install Jenkins through script 
vi jenkins.sh --> sudo chmod 777 jenkins.sh --> ./jenkins.sh
  # jenkins port change from 8080 to 8090
sudo systemctl stop jenkins
sudo systemctl status jenkins
cd /etc/default
sudo vi jenkins   #chnage port HTTP_PORT=8090 and save and exit
cd /lib/systemd/system
sudo vi jenkins.service  #change Environments="Jenkins_port=8090" save and exit
sudo systemctl daemon-reload
sudo systemctl restart jenkins
sudo systemctl status jenkins
EC2 Public IP Address:8090
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
  # Docker
sudo apt-get update
sudo apt-get install docker.io -y
sudo usermod -aG docker $USER
newgrp docker
sudo chmod 777 /var/run/docker.sock
docker run -d --name sonar -p 9000:9000 sonarqube:lts-community
SONAR access: EC2 Public IP Address:9000 (admin, admin)
  # Install trivy through script
vi trivy.sh -->  sudo chmod 777 trivy.sh --> ./trivy.sh

Install all needed plugins in Manage Jenkins --> Available plugins ( Eclipse Temurin jdk, sonarqube scanner, OWASP, ansible, 5 docker, kubernetes)
Configure Java & maven --> Manage Jenkins → Tools → Install JDK(17), jdk-17.0.8.1+1 and Maven3(3.6.0) → Click on Apply and Save
Create Jenkins Pipeline Job --> Build
Sonar dash board --> administration --> security --> users --> token (Sonar-token)
Administration–> Configuration–>Webhooks --> create --> Jenkins, <http://jenkins-public-ip:8080>/sonarqube-webhook/ --> Create
Manage Jenkins → Credentials → Add Secret Text (Sonar-token, xxxxxxxx) --> save
Manage Jenkins --> Credentials (Global) --> Docker --> username(narian318), password(xxxxxxxx) --> save
Manage Jenkins --> System --> sonar-server, xxxx:9000/, Sonar-token --> Apply & Save
Manage Jenkins --> Tools --> Sonarqube Scanner Installation --> sonar-scanner, Install Automatically(Sonarqube scanner 4.8.0.2856)
Manage Jenkins --> Tools --> DP-Check --> Install Automatically (dependancy check(6.5.1) --> Apply & Save
Manage Jenkins --> Tools --> docker --> Install Automatically(latest)

  # install Ansible on the Jenkins server
sudo apt-get update
sudo apt install software-properties-common
sudo add-apt-repository --yes --update ppa:ansible/ansible
sudo apt install python3
sudo apt install ansible -y
sudo apt install ansible-core -y
cd /etc/ansible
sudo vi hosts
[local] #any name you want
<IP of Jenkins>  Esc:wq!(save)

Configure Ansible credentials --> Manage Jenkins --> credentials(global) --> SSH with user pvt key (ID(SSH), user(ubuntu), pvt(paste total pem key of Jenkins EC2 key pair) --> create
Manage Jenkins --> Tools --> Ansible --> name(ansible), path(/usr/bin/) --> save
After running Jenkins pipeline --> output: jenkins-ip:8081/jpetstore    Note: Jenkinsfile, ansible playbooks are from Repo
  # kuberenetes setup
create 2 EC2s(t2.medium,ubuntu 20.04)
  # on both EC2s
sudo apt update
sudo apt install curl
curl -LO https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
sudo su --> hostname master --> bash --> clear
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod –aG docker Ubuntu
newgrp docker
sudo chmod 777 /var/run/docker.sock
sudo curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo tee /etc/apt/sources.list.d/kubernetes.list <<EOF
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo snap install kube-apiserver
  # on Master EC2 only
sudo kubeadm init --pod-network-cidr=10.244.0.0/16
# in case your in root exit from it and run below commands
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
  # on Worker Node only
sudo kubeadm join <master-node-ip>:<master-node-port> --token <token> --discovery-token-ca-cert-hash <hash>
cat .kube/config --> copy the entire configfile text and save as secret.txt in local Laptop
Manage Jenkins --> credentials(global) --> Secret file --> scope(global), ID(k8s), upload our secret.txt file --> create
  # on Jenkins EC2 (means Ansible Master)
ssh-keygen --> cd .ssh --> cat id_rsa.pub  #copy this public key
  # on k8s Master EC2
cd .ssh --> sudo vi authorized_keys --> paste the above public key (from ssh-keygen)
  # on Jenkins EC2 (Ansible Master)
cd /etc/ansible --> sudo vi hosts 
[k8s]#any name you want
<public ip of k8s-master>
ansible -m ping k8s
ansible -m ping all#use this one
  # on k8s Master
kubectl get all
kubectl get svc
access petstore App from Worker Node IP: slave-ip:serviceport(30699);/jpetstore

Jenkins Pipeline --> copy & paste Jenkins pipeline script (from Jenkinsfile) --> Build --> <EC2-public-ip:3000> (To access Amazon App in browser)

  # Jenkins.sh
#!/bin/bash
sudo apt update -y
#sudo apt upgrade -y
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc
echo "deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
sudo apt update -y
sudo apt install temurin-17-jdk -y
/usr/bin/java --version
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee \
                  /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
                  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
                              /etc/apt/sources.list.d/jenkins.list > /dev/null
sudo apt-get update -y
sudo apt-get install jenkins -y
sudo systemctl start jenkins
sudo systemctl status jenkins
  # trivy.sh
sudo apt-get install wget apt-transport-https gnupg lsb-release -y
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor | sudo tee /usr/share/keyrings/trivy.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy -y
