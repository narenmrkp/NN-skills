# Deploy Tic-Tac-Toe Game on EKS using GitHub Actions

This repository demonstrates how to **deploy a Tic-Tac-Toe game on an AWS EKS cluster** using a fully automated **CI/CD pipeline with GitHub Actions**.  
It will help you understand how to integrate GitHub Actions with AWS EKS, Docker, and Kubernetes for a real-world project deployment.

---

## 📌 Prerequisites

Before starting, ensure the following tools and accounts are ready:

- **AWS Account** with programmatic access (IAM User with EKS, ECR, and EC2 permissions).
- **GitHub Account** with a repository containing this project code.
- **kubectl** installed → [Install Guide](https://kubernetes.io/docs/tasks/tools/).
- **AWS CLI** installed and configured (`aws configure`).
- **eksctl** installed → [Install Guide](https://eksctl.io/).
- **Docker** installed and running.
- **Helm** installed → [Install Guide](https://helm.sh/docs/intro/install/).

---

## ⚙️ Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

⚙️ Step 2: Create an EKS Cluster
eksctl create cluster --name tic-tac-toe-cluster --region ap-south-1 --node-type t3.medium --nodes 2
kubectl get nodes





Create EC2 (ubuntu, t2.medium) and create IAM Role with Administrator/S3 Full, EC2 Full, EKS Full accesses and attach this Role to this EC2
Created self-hosted runner to this EC2 from Github Repo Actions Tab
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.310.2.tar.gz -L https://github.com/actions/runner/releases/download/v2.310.2/actions-runner-linux-x64-2.310.2.tar.gz
echo "fb28a1c3715e0a6c5051af0e6eeff9c255009e2eec6fb08bc2708277fbb49f93  actions-runner-linux-x64-2.310.2.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.310.2.tar.gz
./config.sh --url https://github.com/Aj7Ay/Netflix-clone --token A2MXW4323ALGB72GGLH34NLFGI2T4
Note: Enter needed details (Name, labels...etc)
./run.sh

on EC2:
sudo apt-get update
sudo apt install docker.io -y
sudo usermod -aG docker ubuntu
newgrp docker
sudo chmod 777 /var/run/docker.sock
docker run -d --name sonar -p 9000:9000 sonarqube:lts-community
ec2-public-ip:9000    (login admin, password admin)
Integrate Sonarqube and Github Actions properly as per instructions mentioned in Sonarqube dashboard under "Manually" Tab
In Github Repo, create sonar-project.properties file with inside content as sonar.projectKey=Tic-game, Next create yaml file .github/workflows/build.yml
and also configure needed passwords, secrets properly inside of Github Action secrets section
On EC2:
git clone https://github.com/Aj7Ay/TIC-TAC-TOE.git
cd TIC-TAC-TOE
cd Eks-terraform    # (Configure S3 bucket name in the backend file)
terraform init
terraform validate
terraform plan
terraform apply
<ec2-ip:3000>      # output
kubectl get all

cd /home/ubuntu
cd TIC-TAC-TOE
cd Eks-terraform
terraform destroy --auto-approve





