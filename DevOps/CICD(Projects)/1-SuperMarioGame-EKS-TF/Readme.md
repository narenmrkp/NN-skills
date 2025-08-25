Create EC2 (ubuntu, t2.micro) in AWS and Create IAM Role for EC2 with Admistrator access and attach this Role to this EC2
git clone https://github.com/Aj7Ay/k8s-mario.git
cd k8s-mario
sudo chmod +x script.sh
./script.sh    # (This script will install Terraform, AWS cli, Kubectl, Docker)
cd EKS-TF      # (We need to configure s3 bucket name in the backend.tf file as per our wish)
terraform init
terraform validate
terraform plan
terraform apply --auto-approve
aws eks update-kubeconfig --name EKS_CLOUD --region ap-south-1      # (updating the kubernetes config)
cd ..
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get all
kubectl describe service mario-service    # (Paste the ingress link in a browser and we can access the Mario game)

kubectl get all
kubectl delete service mario-service
kubectl delete deployment mario-deployment
terraform destroy --auto-approve


# Deploy Mario Game on AWS EKS using Terraform & Kubernetes

## 1. Create EC2 and IAM Role
- Launch an **EC2 instance** (Ubuntu, `t2.micro`) in AWS.  
- Create an **IAM Role** for EC2 with **Administrator Access**.  
- Attach this IAM Role to your EC2 instance.  

---

## 2. Clone Repository
```bash
git clone https://github.com/Aj7Ay/k8s-mario.git
cd k8s-mario

## 3. Run Installation Script
sudo chmod +x script.sh
./script.sh    # Installs Terraform, AWS CLI, Kubectl, Docker

## 4. Setup Terraform for EKS
cd EKS-TF

# Edit backend.tf and configure your own S3 bucket name
terraform init
terraform validate
terraform plan
terraform apply --auto-approve

## 5. Update Kubeconfig
aws eks update-kubeconfig --name EKS_CLOUD --region ap-south-1

## 6. Deploy Mario Game on Kubernetes
cd ..
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get all
kubectl describe service mario-service    # Copy ingress link and open in browser

## 7. Clean Up Resources
kubectl get all
kubectl delete service mario-service
kubectl delete deployment mario-deployment
terraform destroy --auto-approve

