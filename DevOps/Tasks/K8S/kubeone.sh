create EC2 t2.medium 24.04 ubuntu server
sudo apt update && sudo apt install unzip -y
curl -sfL https://get.kubeone.io | sh

sudo apt-get update && sudo apt-get install -y gnupg software-properties-common
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg > /dev/null
gpg --no-default-keyring --keyring /usr/share/keyrings/hashicorp-archive-keyring.gpg --fingerprint
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update
sudo apt-get install terraform -y

sudo apt install unzip -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

Create IAM user with admin access and copy access, secret access keys
export <VARIABLE_NAME>=<VARIABLE_VALUE>
export <AWS_ACCESS_KEY_ID>=AWS_ACCESS_KEY_ID
export <AWS_SECRET_ACCESS_KEY>=AWS_SECRET_ACCESS_KEY
aws configure

ssh-keygen -t rsa -b 4096
cd ./kubeone_1.9.0_linux_amd64/examples/terraform/aws/
terraform init
vi terraform.tfvars
cluster_name = "kubeone-cluster"
ssh_public_key_file = "~/.ssh/id_rsa.pub"
terraform plan
terraform apply
terraform output -json > tf.json
vi kubeone.yaml
apiVersion: kubeone.k8c.io/v1beta2
kind: KubeOneCluster
versions:
  kubernetes: '1.30.0'
cloudProvider:
  aws: {}
  external: true
kubeone apply -m kubeone.yaml -t tf.json

# Start SSH agent correctly
eval "$(ssh-agent)"
# Verify the environment variables
echo $SSH_AUTH_SOCK
echo $SSH_AGENT_PID
# Add your private key
ssh-add ~/.ssh/id_rsa
# Verify keys are added
ssh-add -l
# Fix SSH directory permissions
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
kubeone apply -m kubeone.yaml -t tf.json
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
chmod +x kubectl
mkdir -p ~/.local/bin
mv ./kubectl ~/.local/bin/kubectl
kubectl --kubeconfig=<cluster_name>-kubeconfig
kubectl get nodes --kubeconfig=kubeone-cluster-kubeconfig
cp kubeone-cluster-kubeconfig ~/.kube/config
kubectl get nodes
kubeone reset --manifest kubeone.yaml -t tf.json
terraform destroy
