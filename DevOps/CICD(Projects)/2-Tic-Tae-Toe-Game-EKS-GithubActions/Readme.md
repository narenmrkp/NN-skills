Tic-Tac-Toe Deployment on Amazon EKS using GitHub Actions
Prerequisites

AWS account with IAM user/role that has EKS, ECR (if using), EC2, IAM permissions.

An EKS cluster (create using Terraform provided in repo or use an existing one).

kubectl and awscli installed (for verification).

GitHub repository with Actions enabled (this repo).

A container registry: Docker Hub (username + token) or Amazon ECR.

Step 1: Provision or Connect to EKS Cluster

If using Terraform in this repo:

Run terraform init

Run terraform apply -auto-approve

If using existing cluster: just note the cluster name and AWS region.

Update kubeconfig to test connectivity:

aws eks --region <AWS_REGION> update-kubeconfig --name <EKS_CLUSTER_NAME>

kubectl get nodes

Step 2: Setup GitHub Authentication

Option A: GitHub OIDC

Create an IAM Role for GitHub OIDC and allow your repo to assume it.

Grant EKS + ECR/DockerHub permissions.

Add AWS_ROLE_TO_ASSUME (role ARN) as GitHub variable.

Option B: Self-Hosted Runner on EC2

Launch Ubuntu EC2 with IAM Role (EKS + registry permissions).

Register it as a self-hosted runner from repo → Settings → Actions → Runners.

Start runner as service.

Step 3: Configure GitHub Repo Secrets & Variables

In your repo → Settings → Secrets and variables → Actions:

Add DOCKERHUB_USERNAME and DOCKERHUB_TOKEN (if using Docker Hub).

Add variables:

AWS_REGION = your AWS region (e.g., ap-south-1)

EKS_CLUSTER_NAME = your cluster name (e.g., EKS_CLOUD)

KUBE_NAMESPACE = target namespace (e.g., default)

IMAGE_REPO = Docker Hub repo (e.g., username/tic-tac-toe) or ECR repo URI

IMAGE_TAG = v1 (or latest)

AWS_ROLE_TO_ASSUME = IAM Role ARN (only if using OIDC)

Step 4: Verify Project Files

In k8s/deployment.yaml → confirm image: <IMAGE_REPO>:<IMAGE_TAG>.

In k8s/service.yaml → confirm type: LoadBalancer and targetPort = app port.

In .github/workflows/deploy.yml → confirm it points to your variables.

Step 5: Run the CI/CD Pipeline

Push changes to the main branch OR trigger workflow manually:

Go to Actions tab → Select workflow → Run workflow

GitHub Actions will automatically:

Build the Docker image

Push image to registry (Docker Hub or ECR)

Update kubeconfig for your EKS cluster

Apply deployment and service manifests

Step 6: Verify Deployment

After pipeline completes, run:

kubectl get deploy,po,svc -n <KUBE_NAMESPACE>

kubectl describe svc tic-tac-toe-service -n <KUBE_NAMESPACE>

Get the service’s external hostname/IP:

kubectl get svc tic-tac-toe-service -n <KUBE_NAMESPACE> -o wide

Open the external link in browser → Tic-Tac-Toe game should load.

Step 7: Cleanup

To remove deployed resources:

kubectl delete svc tic-tac-toe-service -n <KUBE_NAMESPACE>

kubectl delete deploy tic-tac-toe -n <KUBE_NAMESPACE>

If you created the cluster with Terraform:

terraform destroy -auto-approve

Troubleshooting Commands

Check logs of pods:

kubectl logs deploy/tic-tac-toe -n <KUBE_NAMESPACE>

Check events if service pending:

kubectl describe svc tic-tac-toe-service -n <KUBE_NAMESPACE>

Verify kubeconfig context:

kubectl config get-contexts
