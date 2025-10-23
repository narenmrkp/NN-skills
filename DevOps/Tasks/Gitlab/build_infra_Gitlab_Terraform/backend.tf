terraform {
  backend "s3" {
    bucket = "nnbkt2025" # Replace with your actual S3 bucket name
    key    = "Gitlab/terraform.tfstate"
    region = "ap-south-1"
  }
}
