This project explains how to deploy web app on S3 through Github Actions CICD pipeline
AWS S3 bkt policy:
{
"Version": "2012-10-17",
"Statement": [
{
"Sid": "PublicReadForGetBucketObjects",
"Effect": "Allow",
"Principal": "*",
"Action": "s3:GetObject",
"Resource": "arn:aws:s3:::<bucket name>/*"
}
]
}
