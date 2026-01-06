# Terraform Backend Configuration
# Store state in S3 with DynamoDB locking for production

terraform {
  backend "s3" {
    bucket         = "nsi-terraform-state"
    key            = "negative-space-imaging/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# Note: Run the following before applying to create backend:
# aws s3api create-bucket --bucket nsi-terraform-state --region us-east-1
# aws s3api put-bucket-versioning --bucket nsi-terraform-state --versioning-configuration Status=Enabled
# aws dynamodb create-table --table-name terraform-locks --attribute-definitions AttributeName=LockID,AttributeType=S --key-schema AttributeName=LockID,KeyType=HASH --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
