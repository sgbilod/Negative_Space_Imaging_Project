# Terraform Infrastructure as Code - Negative Space Imaging Project

## Overview

This Terraform configuration deploys a production-grade, multi-tier infrastructure on AWS for the Negative Space Imaging Project. It includes:

- **VPC**: Multi-AZ VPC with public/private subnets, NAT Gateway, IGW, and security groups
- **EKS**: Kubernetes cluster with managed node groups, OIDC provider for IRSA, and add-ons
- **RDS**: PostgreSQL database with multi-AZ, encryption, backups, and connection pooling
- **ElastiCache**: Redis cluster with high availability, encryption, and monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              AWS Region (us-east-1)                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │              VPC (10.x.0.0/16)                  │   │
│  ├─────────────────────────────────────────────────┤   │
│  │  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │   Public AZ1 │  │   Public AZ2 │             │   │
│  │  │  Subnets     │  │  Subnets     │             │   │
│  │  │  (IGW)       │  │  (IGW)       │             │   │
│  │  └──────────────┘  └──────────────┘             │   │
│  │       ▲                    ▲                     │   │
│  │       │         NAT        │                     │   │
│  │       └────────────────────┘                     │   │
│  │  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │ Private AZ1  │  │ Private AZ2  │             │   │
│  │  │  EKS Nodes   │  │  EKS Nodes   │             │   │
│  │  │  RDS/Cache   │  │  RDS/Cache   │             │   │
│  │  └──────────────┘  └──────────────┘             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   EKS        │  │   RDS        │  │ElastiCache │  │
│  │ Kubernetes   │  │ PostgreSQL   │  │   Redis    │  │
│  │  Cluster     │  │   Database   │  │   Cache    │  │
│  └──────────────┘  └──────────────┘  └────────────┘  │
│                                                         │
│  Monitoring: CloudWatch Logs, Metrics, Alarms         │
│  Notifications: SNS Topics for events                 │
│  Secrets: AWS Secrets Manager for credentials         │
└─────────────────────────────────────────────────────────┘
```

## Directory Structure

```
k8s/terraform/
├── main.tf                      # Module instantiation
├── provider.tf                  # AWS provider config
├── backend.tf                   # S3 backend for state
├── variables.tf                 # Variable definitions
├── outputs.tf                   # Output definitions
├── terraform.tfvars.dev         # Dev environment config
├── terraform.tfvars.staging     # Staging environment config
├── terraform.tfvars.prod        # Production environment config
├── README.md                    # This file
├── DEPLOYMENT.md                # Deployment instructions
├── modules/
│   ├── vpc/
│   │   ├── main.tf              # VPC resources
│   │   ├── variables.tf         # VPC variables
│   │   └── outputs.tf           # VPC outputs
│   ├── eks/
│   │   ├── main.tf              # EKS cluster and nodes
│   │   ├── variables.tf         # EKS variables
│   │   └── outputs.tf           # EKS outputs
│   ├── rds/
│   │   ├── main.tf              # RDS instance
│   │   ├── variables.tf         # RDS variables
│   │   └── outputs.tf           # RDS outputs
│   └── elasticache/
│       ├── main.tf              # ElastiCache cluster
│       ├── variables.tf         # ElastiCache variables
│       └── outputs.tf           # ElastiCache outputs
```

## Prerequisites

### Required Tools

- **Terraform** >= 1.0
- **AWS CLI** >= 2.0
- **kubectl** >= 1.28
- **aws-iam-authenticator**

### AWS Setup

1. **AWS Account**: Active AWS account with appropriate permissions
2. **S3 Bucket**: For Terraform state (auto-created with setup script)
3. **DynamoDB Table**: For state locking (auto-created with setup script)
4. **IAM Permissions**: Administrator or equivalent permissions

### Environment Setup

```bash
# Install Terraform
brew install terraform  # macOS
# or download from https://www.terraform.io/downloads

# Install AWS CLI
pip install awscli

# Install kubectl
brew install kubectl  # macOS

# Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
```

## Module Composition

### VPC Module (k8s/terraform/modules/vpc/)

**Purpose**: Network infrastructure foundation

**Resources Created**:
- VPC with configurable CIDR (default: 10.0.0.0/16)
- 2 public subnets (one per AZ) with Internet Gateway
- 2 private subnets (one per AZ) with NAT Gateway
- Route tables and associations
- VPC Flow Logs to CloudWatch
- Security groups for EKS, RDS, ElastiCache

**Usage**:
```hcl
module "vpc" {
  source = "./modules/vpc"

  environment        = "prod"
  vpc_cidr          = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b"]
  enable_nat_gateway = true
  enable_flow_logs   = true
}
```

**Key Outputs**:
- `vpc_id`: VPC identifier
- `public_subnet_ids`: For load balancers
- `private_subnet_ids`: For EKS nodes
- `security_group_ids`: For service access

### EKS Module (k8s/terraform/modules/eks/)

**Purpose**: Kubernetes cluster management

**Resources Created**:
- EKS cluster with configurable version
- Managed node groups with auto-scaling
- OIDC provider for IRSA (IAM roles for service accounts)
- EKS add-ons (vpc-cni, coredns, kube-proxy, ebs-csi)
- CloudWatch logging
- IAM roles and policies

**Usage**:
```hcl
module "eks" {
  source = "./modules/eks"

  cluster_name          = "nsi-prod-cluster"
  kubernetes_version    = "1.28"
  instance_types        = ["m5.large"]
  desired_capacity      = 5
  min_capacity          = 3
  max_capacity          = 20
}
```

**Key Outputs**:
- `cluster_endpoint`: Kubernetes API endpoint
- `cluster_name`: Cluster identifier
- `oidc_provider_arn`: For IRSA configuration
- `kubeconfig`: Kubeconfig credentials

### RDS Module (k8s/terraform/modules/rds/)

**Purpose**: Managed PostgreSQL database

**Resources Created**:
- RDS PostgreSQL instance (multi-AZ production)
- Encryption at rest with AWS KMS
- Automated backups with configurable retention
- Enhanced monitoring
- RDS Proxy for connection pooling
- Parameter groups for performance tuning
- Secrets Manager for password storage

**Usage**:
```hcl
module "rds" {
  source = "./modules/rds"

  db_name            = "nsi-prod-db"
  instance_class     = "db.r5.large"
  allocated_storage  = 500
  multi_az           = true
  backup_retention   = 30
}
```

**Key Outputs**:
- `db_instance_endpoint`: Database connection endpoint
- `db_secret_arn`: Password secret in Secrets Manager
- `rds_proxy_endpoint`: Connection pooling endpoint
- `connection_string`: Pre-formatted connection string

### ElastiCache Module (k8s/terraform/modules/elasticache/)

**Purpose**: Redis caching layer

**Resources Created**:
- Redis replication group (cluster mode disabled)
- Multi-AZ with automatic failover
- Encryption at rest and in transit
- Parameter groups for optimization
- CloudWatch metrics and alarms
- Slow log and engine log streaming

**Usage**:
```hcl
module "elasticache" {
  source = "./modules/elasticache"

  cluster_name           = "nsi-prod-cache"
  node_type             = "cache.r6g.xlarge"
  num_cache_nodes       = 6
  automatic_failover    = true
  multi_az_enabled      = true
}
```

**Key Outputs**:
- `cluster_endpoint`: Primary endpoint for writes
- `cluster_reader_endpoint`: Reader endpoint for reads
- `auth_token_secret_arn`: AUTH token in Secrets Manager

## Deployment Guide

### Step 1: Initialize Backend

```bash
# Create S3 bucket for state
aws s3api create-bucket \
  --bucket nsi-terraform-state \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket nsi-terraform-state \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket nsi-terraform-state \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Create DynamoDB table for locking
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

### Step 2: Initialize Terraform

```bash
cd k8s/terraform

# Initialize Terraform
terraform init

# Verify modules
terraform get -update
```

### Step 3: Configure Environment

```bash
# For development
export TF_VARS_FILE="terraform.tfvars.dev"

# For staging
export TF_VARS_FILE="terraform.tfvars.staging"

# For production
export TF_VARS_FILE="terraform.tfvars.prod"
```

### Step 4: Plan Deployment

```bash
# Create plan for review
terraform plan -var-file="$TF_VARS_FILE" -out=tfplan

# Show plan details
terraform show tfplan
```

### Step 5: Apply Configuration

```bash
# Apply the plan
terraform apply tfplan

# Or apply with auto-approve (use with caution)
terraform apply -var-file="$TF_VARS_FILE" -auto-approve
```

### Step 6: Configure kubectl

```bash
# Get cluster name from outputs
CLUSTER_NAME=$(terraform output -raw eks_cluster_name)
REGION="us-east-1"

# Update kubeconfig
aws eks update-kubeconfig \
  --name $CLUSTER_NAME \
  --region $REGION

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

## Configuration

### Environment Variables

Each environment has a dedicated tfvars file:

**Development** (`terraform.tfvars.dev`):
- Minimal resources for cost efficiency
- Single node EKS cluster
- Micro RDS instance
- No multi-AZ, no monitoring

**Staging** (`terraform.tfvars.staging`):
- Moderate resources for testing
- 3-node EKS cluster
- Small RDS instance
- Multi-AZ enabled

**Production** (`terraform.tfvars.prod`):
- Full resources for high availability
- 5-20 node EKS cluster (auto-scaling)
- Large RDS instance (r5.large)
- Full monitoring and logging

### Custom Configuration

To customize for your needs, edit the relevant tfvars file:

```hcl
# Example: Increase EKS capacity
desired_capacity = 10
max_capacity     = 30

# Example: Change database size
rds_instance_class = "db.r5.2xlarge"
allocated_storage  = 1000

# Example: Disable features for cost savings
enable_performance_insights = false
cache_multi_az = false
```

## Security Best Practices

### Implemented Security Features

1. **Network Isolation**:
   - Private subnets for compute and database
   - Security groups restricting traffic
   - VPC Flow Logs for audit

2. **Encryption**:
   - RDS: At-rest with AWS KMS
   - Redis: At-rest and in-transit
   - S3 state: Server-side encryption

3. **Access Control**:
   - IAM roles for services (IRSA)
   - Secrets Manager for credentials
   - Private RDS endpoint

4. **Monitoring**:
   - CloudWatch logs from all services
   - Performance metrics and alarms
   - VPC Flow Logs to CloudWatch

### Additional Hardening

```hcl
# Restrict public API access (production)
endpoint_public_access = false
public_access_cidrs = ["10.0.0.0/8"]

# Enable advanced monitoring
monitoring_interval = 60
enable_performance_insights = true

# Enforce encryption
cache_transit_encryption = true
```

## Cost Optimization

### Estimated Monthly Costs (rough)

**Development**:
- EKS: $73 (cluster) + $15 (t3.micro nodes)
- RDS: $30 (db.t3.micro)
- ElastiCache: $15 (cache.t3.micro)
- **Total: ~$133/month**

**Staging**:
- EKS: $73 (cluster) + $90 (t3.large nodes)
- RDS: $80 (db.t3.small, multi-AZ)
- ElastiCache: $60 (cache.t3.small, 3-node)
- **Total: ~$303/month**

**Production**:
- EKS: $73 (cluster) + $500+ (m5.large nodes)
- RDS: $400+ (db.r5.large, multi-AZ)
- ElastiCache: $300+ (cache.r6g.xlarge, 6-node)
- **Total: ~$1,273+/month**

### Cost Reduction Strategies

1. **Use Spot Instances**: Change `capacity_type = "SPOT"` for 70% savings (non-production)
2. **Downsize Resources**: Use smaller instance types for development
3. **Reserved Instances**: Commit to 1-3 year terms for 30-50% discount
4. **Scheduled Scaling**: Stop resources during off-hours

## Module Dependency Graph

```
main.tf
  │
  ├── provider.tf (AWS provider)
  ├── backend.tf (S3 state)
  │
  └── modules/
      ├── vpc/ (foundation - no dependencies)
      ├── eks/ (depends on vpc)
      ├── rds/ (depends on vpc)
      └── elasticache/ (depends on vpc)
```

## Troubleshooting

### Common Issues

**1. Backend initialization fails**
```bash
# Solution: Verify S3 bucket exists and is accessible
aws s3 ls s3://nsi-terraform-state/
```

**2. EKS cluster creation timeout**
```bash
# Solution: Check IAM permissions and VPC capacity
aws ec2 describe-vpcs
aws iam get-user
```

**3. RDS connection fails**
```bash
# Solution: Verify security groups allow traffic
aws ec2 describe-security-groups --filter Name=group-id,Values=sg-xxxx
```

**4. kubectl cannot connect to cluster**
```bash
# Solution: Update kubeconfig
aws eks update-kubeconfig --name <cluster-name> --region us-east-1
```

## Maintenance

### Regular Tasks

**Weekly**:
- Review CloudWatch alarms
- Check resource utilization
- Monitor costs

**Monthly**:
- Review security group rules
- Update Kubernetes version
- Rotate credentials

**Quarterly**:
- Capacity planning
- Performance optimization
- Disaster recovery test

### Backup and Recovery

```bash
# Export state for backup
terraform state pull > terraform.tfstate.backup

# List resources
terraform state list

# Refresh state
terraform refresh

# Destroy resources (careful!)
terraform destroy -var-file="terraform.tfvars.prod"
```

## Production Readiness Assessment

### ✅ Completed

- [x] Multi-AZ VPC with redundancy
- [x] EKS cluster with managed nodes
- [x] RDS with automated backups
- [x] ElastiCache with failover
- [x] Encryption at rest and in transit
- [x] CloudWatch monitoring
- [x] IAM least privilege
- [x] Secrets Manager integration
- [x] VPC Flow Logs
- [x] SNS notifications

### 🔄 Recommended Next Steps

- [ ] Deploy Prometheus/Grafana for enhanced monitoring
- [ ] Set up CloudTrail for audit logging
- [ ] Configure AWS Backup for centralized backups
- [ ] Implement AWS Systems Manager Session Manager
- [ ] Deploy AWS WAF for API Gateway
- [ ] Configure VPC endpoints for AWS services
- [ ] Set up AWS Config for compliance

### 📋 Deployment Checklist

- [ ] AWS account and credentials configured
- [ ] S3 bucket and DynamoDB table created
- [ ] Terraform initialized with `terraform init`
- [ ] tfvars file selected for environment
- [ ] Plan reviewed with `terraform plan`
- [ ] Applied with `terraform apply`
- [ ] Outputs verified and documented
- [ ] kubectl configured and tested
- [ ] Monitoring dashboards created
- [ ] Runbooks documented
- [ ] Team trained on operations

## Outputs

After successful deployment, Terraform outputs important values:

```bash
# View all outputs
terraform output

# Get specific outputs
terraform output eks_cluster_endpoint
terraform output rds_endpoint
terraform output elasticache_endpoint

# Export outputs for scripts
export CLUSTER_NAME=$(terraform output -raw eks_cluster_name)
export DB_ENDPOINT=$(terraform output -raw rds_address)
export CACHE_ENDPOINT=$(terraform output -raw elasticache_endpoint)
```

## Support and Documentation

- **Terraform Docs**: https://registry.terraform.io/providers/hashicorp/aws/latest/docs
- **AWS EKS**: https://aws.amazon.com/eks/
- **AWS RDS**: https://aws.amazon.com/rds/
- **AWS ElastiCache**: https://aws.amazon.com/elasticache/

---

**Created**: December 14, 2025
**Version**: 1.0
**Maintainer**: DevOps Team
