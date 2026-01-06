# Deployment Instructions - Negative Space Imaging Infrastructure

## Quick Start (5 minutes)

```bash
# 1. Navigate to terraform directory
cd k8s/terraform

# 2. Initialize Terraform
terraform init

# 3. Plan deployment (dev environment)
terraform plan -var-file="terraform.tfvars.dev" -out=tfplan

# 4. Apply configuration
terraform apply tfplan

# 5. Configure kubectl
aws eks update-kubeconfig --name $(terraform output -raw eks_cluster_name)

# 6. Verify cluster
kubectl cluster-info
kubectl get nodes
```

## Full Deployment Guide

### Phase 1: Prerequisites (10 minutes)

#### 1.1 Install Required Tools

```bash
# macOS
brew install terraform awscli kubectl

# Ubuntu/Debian
sudo apt-get install -y terraform awscli kubectl

# Windows (PowerShell as Administrator)
choco install terraform awscli kubernetes-cli

# Verify installations
terraform --version
aws --version
kubectl version --client
```

#### 1.2 Configure AWS Credentials

```bash
# Create AWS credentials file
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Verify credentials
aws sts get-caller-identity
```

#### 1.3 Create S3 Backend (One-time)

```bash
#!/bin/bash
set -e

BUCKET_NAME="nsi-terraform-state"
REGION="us-east-1"

echo "Creating S3 bucket for Terraform state..."
aws s3api create-bucket \
  --bucket $BUCKET_NAME \
  --region $REGION \
  --create-bucket-configuration LocationConstraint=$REGION 2>/dev/null || echo "Bucket already exists"

echo "Enabling versioning..."
aws s3api put-bucket-versioning \
  --bucket $BUCKET_NAME \
  --versioning-configuration Status=Enabled

echo "Enabling encryption..."
aws s3api put-bucket-encryption \
  --bucket $BUCKET_NAME \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

echo "Blocking public access..."
aws s3api put-public-access-block \
  --bucket $BUCKET_NAME \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

echo "Creating DynamoDB table for state locking..."
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region $REGION 2>/dev/null || echo "DynamoDB table already exists"

echo "Backend setup complete!"
```

### Phase 2: Environment Selection (2 minutes)

#### 2.1 Choose Target Environment

```bash
# For Development
export ENVIRONMENT="dev"
export TF_VARS_FILE="terraform.tfvars.dev"

# OR For Staging
export ENVIRONMENT="staging"
export TF_VARS_FILE="terraform.tfvars.staging"

# OR For Production
export ENVIRONMENT="prod"
export TF_VARS_FILE="terraform.tfvars.prod"

echo "Deploying to: $ENVIRONMENT"
```

#### 2.2 Customize Configuration (Optional)

Edit the selected tfvars file:

```bash
# Edit configuration
vim $TF_VARS_FILE

# Key variables to adjust:
# - cluster_name
# - instance_types
# - desired_capacity
# - rds_instance_class
# - allocated_storage
# - cache_node_type
```

### Phase 3: Validation (3 minutes)

#### 3.1 Validate Configuration

```bash
# Navigate to terraform directory
cd k8s/terraform

# Validate syntax
terraform fmt -recursive

# Validate configuration
terraform validate

# Check for errors
echo $?  # Should output 0 for success
```

#### 3.2 Preview Changes

```bash
# Generate and review plan
terraform plan \
  -var-file="$TF_VARS_FILE" \
  -out=tfplan \
  -no-color > plan.txt

# Review plan
cat plan.txt

# Show resource count
echo "Resources to create:"
grep "to be created" plan.txt
```

### Phase 4: Deployment (15-20 minutes)

#### 4.1 Create Infrastructure

```bash
# Apply Terraform configuration
echo "Deploying infrastructure to $ENVIRONMENT..."
terraform apply tfplan

# Monitor progress (watch logs in parallel terminal)
watch aws ec2 describe-instances --filters Name=tag:Environment,Values=$ENVIRONMENT

# Wait for completion
echo "Deployment in progress... This typically takes 15-20 minutes"
sleep 300  # Check after 5 minutes
```

#### 4.2 Verify Deployment

```bash
# Check Terraform outputs
terraform output

# Verify AWS resources
aws ec2 describe-vpcs --filters Name=tag:Environment,Values=$ENVIRONMENT
aws eks describe-cluster --name nsi-$ENVIRONMENT-cluster

# Verify RDS
aws rds describe-db-instances --query 'DBInstances[*].[DBInstanceIdentifier,DBInstanceStatus]'

# Verify ElastiCache
aws elasticache describe-replication-groups --query 'ReplicationGroups[*].[ReplicationGroupId,Status]'
```

### Phase 5: Kubernetes Configuration (5 minutes)

#### 5.1 Update kubeconfig

```bash
# Get cluster name
CLUSTER_NAME=$(terraform output -raw eks_cluster_name)
REGION="us-east-1"

# Update kubeconfig
aws eks update-kubeconfig \
  --name $CLUSTER_NAME \
  --region $REGION

# Verify kubeconfig
cat ~/.kube/config | grep -A 5 "name: $CLUSTER_NAME"
```

#### 5.2 Verify Cluster Access

```bash
# Test cluster connectivity
kubectl cluster-info

# View nodes
kubectl get nodes

# Get node details
kubectl describe nodes

# Expected output: Nodes in Ready state
```

#### 5.3 Deploy Kubernetes Dashboard (Optional)

```bash
# Install metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Install dashboard
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

# Create proxy
kubectl proxy &

# Access at: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

### Phase 6: Database Configuration (5 minutes)

#### 6.1 Get Database Credentials

```bash
# Get RDS endpoint
DB_ENDPOINT=$(terraform output -raw rds_address)
DB_PORT=$(terraform output -raw rds_port)
DB_NAME=$(terraform output -raw rds_database_name)

# Get secret from Secrets Manager
SECRET_NAME="nsi-$ENVIRONMENT-db/master-password"
DB_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id $SECRET_NAME \
  --query SecretString \
  --output text)

echo "RDS Connection String:"
echo "postgresql://postgres:$DB_PASSWORD@$DB_ENDPOINT:$DB_PORT/$DB_NAME"
```

#### 6.2 Test Database Connection

```bash
# Install PostgreSQL client
sudo apt-get install -y postgresql-client  # Ubuntu
brew install libpq  # macOS

# Test connection
psql -h $DB_ENDPOINT -U postgres -d $DB_NAME -c "SELECT version();"
```

#### 6.3 Initialize Database (if needed)

```bash
# Create application schema
psql -h $DB_ENDPOINT -U postgres -d $DB_NAME << EOF
CREATE SCHEMA imaging;
CREATE TABLE imaging.images (
  id SERIAL PRIMARY KEY,
  filename VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
GRANT USAGE ON SCHEMA imaging TO postgres;
GRANT ALL ON SCHEMA imaging TO postgres;
EOF
```

### Phase 7: Cache Configuration (3 minutes)

#### 7.1 Get Redis Credentials

```bash
# Get Redis endpoint
REDIS_ENDPOINT=$(terraform output -raw elasticache_endpoint)
REDIS_READER=$(terraform output -raw elasticache_reader_endpoint)

# Get auth token from Secrets Manager (if configured)
SECRET_NAME="nsi-$ENVIRONMENT-cache/auth-token"
REDIS_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id $SECRET_NAME \
  --query SecretString \
  --output text 2>/dev/null || echo "")

echo "Redis Primary: $REDIS_ENDPOINT"
echo "Redis Reader:  $REDIS_READER"
```

#### 7.2 Test Cache Connection

```bash
# Install Redis client
sudo apt-get install -y redis-tools  # Ubuntu
brew install redis  # macOS

# Test connection
redis-cli -h $(echo $REDIS_ENDPOINT | cut -d: -f1) -p 6379 PING

# Expected: PONG
```

### Phase 8: Monitoring Setup (5 minutes)

#### 8.1 Configure CloudWatch Dashboard

```bash
# Create dashboard (optional - AWS console)
# Navigate to: CloudWatch → Dashboards → Create Dashboard
# Add widgets for:
# - EKS cluster status
# - RDS CPU/Memory
# - ElastiCache evictions
# - Network throughput
```

#### 8.2 Enable Alarms

```bash
# List existing alarms
aws cloudwatch describe-alarms --query 'MetricAlarms[*].[AlarmName,StateValue]'

# Enable specific alarm
aws cloudwatch enable-alarm-actions --alarm-names nsi-$ENVIRONMENT-cpu-utilization-high
```

#### 8.3 Subscribe to Notifications

```bash
# Get SNS topic ARN
SNS_TOPIC=$(terraform output -raw notifications_topic_arn)

# Subscribe to notifications
aws sns subscribe \
  --topic-arn $SNS_TOPIC \
  --protocol email \
  --notification-endpoint your-email@example.com
```

### Phase 9: Validation & Testing (10 minutes)

#### 9.1 Comprehensive Health Check

```bash
#!/bin/bash
set -e

echo "=== Comprehensive Infrastructure Check ==="

# 1. Check VPC
echo "✓ VPC Status"
aws ec2 describe-vpcs --filters "Name=tag:Environment,Values=$ENVIRONMENT" --query 'Vpcs[0].State'

# 2. Check EKS
echo "✓ EKS Cluster"
aws eks describe-cluster --name nsi-$ENVIRONMENT-cluster --query 'cluster.status'
kubectl get nodes --no-headers

# 3. Check RDS
echo "✓ RDS Instance"
aws rds describe-db-instances --db-instance-identifier nsi-$ENVIRONMENT-db --query 'DBInstances[0].DBInstanceStatus'

# 4. Check ElastiCache
echo "✓ ElastiCache Cluster"
aws elasticache describe-replication-groups --replication-group-id nsi-$ENVIRONMENT-cache --query 'ReplicationGroups[0].Status'

# 5. Check Security
echo "✓ Security Groups"
aws ec2 describe-security-groups --filters "Name=tag:Environment,Values=$ENVIRONMENT" --query 'SecurityGroups[*].[GroupName,GroupId]'

echo "=== All Checks Complete ==="
```

#### 9.2 Load Testing (Optional)

```bash
# Deploy test pod
kubectl run -it --image=busybox test-pod -- sh

# Test database from pod
apt-get update && apt-get install -y postgresql-client
psql -h $DB_ENDPOINT -U postgres -d $DB_NAME -c "SELECT 1;"

# Test cache from pod
apt-get install -y redis-tools
redis-cli -h $(echo $REDIS_ENDPOINT | cut -d: -f1) PING

# Exit pod
exit
```

### Phase 10: Documentation (5 minutes)

#### 10.1 Record Outputs

```bash
# Save all outputs to file
terraform output > infrastructure-outputs.txt

# Create deployment record
cat > deployment-record.md << EOF
# Deployment Record

**Date**: $(date)
**Environment**: $ENVIRONMENT
**AWS Region**: us-east-1

## Infrastructure Summary

### Cluster
- **Name**: $(terraform output -raw eks_cluster_name)
- **Endpoint**: $(terraform output -raw eks_cluster_endpoint)

### Database
- **Endpoint**: $(terraform output -raw rds_address)
- **Database**: $(terraform output -raw rds_database_name)

### Cache
- **Endpoint**: $(terraform output -raw elasticache_endpoint)
- **Type**: Redis 7.0

## Deployment Status
- ✅ VPC Created
- ✅ EKS Cluster Active
- ✅ RDS Instance Running
- ✅ ElastiCache Cluster Running
- ✅ Security Groups Configured
- ✅ Monitoring Enabled

## Next Steps
1. Configure application deployments
2. Set up CI/CD pipeline
3. Configure backup policies
4. Schedule regular tests
EOF

cat deployment-record.md
```

#### 10.2 Create Runbooks

```bash
# Create operations runbook
cat > runbook.md << 'EOF'
# Operations Runbook

## Access Cluster
\`\`\`bash
aws eks update-kubeconfig --name nsi-prod-cluster
kubectl cluster-info
\`\`\`

## Check Node Status
\`\`\`bash
kubectl get nodes
kubectl describe nodes
\`\`\`

## Check Database Connection
\`\`\`bash
psql -h <db-endpoint> -U postgres -d imaging_db
\`\`\`

## Scale EKS Cluster
\`\`\`bash
# Modify desired_capacity in terraform.tfvars
terraform apply -var-file="terraform.tfvars.prod"
\`\`\`

## Backup Database
\`\`\`bash
aws rds create-db-snapshot --db-instance-identifier nsi-prod-db
\`\`\`
EOF

cat runbook.md
```

## Rollback Procedures

### Quick Rollback

```bash
# If deployment fails, destroy resources
terraform destroy -var-file="$TF_VARS_FILE" -auto-approve

# Or selectively destroy
terraform destroy -target=module.eks -var-file="$TF_VARS_FILE"
```

### Partial Recovery

```bash
# Refresh state
terraform refresh -var-file="$TF_VARS_FILE"

# Re-apply specific module
terraform apply -target=module.rds -var-file="$TF_VARS_FILE"
```

## Troubleshooting

### EKS Cluster Won't Come Up

```bash
# Check CloudFormation stack
aws cloudformation describe-stacks --query 'Stacks[?Tags[?Key==`Environment`]].StackStatus'

# Check service logs
aws logs tail /aws/eks/nsi-prod-cluster/cluster --follow

# Increase verbosity
TF_LOG=DEBUG terraform apply
```

### RDS Connection Timeout

```bash
# Verify security group
aws ec2 describe-security-groups --group-ids sg-xxxx

# Add EKS security group to RDS
aws ec2 authorize-security-group-ingress \
  --group-id sg-rds \
  --protocol tcp \
  --port 5432 \
  --source-group sg-eks
```

### Cache Evictions High

```bash
# Scale up cache
# Edit terraform.tfvars
cache_node_type = "cache.r6g.2xlarge"  # Larger instance
num_cache_nodes = 10  # More nodes

# Apply changes
terraform apply -var-file="$TF_VARS_FILE"
```

## Deployment Checklist

```
[ ] Prerequisites installed and configured
[ ] AWS credentials working
[ ] S3 backend created
[ ] DynamoDB table created
[ ] Environment selected (dev/staging/prod)
[ ] Configuration reviewed
[ ] terraform init completed
[ ] terraform plan reviewed
[ ] terraform apply completed
[ ] AWS resources verified
[ ] kubeconfig updated
[ ] Cluster connectivity tested
[ ] Database accessible
[ ] Cache accessible
[ ] Monitoring configured
[ ] Alarms enabled
[ ] Documentation complete
[ ] Team trained
```

---

**Deployment Time**: 30-45 minutes (including validation)
**Success Indicators**: All resources in ready state, kubectl access working
