# Terraform Infrastructure Index

## 📑 Quick Navigation

### Getting Started (Read First)
1. **[README.md](README.md)** - Architecture overview and module documentation
2. **[DEPLOYMENT.md](DEPLOYMENT.md)** - Step-by-step deployment guide (10 phases)
3. **[COST_ANALYSIS.md](COST_ANALYSIS.md)** - Cost breakdown and optimization

### Infrastructure Configuration
- **VPC Module** - [modules/vpc/](modules/vpc/README.md)
- **EKS Module** - [modules/eks/](modules/eks/README.md)
- **RDS Module** - [modules/rds/](modules/rds/README.md)
- **ElastiCache Module** - [modules/elasticache/](modules/elasticache/README.md)

### Configuration Files
- **Development** - [terraform.tfvars.dev](terraform.tfvars.dev)
- **Staging** - [terraform.tfvars.staging](terraform.tfvars.staging)
- **Production** - [terraform.tfvars.prod](terraform.tfvars.prod)

### Root Module
- **Provider Config** - [provider.tf](provider.tf)
- **Backend Config** - [backend.tf](backend.tf)
- **Main Config** - [main.tf](main.tf)
- **Variables** - [variables.tf](variables.tf)
- **Outputs** - [outputs.tf](outputs.tf)

---

## 🚀 Quick Start Commands

### Initialize Terraform
```bash
cd k8s/terraform
terraform init
```

### Validate Configuration
```bash
terraform validate
terraform fmt -recursive
```

### Deploy to Development
```bash
terraform plan -var-file="terraform.tfvars.dev" -out=tfplan
terraform apply tfplan
```

### Deploy to Production
```bash
terraform plan -var-file="terraform.tfvars.prod" -out=tfplan
terraform apply tfplan
```

### Cleanup
```bash
terraform destroy -var-file="terraform.tfvars.dev"
```

---

## 📊 Infrastructure Overview

### Services Deployed

| Service | Purpose | Configuration | Status |
|---------|---------|---|--------|
| **VPC** | Network foundation | Multi-AZ, public/private subnets | ✅ Ready |
| **EKS** | Kubernetes cluster | Managed nodes, auto-scaling | ✅ Ready |
| **RDS** | PostgreSQL database | Multi-AZ, encrypted, backups | ✅ Ready |
| **ElastiCache** | Redis cache | Multi-AZ, high-availability | ✅ Ready |

### Environment Options

| Environment | Cluster | Database | Cache | Est. Cost |
|------------|---------|----------|-------|-----------|
| **Dev** | 2 × t3.medium | db.t3.micro | 1 × cache.t3.micro | ~$133/month |
| **Staging** | 3 × t3.large | db.t3.small | 3 × cache.t3.small | ~$303/month |
| **Production** | 5-20 × m5.large | db.r5.large | 6 × cache.r6g.xlarge | ~$1,273+/month |

---

## 🔐 Security Features

- ✅ **Encryption**: KMS at-rest, TLS in-transit
- ✅ **Network**: VPC isolation, security groups, Flow Logs
- ✅ **IAM**: Least-privilege roles, IRSA for K8s
- ✅ **Secrets**: Secrets Manager for passwords/tokens
- ✅ **Audit**: CloudWatch logs, VPC Flow Logs
- ✅ **Monitoring**: CloudWatch metrics and alarms

---

## 📈 Module Statistics

| Module | Files | Lines | Features |
|--------|-------|-------|----------|
| VPC | 3 | 445 | Multi-AZ, NAT, Flow Logs |
| EKS | 3 | 670 | Kubernetes, OIDC, add-ons |
| RDS | 3 | 630 | PostgreSQL, encryption, Proxy |
| ElastiCache | 3 | 435 | Redis, HA, alarms |
| Main | 5 | 452 | Orchestration, SNS |
| Config | 3 | 165 | Dev/staging/prod |
| **Total** | **20** | **2,797** | **Production-ready** |

---

## 💰 Cost Management

### Optimization Strategies

1. **Reserved Instances** - Save 30% with 1-year commitment
2. **Spot Instances** - Save 70% for non-critical workloads
3. **Scheduled Scaling** - Save 40% for non-24/7 operations
4. **Right-Sizing** - Save 20% with appropriate instance types

**Potential Savings**: $200-400/month (25-40% reduction)

See [COST_ANALYSIS.md](COST_ANALYSIS.md) for detailed breakdown.

---

## 🎯 Key Features

### High Availability
- Multi-AZ deployment across all services
- Automatic failover for databases and caches
- Auto-scaling for compute resources
- Health checks and self-healing

### Observability
- Centralized CloudWatch logging
- CloudWatch metrics and alarms
- SNS notifications for events
- VPC Flow Logs for network audit

### Security
- Encryption for all data
- Network isolation with security groups
- IAM least-privilege access
- Secrets Manager for credential storage

### Scalability
- Kubernetes auto-scaling (HPA, VPA)
- EKS cluster auto-scaling
- RDS read replicas
- ElastiCache cluster scaling

---

## 📖 Documentation

### Architecture
- Module dependency diagram
- Network topology
- Data flow diagrams
- Security architecture

### Operations
- Deployment procedures
- Scaling procedures
- Backup/restore procedures
- Troubleshooting guides

### Maintenance
- Upgrade procedures
- Patch management
- Security updates
- Performance optimization

---

## 🔍 Files Overview

### Root Level (`k8s/terraform/`)

**Configuration Files:**
- `provider.tf` - AWS provider with default tags
- `backend.tf` - S3 backend with DynamoDB locking
- `main.tf` - Module instantiation and data sources
- `variables.tf` - Root module variables (220 lines)
- `outputs.tf` - Aggregated outputs from modules

**Environment Files:**
- `terraform.tfvars.dev` - Development configuration
- `terraform.tfvars.staging` - Staging configuration
- `terraform.tfvars.prod` - Production configuration

**Documentation:**
- `README.md` - Architecture and deployment guide
- `DEPLOYMENT.md` - 10-phase deployment guide
- `COST_ANALYSIS.md` - Cost breakdown and optimization

### VPC Module (`k8s/terraform/modules/vpc/`)

- `main.tf` - VPC, subnets, NAT, IGW, route tables, security groups, Flow Logs
- `variables.tf` - VPC configuration variables
- `outputs.tf` - VPC resource identifiers and IPs

**Key Outputs:**
- VPC ID
- Subnet IDs (public and private)
- Security group IDs
- NAT gateway IPs

### EKS Module (`k8s/terraform/modules/eks/`)

- `main.tf` - EKS cluster, node groups, IAM, OIDC, add-ons
- `variables.tf` - Kubernetes and cluster configuration
- `outputs.tf` - Cluster endpoint, kubeconfig, OIDC provider

**Key Outputs:**
- Cluster endpoint
- Cluster ARN
- OIDC provider ARN
- Node group IAM role ARN

### RDS Module (`k8s/terraform/modules/rds/`)

- `main.tf` - RDS instance, KMS, parameter groups, Proxy, Secrets Manager
- `variables.tf` - Database configuration and instance settings
- `outputs.tf` - Connection strings, endpoints, secret ARNs

**Key Outputs:**
- Database endpoint
- Database connection string
- Proxy endpoint (if enabled)
- Secret manager ARN

### ElastiCache Module (`k8s/terraform/modules/elasticache/`)

- `main.tf` - Redis cluster, parameter groups, logs, alarms, events
- `variables.tf` - Cache configuration and node settings
- `outputs.tf` - Connection endpoints and secret ARNs

**Key Outputs:**
- Primary endpoint
- Reader endpoint
- Replication group ARN
- Auth token secret ARN

---

## ✅ Validation Checklist

Before deploying, ensure:

- [ ] AWS credentials configured (`aws sts get-caller-identity`)
- [ ] Terraform installed (`terraform --version`)
- [ ] kubectl installed (`kubectl version --client`)
- [ ] S3 bucket created for state (`nsi-terraform-state`)
- [ ] DynamoDB table created for locking (`terraform-locks`)
- [ ] Environment selected (dev/staging/prod)
- [ ] Configuration reviewed (`cat terraform.tfvars.dev`)
- [ ] Syntax validated (`terraform validate`)
- [ ] Plan reviewed (`terraform plan`)

---

## 🚨 Troubleshooting

### Common Issues

**Issue**: `InvalidInput.Duplicate` error on S3 bucket creation
- **Solution**: S3 bucket names must be globally unique. Modify `backend.tf` with unique name.

**Issue**: EKS cluster creation fails
- **Solution**: Check CloudFormation stack status. Verify IAM permissions.

**Issue**: RDS fails to create
- **Solution**: Verify db.t3.micro availability in selected region. Check storage constraints.

**Issue**: Cannot connect to kubectl
- **Solution**: Run `aws eks update-kubeconfig --name nsi-{env}-cluster`

See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting) for comprehensive troubleshooting guide.

---

## 📞 Support

### Getting Help

1. **Review Documentation**
   - README.md for architecture overview
   - DEPLOYMENT.md for step-by-step guide
   - Each module's variables.tf for configuration options

2. **Check Logs**
   - AWS CloudFormation events
   - CloudWatch Logs
   - Terraform debug logs (`TF_LOG=DEBUG`)

3. **Verify Setup**
   - AWS credentials working
   - Required S3/DynamoDB resources created
   - Security group rules correct
   - IAM permissions sufficient

4. **Consult Troubleshooting**
   - DEPLOYMENT.md troubleshooting section
   - Module-specific issues in README.md

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Read README.md
- [ ] Review DEPLOYMENT.md
- [ ] Understand COST_ANALYSIS.md
- [ ] Select environment (dev/staging/prod)
- [ ] Configure AWS credentials

### Deployment
- [ ] Run terraform init
- [ ] Run terraform validate
- [ ] Review terraform plan
- [ ] Run terraform apply
- [ ] Monitor progress

### Post-Deployment
- [ ] Update kubeconfig
- [ ] Verify kubectl access
- [ ] Test database connectivity
- [ ] Test cache connectivity
- [ ] Configure monitoring

### Validation
- [ ] All AWS resources created
- [ ] Kubernetes nodes healthy
- [ ] Database accessible
- [ ] Cache accessible
- [ ] Logs streaming to CloudWatch

---

## 🔄 Common Operations

### Scale EKS Cluster

```bash
# Edit terraform.tfvars
desired_capacity = 10  # Increase from 5

# Apply changes
terraform apply -var-file="terraform.tfvars.prod"
```

### Upgrade Kubernetes Version

```bash
# Edit terraform.tfvars
kubernetes_version = "1.29"

# Plan and apply
terraform plan -var-file="terraform.tfvars.prod"
terraform apply -var-file="terraform.tfvars.prod"
```

### Increase Database Storage

```bash
# Edit terraform.tfvars
allocated_storage = 500  # Increase from 200

# Apply changes
terraform apply -var-file="terraform.tfvars.prod"
```

### Add Cache Nodes

```bash
# Edit terraform.tfvars
num_cache_nodes = 10  # Increase from 6

# Apply changes
terraform apply -var-file="terraform.tfvars.prod"
```

---

## 📚 Learning Resources

- [Terraform Documentation](https://www.terraform.io/docs)
- [AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [EKS Best Practices Guide](https://aws.github.io/aws-eks-best-practices/)
- [RDS Best Practices](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_BestPractices.html)

---

## 🎉 You're All Set!

The infrastructure is production-ready and waiting for deployment.

**Next Step**: Read [DEPLOYMENT.md](DEPLOYMENT.md) and begin deployment.

---

**Last Updated**: December 14, 2025
**Terraform Version**: >= 1.0
**AWS Provider**: ~> 5.0
**Status**: ✅ Production Ready
