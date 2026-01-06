# Main Terraform Outputs

# VPC Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "nat_gateway_ips" {
  description = "NAT Gateway public IPs"
  value       = module.vpc.nat_gateway_ips
}

# EKS Outputs
output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_version" {
  description = "EKS cluster Kubernetes version"
  value       = module.eks.cluster_version
}

output "eks_cluster_certificate_authority_data" {
  description = "EKS cluster certificate authority data"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "eks_oidc_provider_arn" {
  description = "EKS OIDC provider ARN (for IRSA)"
  value       = module.eks.oidc_provider_arn
}

output "eks_kubeconfig_command" {
  description = "Command to update kubeconfig"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.aws_region}"
}

# RDS Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
}

output "rds_address" {
  description = "RDS instance address"
  value       = module.rds.db_instance_address
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.rds.db_instance_port
}

output "rds_username" {
  description = "RDS master username"
  value       = module.rds.db_instance_username
  sensitive   = true
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.db_name
}

output "rds_secret_arn" {
  description = "RDS password secret ARN"
  value       = module.rds.db_secret_arn
}

output "rds_proxy_endpoint" {
  description = "RDS Proxy endpoint (if enabled)"
  value       = module.rds.rds_proxy_endpoint
}

output "rds_connection_string" {
  description = "RDS connection string (without password)"
  value       = module.rds.connection_string
  sensitive   = true
}

# ElastiCache Outputs
output "elasticache_endpoint" {
  description = "ElastiCache primary endpoint"
  value       = module.elasticache.cluster_endpoint
}

output "elasticache_reader_endpoint" {
  description = "ElastiCache reader endpoint"
  value       = module.elasticache.cluster_reader_endpoint
}

output "elasticache_replication_group_id" {
  description = "ElastiCache replication group ID"
  value       = module.elasticache.replication_group_id
}

output "elasticache_secret_arn" {
  description = "ElastiCache auth token secret ARN"
  value       = module.elasticache.auth_token_secret_arn
}

# SNS Topic
output "notifications_topic_arn" {
  description = "SNS topic ARN for notifications"
  value       = aws_sns_topic.notifications.arn
}

# Summary Output
output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    environment = var.environment
    region      = var.aws_region
    vpc_id      = module.vpc.vpc_id
    eks_cluster = {
      name     = module.eks.cluster_name
      endpoint = module.eks.cluster_endpoint
      version  = module.eks.cluster_version
    }
    rds_database = {
      endpoint = module.rds.db_instance_endpoint
      engine   = "PostgreSQL"
      address  = module.rds.db_instance_address
    }
    elasticache = {
      endpoint = module.elasticache.cluster_endpoint
      engine   = "Redis"
    }
  }
}
