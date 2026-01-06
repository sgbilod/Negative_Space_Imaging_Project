# Main Terraform Variables

# AWS Configuration
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "flow_logs_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# EKS Configuration
variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "instance_types" {
  description = "EKS worker node instance types"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "desired_capacity" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

variable "min_capacity" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "capacity_type" {
  description = "EKS node group capacity type (ON_DEMAND or SPOT)"
  type        = string
  default     = "ON_DEMAND"
}

variable "disk_size" {
  description = "EKS worker node disk size in GB"
  type        = number
  default     = 50
}

variable "endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "public_access_cidrs" {
  description = "CIDR blocks allowed for public endpoint access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# RDS Configuration
variable "db_name" {
  description = "RDS instance name"
  type        = string
  default     = "nsi-imaging-db"
}

variable "database_name" {
  description = "Initial database name"
  type        = string
  default     = "imaging_db"
}

variable "master_username" {
  description = "RDS master username"
  type        = string
  default     = "postgres"
  sensitive   = true
}

variable "master_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "14.8"
}

variable "allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 100
}

variable "storage_type" {
  description = "Storage type"
  type        = string
  default     = "gp3"
}

variable "iops" {
  description = "IOPS for storage"
  type        = number
  default     = 3000
}

variable "multi_az" {
  description = "Enable multi-AZ RDS"
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "RDS backup retention in days"
  type        = number
  default     = 30
}

variable "backup_window" {
  description = "RDS backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "maintenance_window" {
  description = "RDS maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "deletion_protection" {
  description = "Enable RDS deletion protection"
  type        = bool
  default     = true
}

variable "skip_final_snapshot" {
  description = "Skip final snapshot (dev only)"
  type        = bool
  default     = false
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval"
  type        = number
  default     = 60
}

variable "enable_performance_insights" {
  description = "Enable Performance Insights"
  type        = bool
  default     = true
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention in days"
  type        = number
  default     = 7
}

variable "enable_proxy" {
  description = "Enable RDS Proxy"
  type        = bool
  default     = true
}

# ElastiCache Configuration
variable "cache_cluster_name" {
  description = "ElastiCache cluster name"
  type        = string
  default     = "nsi-cache"
}

variable "cache_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "cache_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 3
}

variable "cache_port" {
  description = "Redis port"
  type        = number
  default     = 6379
}

variable "cache_automatic_failover" {
  description = "Enable automatic failover"
  type        = bool
  default     = true
}

variable "cache_multi_az" {
  description = "Enable multi-AZ"
  type        = bool
  default     = true
}

variable "cache_transit_encryption" {
  description = "Enable transit encryption"
  type        = bool
  default     = true
}

variable "cache_auth_token" {
  description = "Redis auth token"
  type        = string
  default     = ""
  sensitive   = true
}

# Logging Configuration
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Common Tags
variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "NegativeSpaceImaging"
    ManagedBy   = "Terraform"
    CostCenter  = "Engineering"
  }
}
