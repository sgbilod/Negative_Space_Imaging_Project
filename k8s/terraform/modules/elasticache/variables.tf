# ElastiCache Module - Variables

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "cluster_name" {
  description = "Name for the ElastiCache cluster"
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*[a-z0-9]$", var.cluster_name))
    error_message = "Cluster name must start and end with alphanumeric, contain only lowercase alphanumeric and hyphens."
  }
}

variable "engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "num_cache_nodes" {
  description = "Number of cache nodes in the cluster"
  type        = number
  default     = 3
  validation {
    condition     = var.num_cache_nodes >= 1 && var.num_cache_nodes <= 6
    error_message = "Number of cache nodes must be between 1 and 6."
  }
}

variable "port" {
  description = "Redis port"
  type        = number
  default     = 6379
  validation {
    condition     = var.port >= 1024 && var.port <= 65535
    error_message = "Port must be between 1024 and 65535."
  }
}

variable "automatic_failover_enabled" {
  description = "Enable automatic failover"
  type        = bool
  default     = true
}

variable "multi_az_enabled" {
  description = "Enable multi-AZ"
  type        = bool
  default     = true
}

variable "transit_encryption_enabled" {
  description = "Enable transit encryption (TLS)"
  type        = bool
  default     = true
}

variable "auth_token" {
  description = "Redis AUTH token for encryption in transit"
  type        = string
  default     = ""
  sensitive   = true
  validation {
    condition     = length(var.auth_token) == 0 || (length(var.auth_token) >= 16 && length(var.auth_token) <= 128)
    error_message = "Auth token must be empty or between 16 and 128 characters."
  }
}

variable "subnet_ids" {
  description = "List of subnet IDs for ElastiCache subnet group"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group ID for ElastiCache cluster"
  type        = string
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for ElastiCache events and alarms"
  type        = string
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "NegativeSpaceImaging"
    ManagedBy = "Terraform"
  }
}
