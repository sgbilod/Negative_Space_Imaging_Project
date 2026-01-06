# RDS Module - Variables

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "db_name" {
  description = "Name for the RDS instance"
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*[a-z0-9]$", var.db_name))
    error_message = "DB name must start and end with alphanumeric, contain only lowercase alphanumeric and hyphens."
  }
}

variable "database_name" {
  description = "Name of the initial database"
  type        = string
  default     = "imaging_db"
}

variable "master_username" {
  description = "Master username for the database"
  type        = string
  default     = "postgres"
  sensitive   = true
}

variable "master_password" {
  description = "Master password for the database"
  type        = string
  sensitive   = true
  validation {
    condition     = length(var.master_password) >= 8
    error_message = "Master password must be at least 8 characters."
  }
}

variable "instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "14.8"
}

variable "allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 100
  validation {
    condition     = var.allocated_storage >= 20 && var.allocated_storage <= 65536
    error_message = "Allocated storage must be between 20 and 65536 GB."
  }
}

variable "storage_type" {
  description = "Storage type (gp3, gp2, or io1)"
  type        = string
  default     = "gp3"
  validation {
    condition     = contains(["gp3", "gp2", "io1"], var.storage_type)
    error_message = "Storage type must be gp3, gp2, or io1."
  }
}

variable "iops" {
  description = "IOPS for storage (required for io1, optional for gp3)"
  type        = number
  default     = 3000
}

variable "multi_az" {
  description = "Enable multi-AZ deployment"
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
  validation {
    condition     = var.backup_retention_period >= 1 && var.backup_retention_period <= 35
    error_message = "Backup retention must be between 1 and 35 days."
  }
}

variable "backup_window" {
  description = "Backup window (UTC)"
  type        = string
  default     = "03:00-04:00"
}

variable "maintenance_window" {
  description = "Maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = true
}

variable "skip_final_snapshot" {
  description = "Skip final snapshot on deletion (dev only)"
  type        = bool
  default     = false
}

variable "subnet_ids" {
  description = "List of subnet IDs for RDS subnet group"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group ID for RDS instance"
  type        = string
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval (0, 1, 5, 10, 15, 30, 60)"
  type        = number
  default     = 60
}

variable "enable_performance_insights" {
  description = "Enable Performance Insights"
  type        = bool
  default     = true
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention period in days"
  type        = number
  default     = 7
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "max_connections" {
  description = "Max connections parameter"
  type        = number
  default     = 200
}

variable "shared_buffers" {
  description = "Shared buffers parameter"
  type        = string
  default     = "{DBInstanceClassMemory/32768}"
}

variable "effective_cache_size" {
  description = "Effective cache size parameter"
  type        = string
  default     = "{DBInstanceClassMemory*3/4}"
}

variable "work_mem" {
  description = "Work memory parameter"
  type        = string
  default     = "16384"
}

variable "maintenance_work_mem" {
  description = "Maintenance work memory parameter"
  type        = string
  default     = "65536"
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for RDS events"
  type        = string
}

variable "enable_proxy" {
  description = "Enable RDS Proxy for connection pooling"
  type        = bool
  default     = true
}

variable "proxy_max_connections" {
  description = "Maximum connections for RDS Proxy"
  type        = number
  default     = 100
}

variable "proxy_max_idle_connections" {
  description = "Maximum idle connections for RDS Proxy"
  type        = number
  default     = 50
}

variable "proxy_connection_borrow_timeout" {
  description = "Connection borrow timeout in seconds"
  type        = number
  default     = 120
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "NegativeSpaceImaging"
    ManagedBy = "Terraform"
  }
}
