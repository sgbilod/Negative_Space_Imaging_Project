# EKS Module - Variables

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  validation {
    condition     = length(var.cluster_name) <= 100 && can(regex("^[a-zA-Z][a-zA-Z0-9-]*$", var.cluster_name))
    error_message = "Cluster name must start with letter, contain only alphanumeric and hyphens, max 100 chars."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version to use for the EKS cluster"
  type        = string
  default     = "1.28"
}

variable "subnet_ids" {
  description = "List of subnet IDs for the EKS cluster"
  type        = list(string)
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for worker nodes"
  type        = list(string)
}

variable "eks_control_plane_security_group_id" {
  description = "Security group ID for EKS control plane"
  type        = string
}

variable "instance_types" {
  description = "List of instance types for worker nodes"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "desired_capacity" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
  validation {
    condition     = var.desired_capacity >= 1 && var.desired_capacity <= 100
    error_message = "Desired capacity must be between 1 and 100."
  }
}

variable "min_capacity" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
  validation {
    condition     = var.min_capacity >= 1
    error_message = "Min capacity must be at least 1."
  }
}

variable "max_capacity" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
  validation {
    condition     = var.max_capacity >= 1
    error_message = "Max capacity must be at least 1."
  }
}

variable "capacity_type" {
  description = "Type of capacity for node group (ON_DEMAND or SPOT)"
  type        = string
  default     = "ON_DEMAND"
  validation {
    condition     = contains(["ON_DEMAND", "SPOT"], var.capacity_type)
    error_message = "Capacity type must be ON_DEMAND or SPOT."
  }
}

variable "disk_size" {
  description = "EBS volume size for worker nodes in GB"
  type        = number
  default     = 50
  validation {
    condition     = var.disk_size >= 20 && var.disk_size <= 16384
    error_message = "Disk size must be between 20 and 16384 GB."
  }
}

variable "endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "vpc_cni_version" {
  description = "Version of VPC CNI add-on"
  type        = string
  default     = null
}

variable "coredns_version" {
  description = "Version of CoreDNS add-on"
  type        = string
  default     = null
}

variable "kube_proxy_version" {
  description = "Version of kube-proxy add-on"
  type        = string
  default     = null
}

variable "ebs_csi_version" {
  description = "Version of EBS CSI driver add-on"
  type        = string
  default     = null
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "NegativeSpaceImaging"
    ManagedBy = "Terraform"
  }
}
