# Terraform Configuration
# © 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket = "negative-space-terraform-state"
    key    = "global/s3/terraform.tfstate"
    region = "us-east-1"

    dynamodb_table = "terraform-locks"
    encrypt        = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.22"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

provider "aws" {
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = "Negative Space Imaging"
      Environment = "Production"
      Terraform   = "true"
    }
  }
}

module "vpc" {
  source = "./modules/vpc"

  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets    = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets     = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]
}

module "eks" {
  source = "./modules/eks"

  cluster_name    = "negative-space-cluster"
  cluster_version = "1.28"

  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids

  node_groups = {
    quantum = {
      instance_types = ["g5.48xlarge"]
      min_size      = 10
      max_size      = 100
      desired_size  = 10
    }
    visualization = {
      instance_types = ["p4d.24xlarge"]
      min_size      = 5
      max_size      = 50
      desired_size  = 5
    }
    general = {
      instance_types = ["m6i.32xlarge"]
      min_size      = 5
      max_size      = 30
      desired_size  = 5
    }
  }
}

module "storage" {
  source = "./modules/storage"

  ebs_storage_class = {
    name        = "quantum-storage"
    type        = "io2"
    iops_per_gb = 100
  }

  efs_storage_class = {
    name = "quantum-shared-storage"
  }
}

module "monitoring" {
  source = "./modules/monitoring"

  prometheus_retention    = "30d"
  grafana_admin_password = var.grafana_admin_password
  alert_webhook_url      = var.alert_webhook_url
}

module "security" {
  source = "./modules/security"

  vault_kms_key_id = aws_kms_key.vault.id
  certificate_arn  = aws_acm_certificate.main.arn
}
