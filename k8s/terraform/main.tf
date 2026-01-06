# Main Terraform Configuration
# Instantiates all infrastructure modules

locals {
  environment = var.environment
  common_tags = merge(
    var.common_tags,
    {
      Environment = local.environment
      ManagedBy   = "Terraform"
      Project     = "NegativeSpaceImaging"
    }
  )
}

# SNS Topic for notifications (shared across modules)
resource "aws_sns_topic" "notifications" {
  name = "${local.environment}-nsi-notifications"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.environment}-nsi-notifications"
    }
  )
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  environment             = var.environment
  vpc_cidr                = var.vpc_cidr
  availability_zones      = data.aws_availability_zones.available.names
  enable_nat_gateway      = var.enable_nat_gateway
  enable_flow_logs        = var.enable_flow_logs
  flow_logs_retention_days = var.flow_logs_retention_days
  common_tags             = local.common_tags
}

# EKS Module
module "eks" {
  source = "./modules/eks"

  environment                      = var.environment
  cluster_name                     = var.cluster_name
  kubernetes_version               = var.kubernetes_version
  subnet_ids                       = concat(module.vpc.public_subnet_ids, module.vpc.private_subnet_ids)
  private_subnet_ids               = module.vpc.private_subnet_ids
  eks_control_plane_security_group_id = module.vpc.eks_control_plane_security_group_id
  instance_types                   = var.instance_types
  desired_capacity                 = var.desired_capacity
  min_capacity                     = var.min_capacity
  max_capacity                     = var.max_capacity
  capacity_type                    = var.capacity_type
  disk_size                        = var.disk_size
  endpoint_public_access           = var.endpoint_public_access
  public_access_cidrs              = var.public_access_cidrs
  log_retention_days               = var.log_retention_days
  common_tags                      = local.common_tags

  depends_on = [module.vpc]
}

# RDS Module
module "rds" {
  source = "./modules/rds"

  environment                      = var.environment
  db_name                          = var.db_name
  database_name                    = var.database_name
  master_username                  = var.master_username
  master_password                  = var.master_password
  instance_class                   = var.rds_instance_class
  engine_version                   = var.rds_engine_version
  allocated_storage                = var.allocated_storage
  storage_type                     = var.storage_type
  iops                             = var.iops
  multi_az                         = var.multi_az
  backup_retention_period          = var.backup_retention_period
  backup_window                    = var.backup_window
  maintenance_window               = var.maintenance_window
  deletion_protection              = var.deletion_protection
  skip_final_snapshot              = var.skip_final_snapshot
  subnet_ids                       = module.vpc.private_subnet_ids
  security_group_id                = module.vpc.rds_security_group_id
  monitoring_interval              = var.monitoring_interval
  enable_performance_insights      = var.enable_performance_insights
  performance_insights_retention_period = var.performance_insights_retention_period
  log_retention_days               = var.log_retention_days
  sns_topic_arn                    = aws_sns_topic.notifications.arn
  enable_proxy                     = var.enable_proxy
  common_tags                      = local.common_tags

  depends_on = [module.vpc]
}

# ElastiCache Module
module "elasticache" {
  source = "./modules/elasticache"

  environment                    = var.environment
  cluster_name                   = var.cache_cluster_name
  engine_version                 = var.cache_engine_version
  node_type                      = var.cache_node_type
  num_cache_nodes                = var.num_cache_nodes
  port                           = var.cache_port
  automatic_failover_enabled     = var.cache_automatic_failover
  multi_az_enabled               = var.cache_multi_az
  transit_encryption_enabled     = var.cache_transit_encryption
  auth_token                     = var.cache_auth_token
  subnet_ids                     = module.vpc.private_subnet_ids
  security_group_id              = module.vpc.elasticache_security_group_id
  log_retention_days             = var.log_retention_days
  sns_topic_arn                  = aws_sns_topic.notifications.arn
  common_tags                    = local.common_tags

  depends_on = [module.vpc]
}

# Data source for available AZs
data "aws_availability_zones" "available" {
  state = "available"
}
