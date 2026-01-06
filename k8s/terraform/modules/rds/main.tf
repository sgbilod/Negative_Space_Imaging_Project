# RDS Module - Main Configuration
# Manages RDS PostgreSQL instance with multi-AZ, encryption, and backups

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# RDS Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${var.db_name}-subnet-group"
  subnet_ids = var.subnet_ids

  tags = merge(
    var.common_tags,
    {
      Name = "${var.db_name}-subnet-group"
    }
  )
}

# RDS Parameter Group
resource "aws_db_parameter_group" "main" {
  name   = "${var.db_name}-parameter-group"
  family = "postgres${split(".", var.engine_version)[0]}"

  # Performance tuning parameters
  parameter {
    name  = "max_connections"
    value = var.max_connections
  }

  parameter {
    name  = "shared_buffers"
    value = var.shared_buffers
  }

  parameter {
    name  = "effective_cache_size"
    value = var.effective_cache_size
  }

  parameter {
    name  = "work_mem"
    value = var.work_mem
  }

  parameter {
    name  = "maintenance_work_mem"
    value = var.maintenance_work_mem
  }

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  tags = merge(
    var.common_tags,
    {
      Name = "${var.db_name}-parameter-group"
    }
  )
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier              = var.db_name
  engine                  = "postgres"
  engine_version          = var.engine_version
  instance_class          = var.instance_class
  allocated_storage       = var.allocated_storage
  storage_type            = var.storage_type
  storage_encrypted       = true
  kms_key_id              = aws_kms_key.rds.arn
  db_name                 = var.database_name
  username                = var.master_username
  password                = var.master_password
  db_subnet_group_name    = aws_db_subnet_group.main.name
  parameter_group_name    = aws_db_parameter_group.main.name
  vpc_security_group_ids  = [var.security_group_id]
  publicly_accessible     = false
  multi_az                = var.multi_az
  skip_final_snapshot     = var.skip_final_snapshot
  copy_tags_to_snapshot   = true
  backup_retention_period = var.backup_retention_period
  backup_window           = var.backup_window
  maintenance_window      = var.maintenance_window
  deletion_protection     = var.deletion_protection
  enabled_cloudwatch_logs_exports = [
    "postgresql"
  ]
  enable_iam_database_authentication = true
  monitoring_interval              = var.monitoring_interval
  monitoring_role_arn              = aws_iam_role.rds_monitoring.arn
  enable_performance_insights       = var.enable_performance_insights
  performance_insights_retention_period = var.performance_insights_retention_period
  iops                             = var.iops
  ca_cert_identifier               = "rds-ca-2019"

  tags = merge(
    var.common_tags,
    {
      Name = var.db_name
    }
  )

  depends_on = [aws_db_subnet_group.main]

  lifecycle {
    ignore_changes = [password]
  }
}

# KMS Key for RDS Encryption
resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = merge(
    var.common_tags,
    {
      Name = "${var.db_name}-kms-key"
    }
  )
}

resource "aws_kms_alias" "rds" {
  name          = "alias/${var.db_name}"
  target_key_id = aws_kms_key.rds.key_id
}

# IAM Role for RDS Monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.db_name}-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# RDS Enhanced Monitoring
resource "aws_cloudwatch_log_group" "rds_postgresql" {
  name              = "/aws/rds/instance/${var.db_name}/postgresql"
  retention_in_days = var.log_retention_days

  tags = merge(
    var.common_tags,
    {
      Name = "${var.db_name}-logs"
    }
  )
}

# Secrets Manager - DB Password
resource "aws_secretsmanager_secret" "db_password" {
  name                    = "${var.db_name}/master-password"
  description             = "RDS master password for ${var.db_name}"
  recovery_window_in_days = 7

  tags = merge(
    var.common_tags,
    {
      Name = "${var.db_name}-secret"
    }
  )
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = var.master_password
}

# RDS Event Subscription for Notifications
resource "aws_db_event_subscription" "main" {
  name      = "${var.db_name}-events"
  sns_topic = var.sns_topic_arn
  source_type = "db-instance"

  event_categories = [
    "availability",
    "backup",
    "failover",
    "failure",
    "maintenance",
    "recovery"
  ]

  tags = merge(
    var.common_tags,
    {
      Name = "${var.db_name}-events"
    }
  )
}

# RDS Proxy for Connection Pooling
resource "aws_db_proxy" "main" {
  count   = var.enable_proxy ? 1 : 0
  name    = "${var.db_name}-proxy"
  role_arn = aws_iam_role.proxy.arn
  engine_family = "POSTGRESQL"
  auth {
    auth_scheme = "SECRETS"
    secret_arn  = aws_secretsmanager_secret.db_password.arn
  }

  database_auth_config_auth_scheme = "SECRETS"
  max_connections = var.proxy_max_connections
  max_idle_connections = var.proxy_max_idle_connections
  connection_borrow_timeout = var.proxy_connection_borrow_timeout
  session_pinning_filters = ["EXCLUDE_VARIABLE_SETS"]

  tags = merge(
    var.common_tags,
    {
      Name = "${var.db_name}-proxy"
    }
  )

  depends_on = [aws_db_instance.main]
}

resource "aws_db_proxy_target_group" "main" {
  count          = var.enable_proxy ? 1 : 0
  db_proxy_name  = aws_db_proxy.main[0].name
  name           = "default"
  db_instance_identifiers = [aws_db_instance.main.identifier]

  connection_pool_config {
    max_connections              = var.proxy_max_connections
    max_idle_connections         = var.proxy_max_idle_connections
    connection_borrow_timeout    = var.proxy_connection_borrow_timeout
    init_query                   = ""
    session_pinning_filters      = ["EXCLUDE_VARIABLE_SETS"]
  }
}

# IAM Role for RDS Proxy
resource "aws_iam_role" "proxy" {
  name = "${var.db_name}-proxy-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy" "proxy_secrets" {
  name = "${var.db_name}-proxy-secrets-policy"
  role = aws_iam_role.proxy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:DescribeSecret",
          "secretsmanager:ListSecretVersionIds",
          "secretsmanager:GetResourcePolicy",
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.db_password.arn
      }
    ]
  })
}
