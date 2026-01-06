# ElastiCache Module - Main Configuration
# Manages Redis cluster with high availability and encryption

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.cluster_name}-subnet-group"
  subnet_ids = var.subnet_ids

  tags = merge(
    var.common_tags,
    {
      Name = "${var.cluster_name}-subnet-group"
    }
  )
}

# ElastiCache Parameter Group
resource "aws_elasticache_parameter_group" "main" {
  name   = "${var.cluster_name}-parameter-group"
  family = "redis${split(".", var.engine_version)[0]}.${split(".", var.engine_version)[1]}"

  # Performance tuning parameters
  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  parameter {
    name  = "tcp-keepalive"
    value = "300"
  }

  tags = merge(
    var.common_tags,
    {
      Name = "${var.cluster_name}-parameter-group"
    }
  )
}

# ElastiCache Replication Group (Redis Cluster)
resource "aws_elasticache_replication_group" "main" {
  replication_group_description = "Redis cluster for ${var.cluster_name}"
  engine                         = "redis"
  engine_version                 = var.engine_version
  node_type                      = var.node_type
  num_cache_clusters             = var.num_cache_nodes
  parameter_group_name           = aws_elasticache_parameter_group.main.name
  port                           = var.port
  security_group_ids             = [var.security_group_id]
  subnet_group_name              = aws_elasticache_subnet_group.main.name
  automatic_failover_enabled     = var.automatic_failover_enabled
  multi_az_enabled               = var.multi_az_enabled
  at_rest_encryption_enabled     = true
  transit_encryption_enabled     = var.transit_encryption_enabled
  auth_token                     = var.auth_token
  auto_minor_version_upgrade     = true
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "slow-log"
  }

  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_engine_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "engine-log"
  }

  tags = merge(
    var.common_tags,
    {
      Name = var.cluster_name
    }
  )

  depends_on = [aws_elasticache_subnet_group.main]
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "redis_slow_log" {
  name              = "/aws/elasticache/${var.cluster_name}/slow-log"
  retention_in_days = var.log_retention_days

  tags = merge(
    var.common_tags,
    {
      Name = "${var.cluster_name}-slow-log"
    }
  )
}

resource "aws_cloudwatch_log_group" "redis_engine_log" {
  name              = "/aws/elasticache/${var.cluster_name}/engine-log"
  retention_in_days = var.log_retention_days

  tags = merge(
    var.common_tags,
    {
      Name = "${var.cluster_name}-engine-log"
    }
  )
}

# Secrets Manager - Auth Token
resource "aws_secretsmanager_secret" "auth_token" {
  count                   = var.auth_token != "" ? 1 : 0
  name                    = "${var.cluster_name}/auth-token"
  description             = "Redis auth token for ${var.cluster_name}"
  recovery_window_in_days = 7

  tags = merge(
    var.common_tags,
    {
      Name = "${var.cluster_name}-auth-token-secret"
    }
  )
}

resource "aws_secretsmanager_secret_version" "auth_token" {
  count         = var.auth_token != "" ? 1 : 0
  secret_id     = aws_secretsmanager_secret.auth_token[0].id
  secret_string = var.auth_token
}

# ElastiCache Event Subscription
resource "aws_elasticache_event_subscription" "main" {
  name      = "${var.cluster_name}-events"
  sns_topic = var.sns_topic_arn
  source_type = "cluster"

  event_categories = [
    "availability",
    "backup",
    "configuration change",
    "creation",
    "deletion",
    "failover",
    "maintenance",
    "notification",
    "recovery"
  ]

  tags = merge(
    var.common_tags,
    {
      Name = "${var.cluster_name}-events"
    }
  )
}

# CloudWatch Alarms for cache performance
resource "aws_cloudwatch_metric_alarm" "cpu_utilization" {
  alarm_name          = "${var.cluster_name}-cpu-utilization-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "75"
  alarm_description   = "Alert when ElastiCache CPU utilization is high"
  treat_missing_data  = "notBreaching"

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.main.id
  }

  alarm_actions = [var.sns_topic_arn]

  tags = var.common_tags
}

resource "aws_cloudwatch_metric_alarm" "evictions" {
  alarm_name          = "${var.cluster_name}-evictions-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Evictions"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "1000"
  alarm_description   = "Alert when cache evictions are high"
  treat_missing_data  = "notBreaching"

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.main.id
  }

  alarm_actions = [var.sns_topic_arn]

  tags = var.common_tags
}

resource "aws_cloudwatch_metric_alarm" "swap_usage" {
  alarm_name          = "${var.cluster_name}-swap-usage-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "SwapUsage"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Maximum"
  threshold           = "52428800" # 50 MB
  alarm_description   = "Alert when swap usage is detected"
  treat_missing_data  = "notBreaching"

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.main.id
  }

  alarm_actions = [var.sns_topic_arn]

  tags = var.common_tags
}
