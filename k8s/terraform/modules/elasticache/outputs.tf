# ElastiCache Module - Outputs

output "replication_group_id" {
  description = "ElastiCache replication group ID"
  value       = aws_elasticache_replication_group.main.id
}

output "replication_group_arn" {
  description = "ElastiCache replication group ARN"
  value       = aws_elasticache_replication_group.main.arn
}

output "primary_endpoint_address" {
  description = "Primary endpoint address"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "reader_endpoint_address" {
  description = "Reader endpoint address for read operations"
  value       = aws_elasticache_replication_group.main.reader_endpoint_address
}

output "port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.main.port
}

output "engine_version" {
  description = "Redis engine version"
  value       = aws_elasticache_replication_group.main.engine_version
}

output "member_clusters" {
  description = "List of member cluster IDs"
  value       = aws_elasticache_replication_group.main.member_clusters
}

output "cluster_endpoint" {
  description = "Cluster endpoint for connection"
  value       = "${aws_elasticache_replication_group.main.primary_endpoint_address}:${aws_elasticache_replication_group.main.port}"
}

output "cluster_reader_endpoint" {
  description = "Cluster reader endpoint for read-only operations"
  value       = "${aws_elasticache_replication_group.main.reader_endpoint_address}:${aws_elasticache_replication_group.main.port}"
}

output "subnet_group_name" {
  description = "ElastiCache subnet group name"
  value       = aws_elasticache_subnet_group.main.name
}

output "parameter_group_name" {
  description = "ElastiCache parameter group name"
  value       = aws_elasticache_parameter_group.main.name
}

output "security_group_id" {
  description = "Security group ID for ElastiCache cluster"
  value       = aws_elasticache_replication_group.main.security_group_ids[0]
}

output "auth_token_secret_arn" {
  description = "Secrets Manager secret ARN for auth token"
  value       = try(aws_secretsmanager_secret.auth_token[0].arn, null)
}

output "auth_token_secret_name" {
  description = "Secrets Manager secret name for auth token"
  value       = try(aws_secretsmanager_secret.auth_token[0].name, null)
}

output "slow_log_group_name" {
  description = "CloudWatch log group name for slow log"
  value       = aws_cloudwatch_log_group.redis_slow_log.name
}

output "engine_log_group_name" {
  description = "CloudWatch log group name for engine log"
  value       = aws_cloudwatch_log_group.redis_engine_log.name
}

output "redis_connection_string" {
  description = "Redis connection string (without auth token)"
  value       = "redis://${aws_elasticache_replication_group.main.primary_endpoint_address}:${aws_elasticache_replication_group.main.port}/0"
}

output "redis_reader_connection_string" {
  description = "Redis reader connection string (without auth token)"
  value       = "redis://${aws_elasticache_replication_group.main.reader_endpoint_address}:${aws_elasticache_replication_group.main.port}/0"
}
