# RDS Module - Outputs

output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.main.id
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.main.arn
}

output "db_instance_address" {
  description = "RDS instance endpoint address"
  value       = aws_db_instance.main.address
}

output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "db_instance_username" {
  description = "RDS instance master username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

output "db_instance_resource_id" {
  description = "RDS instance resource ID"
  value       = aws_db_instance.main.resource_id
}

output "db_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "db_subnet_group_id" {
  description = "DB subnet group ID"
  value       = aws_db_subnet_group.main.id
}

output "db_parameter_group_id" {
  description = "DB parameter group ID"
  value       = aws_db_parameter_group.main.id
}

output "db_security_group_id" {
  description = "DB security group ID"
  value       = aws_db_instance.main.vpc_security_group_ids[0]
}

output "kms_key_id" {
  description = "KMS key ID for RDS encryption"
  value       = aws_kms_key.rds.id
}

output "kms_key_arn" {
  description = "KMS key ARN for RDS encryption"
  value       = aws_kms_key.rds.arn
}

output "db_secret_arn" {
  description = "Secrets Manager secret ARN for DB password"
  value       = aws_secretsmanager_secret.db_password.arn
}

output "db_secret_name" {
  description = "Secrets Manager secret name for DB password"
  value       = aws_secretsmanager_secret.db_password.name
}

output "rds_proxy_endpoint" {
  description = "RDS Proxy endpoint"
  value       = try(aws_db_proxy.main[0].endpoint, null)
}

output "rds_proxy_arn" {
  description = "RDS Proxy ARN"
  value       = try(aws_db_proxy.main[0].arn, null)
}

output "connection_string" {
  description = "RDS instance connection string (without password)"
  value       = "postgresql://${aws_db_instance.main.username}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
  sensitive   = true
}

output "proxy_connection_string" {
  description = "RDS Proxy connection string (without password)"
  value       = try("postgresql://${aws_db_instance.main.username}@${aws_db_proxy.main[0].endpoint}/${aws_db_instance.main.db_name}", null)
  sensitive   = true
}

output "monitoring_role_arn" {
  description = "IAM role ARN for RDS monitoring"
  value       = aws_iam_role.rds_monitoring.arn
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name for PostgreSQL logs"
  value       = aws_cloudwatch_log_group.rds_postgresql.name
}
