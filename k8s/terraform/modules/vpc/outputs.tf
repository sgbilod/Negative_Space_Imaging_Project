# VPC Module - Outputs

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "VPC CIDR block"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "public_subnet_cidrs" {
  description = "List of public subnet CIDR blocks"
  value       = aws_subnet.public[*].cidr_block
}

output "private_subnet_cidrs" {
  description = "List of private subnet CIDR blocks"
  value       = aws_subnet.private[*].cidr_block
}

output "nat_gateway_ids" {
  description = "List of NAT Gateway IDs"
  value       = aws_nat_gateway.main[*].id
}

output "nat_gateway_ips" {
  description = "List of NAT Gateway public IPs"
  value       = aws_eip.nat[*].public_ip
}

output "internet_gateway_id" {
  description = "Internet Gateway ID"
  value       = aws_internet_gateway.main.id
}

output "eks_control_plane_security_group_id" {
  description = "EKS control plane security group ID"
  value       = aws_security_group.eks_control_plane.id
}

output "eks_worker_nodes_security_group_id" {
  description = "EKS worker nodes security group ID"
  value       = aws_security_group.eks_worker_nodes.id
}

output "rds_security_group_id" {
  description = "RDS security group ID"
  value       = aws_security_group.rds.id
}

output "elasticache_security_group_id" {
  description = "ElastiCache security group ID"
  value       = aws_security_group.elasticache.id
}

output "vpc_flow_logs_group_name" {
  description = "CloudWatch log group name for VPC Flow Logs"
  value       = try(aws_cloudwatch_log_group.vpc_flow_logs[0].name, null)
}

output "vpc_endpoints_config" {
  description = "Configuration for VPC endpoints"
  value = {
    vpc_id                                    = aws_vpc.main.id
    eks_control_plane_security_group_id       = aws_security_group.eks_control_plane.id
    eks_worker_nodes_security_group_id        = aws_security_group.eks_worker_nodes.id
    rds_security_group_id                     = aws_security_group.rds.id
    elasticache_security_group_id             = aws_security_group.elasticache.id
  }
}
