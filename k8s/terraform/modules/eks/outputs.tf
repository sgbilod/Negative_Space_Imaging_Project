# EKS Module - Outputs

output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_version" {
  description = "EKS cluster Kubernetes version"
  value       = aws_eks_cluster.main.version
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded cluster certificate authority data"
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "node_group_id" {
  description = "EKS node group ID"
  value       = aws_eks_node_group.main.id
}

output "node_group_status" {
  description = "EKS node group status"
  value       = aws_eks_node_group.main.status
}

output "oidc_provider_arn" {
  description = "OIDC provider ARN for IRSA"
  value       = aws_iam_openid_connect_provider.cluster.arn
}

output "oidc_provider_url" {
  description = "OIDC provider URL"
  value       = aws_iam_openid_connect_provider.cluster.url
}

output "oidc_provider_thumbprint" {
  description = "OIDC provider thumbprint"
  value       = data.tls_certificate.cluster.certificates[0].sha1_fingerprint
}

output "kubeconfig" {
  description = "Kubeconfig for EKS cluster"
  value = {
    apiVersion      = "apiextensions.k8s.io/v1"
    kind            = "CustomResourceDefinition"
    cluster_name    = aws_eks_cluster.main.name
    endpoint        = aws_eks_cluster.main.endpoint
    ca_cert_data    = aws_eks_cluster.main.certificate_authority[0].data
    region          = data.aws_region.current.name
  }
  sensitive = true
}

output "node_iam_role_arn" {
  description = "IAM role ARN for worker nodes"
  value       = aws_iam_role.eks_node_group.arn
}

output "node_iam_role_name" {
  description = "IAM role name for worker nodes"
  value       = aws_iam_role.eks_node_group.name
}

output "cluster_log_group_name" {
  description = "CloudWatch log group name for cluster logs"
  value       = aws_cloudwatch_log_group.eks_cluster.name
}

output "irsa_roles" {
  description = "IAM roles for service accounts"
  value = {
    vpc_cni_role_arn  = aws_iam_role.vpc_cni.arn
    ebs_csi_role_arn  = aws_iam_role.ebs_csi.arn
  }
}

output "cluster_autoscaler_role_arn" {
  description = "ARN for cluster autoscaler (to be created separately)"
  value       = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.cluster_name}-cluster-autoscaler-role"
}
