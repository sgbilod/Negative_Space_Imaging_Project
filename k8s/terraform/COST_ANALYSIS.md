# Infrastructure Cost Analysis & Optimization

## Executive Summary

The Negative Space Imaging infrastructure spans three environments with varying resource levels:
- **Development**: ~$133/month (minimal, cost-optimized)
- **Staging**: ~$303/month (balanced)
- **Production**: ~$1,273+/month (full features, high availability)

**Total Multi-Environment Cost**: ~$1,709/month

## Detailed Cost Breakdown

### Development Environment (~$133/month)

#### VPC
| Resource | Count | Size | Cost/Month |
|----------|-------|------|-----------|
| NAT Gateway | 1 | - | $32 |
| Data Transfer | - | ~10GB | $0.90 |
| **VPC Subtotal** | - | - | **$33** |

#### EKS
| Resource | Count | Type | Cost/Month |
|----------|-------|------|-----------|
| EKS Cluster | 1 | - | $73 |
| EC2 Nodes | 2 | t3.medium | $15 |
| **EKS Subtotal** | - | - | **$88** |

#### RDS
| Resource | Type | Config | Cost/Month |
|----------|------|--------|-----------|
| Database Instance | db.t3.micro | Single AZ | $15 |
| Storage | 20GB gp2 | - | $2 |
| **RDS Subtotal** | - | - | **$17** |

#### ElastiCache
| Resource | Type | Config | Cost/Month |
|----------|------|--------|-----------|
| Cache Node | cache.t3.micro | 1 node | $15 |
| **Cache Subtotal** | - | - | **$15** |

#### **Total Dev**: $133/month

**Cost Optimization Tips**:
- ✅ Already using lowest cost instance types
- ✅ Single-node Redis (no HA overhead)
- ✅ Minimal storage allocation
- Consider: Stop resources during off-hours (–50%)

---

### Staging Environment (~$303/month)

#### VPC
| Resource | Count | Size | Cost/Month |
|----------|-------|------|-----------|
| NAT Gateways | 2 | - | $64 |
| Data Transfer | - | ~50GB | $4.50 |
| **VPC Subtotal** | - | - | **$69** |

#### EKS
| Resource | Count | Type | Cost/Month |
|----------|-------|------|-----------|
| EKS Cluster | 1 | - | $73 |
| EC2 Nodes | 3 | t3.large | $164 |
| **EKS Subtotal** | - | - | **$237** |

#### RDS
| Resource | Type | Config | Cost/Month |
|----------|------|--------|-----------|
| Database Instance | db.t3.small | Multi-AZ (2x) | $80 |
| Storage | 50GB gp2 | - | $5 |
| **RDS Subtotal** | - | - | **$85** |

#### ElastiCache
| Resource | Type | Config | Cost/Month |
|----------|------|--------|-----------|
| Cache Nodes | cache.t3.small | 3 nodes | $45 |
| Data Replication | - | Multi-AZ | $0 |
| **Cache Subtotal** | - | - | **$45** |

#### **Total Staging**: ~$303/month

**Cost Optimization Tips**:
- Consider: Spot instances for non-critical workloads (–40%)
- Consider: Reserved instances for 1-year commitment (–30%)
- Consider: Scheduled scaling (stop at 22:00, start at 08:00)

---

### Production Environment (~$1,273+/month)

#### VPC
| Resource | Count | Size | Cost/Month |
|----------|-------|------|-----------|
| NAT Gateways | 2 | - | $64 |
| VPC Endpoints | 3 | - | $21 |
| Data Transfer | - | ~500GB | $45 |
| **VPC Subtotal** | - | - | **$130** |

#### EKS
| Resource | Count | Type | Cost/Month |
|----------|-------|------|-----------|
| EKS Cluster | 1 | - | $73 |
| EC2 Nodes | 5+ | m5.large/xlarge | $550+ |
| Auto-Scaling | - | - | $0 |
| **EKS Subtotal** | - | - | **$623+** |

#### RDS
| Resource | Type | Config | Cost/Month |
|----------|------|--------|-----------|
| Database Instance | db.r5.large | Multi-AZ (2x) | $450 |
| Storage | 200GB gp3 | - | $20 |
| Backups | 30-day | - | $0 |
| **RDS Subtotal** | - | - | **$470** |

#### ElastiCache
| Resource | Type | Config | Cost/Month |
|----------|------|--------|-----------|
| Cache Nodes | cache.r6g.xlarge | 6 nodes | $300 |
| Data Replication | - | Multi-AZ | $0 |
| **Cache Subtotal** | - | - | **$300** |

#### **Total Production**: ~$1,273+/month

---

## Cost Optimization Strategies

### 1. Reserved Instances (RI) – Save 30-40%

#### Commit for Stability
```
1-year RI savings:
- EKS nodes (m5.large): $500 → $350 = 30% savings
- RDS (db.r5.large): $450 → $315 = 30% savings
- Total prod savings: ~$420/month or $5,040/year
```

#### Implementation
```bash
# Purchase 1-year Reserved Instance for m5.large
aws ec2 purchase-reserved-instances-offering \
  --instance-type m5.large \
  --availability-zone us-east-1a \
  --term-length 31536000 \
  --offering-type 1-year \
  --instance-count 5
```

### 2. Spot Instances – Save 70%

#### For Non-Critical Workloads
```
Cost: $550 (on-demand m5.large)
With Spot: $165 (70% cheaper)
Savings: $385/month per 5 nodes
Risk: Interruption rate ~2%
```

#### Implementation
```hcl
# In terraform.tfvars.prod
spot_price = "0.25"        # Max price for spot instance
capacity_type = "spot"     # Use spot instead of on-demand
```

### 3. Scheduled Scaling – Save 20-40%

#### Non-24/7 Workloads
```bash
# Scale down 18:00-06:00 (10 hours/day)
# Save 42% on compute costs

Daily costs:
24h @ $623 = $623/day
14h @ $623 + 10h @ $312 = $874/day * 30 = ~$366/month savings
```

#### Implementation
```hcl
# In main.tf
resource "aws_autoscaling_schedule" "scale_down" {
  scheduled_action_name = "scale-down-evening"
  min_size = 1
  max_size = 5
  desired_capacity = 1
  recurrence = "0 18 * * *"  # 18:00 UTC daily
}

resource "aws_autoscaling_schedule" "scale_up" {
  scheduled_action_name = "scale-up-morning"
  min_size = 3
  max_size = 20
  desired_capacity = 5
  recurrence = "0 6 * * *"   # 06:00 UTC daily
}
```

### 4. Right-Sizing – Save 10-30%

#### Current vs Optimized
```
Prod Instance Sizes:
Current: m5.large (2 vCPU, 8GB RAM) × 5 = $550/month
Optimized: t3.large (2 vCPU, 8GB RAM) × 5 = $332/month
Savings: 40% ($218/month)

Note: t3 offers 20% better performance/$ but less consistent throughput
```

#### Decision Tree
```
If: Throughput is consistent & high
Then: Keep m5.large

If: Throughput varies & lower baseline
Then: Switch to t3.large (burstable)

If: Workload is bursty & unpredictable
Then: Switch to t4g (Graviton, 20% cheaper)
```

### 5. Storage Optimization – Save 10-20%

#### RDS Storage
```
Current: gp2 @ $0.10/GB = $20/month (200GB)
Optimized: gp3 @ $0.08/GB = $16/month (200GB)
Savings: 20% ($4/month) + better performance
```

#### ElastiCache Memory
```
Current: cache.r6g.xlarge × 6 = 24GB total = $300/month
If memory usage is <16GB total:
Optimized: cache.r6g.large × 4 = 16GB total = $160/month
Savings: 47% ($140/month)
```

### 6. VPC Endpoint Optimization

#### Current Costs
```
3 VPC Endpoints × $7.20 = $21.60/month
Data transfer: $0.01/GB
```

#### Savings Strategy
```
Remove endpoints for low-traffic services:
- Keep S3 endpoint (highest traffic)
- Remove DynamoDB endpoint (if unused)
- Remove ECR endpoint (if pulling images rarely)
Savings: ~$14/month
```

---

## Cost Monitoring & Alerts

### AWS Cost Explorer Setup

```bash
# Enable Cost Anomaly Detection
aws ce create-anomaly-monitor \
  --anomaly-monitor '{
    "MonitorName": "nsi-prod-costs",
    "MonitorType": "DIMENSIONAL",
    "MonitorDimension": "SERVICE",
    "MonitorSpecification": "NONE"
  }'
```

### CloudWatch Alarms for Cost Control

```hcl
# In main.tf
resource "aws_cloudwatch_metric_alarm" "billing_alert" {
  alarm_name          = "nsi-prod-billing-alert"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"  # 1 day
  statistic           = "Maximum"
  threshold           = "1500"   # Alert if > $1500/day
  alarm_actions       = [aws_sns_topic.notifications.arn]
}
```

### Monthly Budget Report

```bash
#!/bin/bash
# Generate monthly cost report

aws ce get-cost-and-usage \
  --time-period Start=$(date -d "1 month ago" +%Y-%m-01),End=$(date +%Y-%m-01) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --output json | \
  jq '.ResultsByTime[0].Groups | sort_by(.Metrics.BlendedCost | tonumber) |
    .[] | "\(.Keys[0]): $\(.Metrics.BlendedCost.Amount)"'
```

---

## Total Cost of Ownership (TCO)

### 1-Year Projection

```
Month  Dev     Staging  Prod    Total/Mo
1      $133    $303     $1,273  $1,709
...
12     $133    $303     $873*   $1,309

*Production with 1-year RIs and spot instances

Year 1 Total: $18,000 (average)
Savings with optimization: $4,000-6,000
Optimized Year 1 Total: $12,000-14,000
```

### 3-Year TCO

```
Without Optimization:
Year 1: $18,000
Year 2: $18,600 (5% inflation)
Year 3: $19,530
Total: $56,130

With Optimization (RIs + Spot + Scheduled Scaling):
Year 1: $13,000
Year 2: $13,650 (5% inflation)
Year 3: $14,333
Total: $40,983

TCO Savings: $15,147 (27% reduction)
```

---

## Recommendation Matrix

### By Use Case

| Use Case | Recommended Config | Est. Cost | Notes |
|----------|-------------------|-----------|-------|
| Dev/Test | Dev config | $133/mo | Cost-optimized |
| Demo/PoC | Dev config | $133/mo | Perfect for trials |
| Staging | Staging config | $303/mo | Good balance |
| Production | Prod + RI | $873/mo | RI + spot discount |
| Enterprise | Prod + RI + VPC | $900/mo | Full control plane |
| High-Performance | Prod (m5) + RI | $950/mo | Best throughput |

### Cost Reduction Checklist

- [ ] Implement 1-year Reserved Instances (–30%)
- [ ] Enable spot instances for non-critical (–70%)
- [ ] Schedule scaling for non-24/7 (–40%)
- [ ] Right-size instances (–10-30%)
- [ ] Optimize storage (–10-20%)
- [ ] Remove unused VPC endpoints (–$15/mo)
- [ ] Enable AWS CloudTrail (–$2/mo)
- [ ] Delete snapshots older than 30 days (–20%)
- [ ] Enable S3 Intelligent-Tiering (–15%)
- [ ] Set up cost alerts (–5% waste)

**Potential Monthly Savings: $200-400 (25-40% reduction)**

---

## Budget Planning Template

```hcl
# In terraform.tfvars

# Cost Control
max_monthly_budget = 1500  # Alert if exceeded
preferred_cost_tier = "standard"  # standard | economy | performance

# Resource Sizing Strategy
cost_optimization_enabled = true
use_spot_instances = true
use_reserved_instances = true
scheduled_scaling_enabled = true
```

---

**Last Updated**: December 14, 2025
**Review Frequency**: Monthly
**Next Review**: January 14, 2026
