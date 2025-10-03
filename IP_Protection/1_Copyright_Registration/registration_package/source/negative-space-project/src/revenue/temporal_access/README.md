# Temporal Access Control System (TACS)

The Temporal Access Control System (TACS) is an advanced security framework that controls access to digital assets based on temporal, spatial, astronomical, and identity conditions. It creates time-locked content that can only be accessed under specific combinations of circumstances, providing an unprecedented level of security and controlled data disclosure.

## Overview

The Temporal Access Control System leverages multiple dimensions of access control:

1. **Temporal Conditions** - Access based on time of day, day of week, date ranges, or specific timeframes
2. **Spatial Conditions** - Access based on geographic location within specified boundaries
3. **Astronomical Conditions** - Access triggered by celestial events like full moons, solstices, or planetary alignments
4. **Identity Conditions** - Access based on user roles, attributes, or verification status
5. **Quantum Verification** - Integration with the Quantum Entangled Ledger for cryptographic verification
6. **Compound Conditions** - Logical combinations of multiple conditions (AND, OR, XOR)

## Key Components

### Access Conditions

The fundamental building blocks of the system, conditions define the specific circumstances under which access is granted:

```python
# Create a business hours condition
business_hours = AccessCondition(
    condition_type="temporal",
    parameters={
        "hours_of_day": list(range(9, 18)),  # 9 AM to 6 PM
        "days_of_week": [0, 1, 2, 3, 4]      # Monday to Friday
    },
    description="Business hours only"
)

# Create a location-based condition
office_location = AccessCondition(
    condition_type="spatial",
    parameters={
        "location": [40.7128, -74.0060],  # NYC coordinates
        "radius": 500  # 500 meters
    },
    description="Office premises only"
)
```

### Access Policies

Policies combine one or more conditions with logical operators:

```python
# Create a policy that requires both conditions
secure_access_policy = TemporalAccessPolicy(
    name="Secure Office Access",
    conditions=[business_hours, office_location],
    logical_operator="AND",
    description="Access allowed only during business hours and from the office"
)
```

### Encrypted Resources

Digital assets protected by access policies:

```python
# Encrypt sensitive data with the policy
encrypted_resource = tacs.encrypt_resource(
    data=sensitive_data,
    policy_id=secure_access_policy.policy_id,
    resource_type="confidential_document"
)
```

### Access Control

The system evaluates access attempts against the conditions:

```python
# Attempt to access the resource
access_result = tacs.access_resource(
    resource_id=encrypted_resource.resource_id,
    context={
        "current_time": datetime.now(),
        "current_location": [40.7129, -74.0061],
        "identity": {"id": "employee123", "roles": ["employee"]}
    }
)

if access_result["granted"]:
    # Access granted, use the decrypted data
    data = access_result["data"]
else:
    # Access denied
    reason = access_result["reason"]
```

## Integration with Other Systems

### Quantum Entangled Ledger

The TACS integrates with the Quantum Entangled Ledger to provide quantum-verified access controls:

```python
# Initialize with quantum ledger
quantum_ledger = QuantumEntangledLedger()
tacs = TemporalAccessControlSystem(quantum_ledger=quantum_ledger)

# Create a quantum verification condition
quantum_condition = AccessCondition(
    condition_type="quantum",
    parameters={
        "verification_type": "document",
        "record_id": quantum_record_id
    }
)
```

### Mnemonic Data Architecture

TACS can be used to control access to specific parts of the Mnemonic Data Architecture:

```python
# Initialize with mnemonic architecture
mda = MnemonicDataArchitecture()
tacs = TemporalAccessControlSystem(mnemonic_data_architecture=mda)

# Control access to specific data clusters
mda.set_access_control(cluster_id, tacs.policies[policy_id])
```

## Advanced Features

### Astronomical Access Triggers

The system can use celestial events as access triggers:

```python
# Create a full moon access condition
full_moon_condition = AccessCondition(
    condition_type="astronomical",
    parameters={
        "event": "full_moon"
    },
    description="Access allowed only during full moon"
)

# Create a planetary alignment condition
alignment_condition = AccessCondition(
    condition_type="astronomical",
    parameters={
        "event": "planetary_alignment",
        "planets": ["mars", "jupiter", "saturn"],
        "max_angle": 15  # 15 degrees
    },
    description="Access allowed during alignment of Mars, Jupiter, and Saturn"
)
```

### Time-Limited Access

Restrict access to specific time windows:

```python
# Create a time window condition
limited_time_condition = AccessCondition(
    condition_type="temporal",
    parameters={
        "start_time": "2025-09-01T00:00:00",
        "end_time": "2025-09-30T23:59:59"
    },
    description="September 2025 access only"
)
```

### Role-Based Access Control

Restrict access based on user roles:

```python
# Create a role-based condition
admin_condition = AccessCondition(
    condition_type="identity",
    parameters={
        "allowed_roles": ["admin", "security_officer"]
    },
    description="Administrators and security officers only"
)
```

## Use Cases

### 1. Financial Trading Systems

Restrict trading activities to market hours and authorized locations:

```python
trading_policy = tacs.create_policy(
    name="Secure Trading Policy",
    conditions=[
        {
            "condition_type": "temporal",
            "parameters": {
                "hours_of_day": list(range(9, 17)),  # 9 AM to 5 PM
                "days_of_week": [0, 1, 2, 3, 4]      # Monday to Friday
            }
        },
        {
            "condition_type": "spatial",
            "parameters": {
                "location": [40.7064, -74.0094],  # Wall Street coordinates
                "radius": 500  # 500 meters
            }
        },
        {
            "condition_type": "identity",
            "parameters": {
                "allowed_roles": ["trader", "risk_manager"]
            }
        }
    ],
    logical_operator="AND"
)
```

### 2. Legal Document Access

Control access to legal documents based on case status and jurisdiction:

```python
legal_document_policy = tacs.create_policy(
    name="Legal Document Access",
    conditions=[
        {
            "condition_type": "identity",
            "parameters": {
                "allowed_roles": ["attorney", "paralegal", "judge"]
            }
        },
        {
            "condition_type": "compound",
            "parameters": {
                "operator": "OR",
                "conditions": [
                    {
                        "condition_type": "spatial",
                        "parameters": {
                            "location": [38.8977, -77.0365],  # DC coordinates
                            "radius": 10000  # 10 km
                        }
                    },
                    {
                        "condition_type": "identity",
                        "parameters": {
                            "allowed_attributes": {"security_clearance": "top_secret"}
                        }
                    }
                ]
            }
        }
    ]
)
```

### 3. Time Capsule Documents

Create digital time capsules that only unlock on specific dates or astronomical events:

```python
time_capsule_policy = tacs.create_policy(
    name="50 Year Time Capsule",
    conditions=[
        {
            "condition_type": "temporal",
            "parameters": {
                "start_time": "2075-01-01T00:00:00"
            }
        }
    ]
)
```

## Revenue Opportunities

The Temporal Access Control System offers significant revenue potential through:

1. **Enterprise SaaS Licensing** - Tiered subscription models for organizations
2. **Specialized Industry Solutions** - Customized packages for finance, healthcare, legal
3. **API Access** - Developer access for integration into third-party applications
4. **Consulting Services** - Implementation and customization services
5. **Bundle Pricing** - Discounted pricing when combined with other Negative Space systems

For detailed revenue projections, see [Revenue Opportunities](revenue_opportunities.md).

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from temporal_access_control import TemporalAccessControlSystem

# Initialize the system
tacs = TemporalAccessControlSystem()

# Create a simple time-based policy
policy_result = tacs.create_policy(
    name="Business Hours Policy",
    conditions=[{
        "condition_type": "temporal",
        "parameters": {
            "hours_of_day": list(range(9, 18)),  # 9 AM to 6 PM
            "days_of_week": [0, 1, 2, 3, 4]  # Monday to Friday
        }
    }]
)

# Encrypt sensitive data
sensitive_data = b"Confidential information"
encryption_result = tacs.encrypt_resource(
    data=sensitive_data,
    policy_id=policy_result["policy_id"],
    resource_type="text"
)

# Try to access during business hours
access_result = tacs.access_resource(
    resource_id=encryption_result["resource_id"],
    context={
        "current_time": datetime.now().replace(hour=14),  # 2 PM
        "identity": {"id": "user123", "roles": ["employee"]}
    }
)

if access_result["granted"]:
    print("Access granted! Data:", access_result["data"])
else:
    print("Access denied. Reason:", access_result["reason"])
```

### Example Implementations

See the [usage_example.py](usage_example.py) file for complete implementation examples.

## Technical Documentation

For detailed technical documentation on each component and method, refer to the docstrings within the code or generate documentation using:

```bash
pdoc --html temporal_access_control.py
```
