"""
Example usage of the Temporal Access Control System.

This script demonstrates how to create policies, encrypt resources, and control access
based on temporal, spatial, and astronomical conditions.
"""

import os
import base64
import json
from datetime import datetime, timedelta
import hashlib

from src.revenue.temporal_access.temporal_access_control import (
    TemporalAccessControlSystem,
    AccessCondition,
    TemporalAccessPolicy
)
from src.revenue.quantum_ledger.quantum_entangled_ledger import QuantumEntangledLedger


def create_simple_time_policy():
    """Create a simple time-based policy."""
    # Initialize the system
    tacs = TemporalAccessControlSystem()
    
    # Create a temporal condition for business hours
    conditions = [{
        "condition_type": "temporal",
        "parameters": {
            "hours_of_day": list(range(9, 18)),  # 9 AM to 6 PM
            "days_of_week": [0, 1, 2, 3, 4]  # Monday to Friday
        },
        "description": "Business hours (9 AM - 6 PM, Monday to Friday)"
    }]
    
    # Create the policy
    result = tacs.create_policy(
        name="Business Hours Policy",
        conditions=conditions,
        description="Allow access only during business hours"
    )
    
    print(f"Created policy: {result}")
    
    # Create some sample data
    sensitive_data = b"This is confidential business information"
    
    # Encrypt the data with the policy
    encryption_result = tacs.encrypt_resource(
        data=sensitive_data,
        policy_id=result["policy_id"],
        resource_type="text"
    )
    
    print(f"Encrypted resource: {encryption_result}")
    
    # Try to access during business hours
    business_hours_context = {
        "current_time": datetime.now().replace(hour=14, minute=30),  # 2:30 PM
        "identity": {"id": "employee123", "roles": ["employee"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=business_hours_context
    )
    
    print(f"Access during business hours: {access_result['granted']}")
    if access_result['granted']:
        print(f"Accessed data: {access_result['data']}")
    
    # Try to access outside business hours
    after_hours_context = {
        "current_time": datetime.now().replace(hour=22, minute=30),  # 10:30 PM
        "identity": {"id": "employee123", "roles": ["employee"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=after_hours_context
    )
    
    print(f"Access after hours: {access_result['granted']}")
    print(f"Reason: {access_result['reason']}")
    
    return tacs


def create_location_based_policy():
    """Create a location-based access policy."""
    tacs = TemporalAccessControlSystem()
    
    # Create a spatial condition for office location
    office_coordinates = [40.7128, -74.0060]  # Example: New York City coordinates
    conditions = [{
        "condition_type": "spatial",
        "parameters": {
            "location": office_coordinates,
            "radius": 500  # 500 meters radius
        },
        "description": "Must be within 500m of the office"
    }]
    
    # Create the policy
    result = tacs.create_policy(
        name="Office Location Policy",
        conditions=conditions,
        description="Allow access only when in the office"
    )
    
    # Create some sample data
    sensitive_data = b"Internal office documents"
    
    # Encrypt the data with the policy
    encryption_result = tacs.encrypt_resource(
        data=sensitive_data,
        policy_id=result["policy_id"],
        resource_type="document"
    )
    
    # Try to access from the office
    in_office_context = {
        "current_location": [40.7129, -74.0061],  # Within office radius
        "identity": {"id": "employee456", "roles": ["employee"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=in_office_context
    )
    
    print(f"Access from office: {access_result['granted']}")
    
    # Try to access from outside the office
    remote_context = {
        "current_location": [42.3601, -71.0589],  # Boston coordinates
        "identity": {"id": "employee456", "roles": ["employee"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=remote_context
    )
    
    print(f"Access from remote location: {access_result['granted']}")
    print(f"Reason: {access_result['reason']}")
    
    return tacs


def create_role_based_policy():
    """Create a role-based access policy."""
    tacs = TemporalAccessControlSystem()
    
    # Create an identity condition for admin access
    conditions = [{
        "condition_type": "identity",
        "parameters": {
            "allowed_roles": ["admin", "security_officer"]
        },
        "description": "Must have admin or security officer role"
    }]
    
    # Create the policy
    result = tacs.create_policy(
        name="Admin Access Policy",
        conditions=conditions,
        description="Allow access only for administrators"
    )
    
    # Create some sample data
    sensitive_data = b"System configuration data"
    
    # Encrypt the data with the policy
    encryption_result = tacs.encrypt_resource(
        data=sensitive_data,
        policy_id=result["policy_id"],
        resource_type="configuration"
    )
    
    # Try to access as admin
    admin_context = {
        "identity": {"id": "admin001", "roles": ["admin"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=admin_context
    )
    
    print(f"Access as admin: {access_result['granted']}")
    
    # Try to access as regular employee
    employee_context = {
        "identity": {"id": "employee789", "roles": ["employee"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=employee_context
    )
    
    print(f"Access as employee: {access_result['granted']}")
    print(f"Reason: {access_result['reason']}")
    
    return tacs


def create_compound_policy():
    """Create a compound policy with multiple conditions."""
    tacs = TemporalAccessControlSystem()
    
    # Create a policy with multiple conditions (business hours AND in office)
    conditions = [
        {
            "condition_type": "temporal",
            "parameters": {
                "hours_of_day": list(range(9, 18)),  # 9 AM to 6 PM
                "days_of_week": [0, 1, 2, 3, 4]  # Monday to Friday
            },
            "description": "Business hours (9 AM - 6 PM, Monday to Friday)"
        },
        {
            "condition_type": "spatial",
            "parameters": {
                "location": [40.7128, -74.0060],  # NYC coordinates
                "radius": 500  # 500 meters radius
            },
            "description": "Must be within 500m of the office"
        },
        {
            "condition_type": "identity",
            "parameters": {
                "allowed_roles": ["financial_analyst", "accountant", "finance_manager"]
            },
            "description": "Must have a finance role"
        }
    ]
    
    # Create the policy with AND operator (all conditions must be met)
    result = tacs.create_policy(
        name="Secure Financial Data Policy",
        conditions=conditions,
        logical_operator="AND",
        description="Strict policy for financial data access"
    )
    
    # Create some sample data
    sensitive_data = b"Quarterly financial reports and projections"
    
    # Encrypt the data with the policy
    encryption_result = tacs.encrypt_resource(
        data=sensitive_data,
        policy_id=result["policy_id"],
        resource_type="financial_report"
    )
    
    # Try to access with all conditions met
    valid_context = {
        "current_time": datetime.now().replace(hour=14, minute=30),  # 2:30 PM
        "current_location": [40.7129, -74.0061],  # Within office radius
        "identity": {"id": "analyst001", "roles": ["financial_analyst"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=valid_context
    )
    
    print(f"Access with all conditions met: {access_result['granted']}")
    
    # Try to access with one condition not met (outside business hours)
    invalid_time_context = {
        "current_time": datetime.now().replace(hour=22, minute=30),  # 10:30 PM
        "current_location": [40.7129, -74.0061],  # Within office radius
        "identity": {"id": "analyst001", "roles": ["financial_analyst"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=invalid_time_context
    )
    
    print(f"Access outside business hours: {access_result['granted']}")
    print(f"Reason: {access_result['reason']}")
    
    return tacs


def integrate_with_quantum_ledger():
    """Demonstrate integration with the Quantum Entangled Ledger."""
    # Initialize components
    quantum_ledger = QuantumEntangledLedger()
    tacs = TemporalAccessControlSystem(quantum_ledger=quantum_ledger)
    
    # Create a quantum verification condition
    # First, create a document in the quantum ledger
    document_content = "Highly confidential legal agreement"
    document_hash = hashlib.sha256(document_content.encode()).hexdigest()
    
    entanglement_result = quantum_ledger.entangle_document(
        document_hash=document_hash,
        spatial_coordinates=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        metadata={"type": "legal_document"}
    )
    
    record_id = entanglement_result["record"]["record_id"]
    
    # Create a policy with quantum verification
    conditions = [{
        "condition_type": "quantum",
        "parameters": {
            "verification_type": "document",
            "record_id": record_id
        },
        "description": "Document must be verified by quantum ledger"
    }]
    
    policy_result = tacs.create_policy(
        name="Quantum Verified Document Policy",
        conditions=conditions,
        description="Access requires quantum verification"
    )
    
    # Encrypt some data with this policy
    sensitive_data = b"Content protected by quantum verification"
    
    encryption_result = tacs.encrypt_resource(
        data=sensitive_data,
        policy_id=policy_result["policy_id"],
        resource_type="legal_document"
    )
    
    # Try to access with valid quantum verification
    valid_context = {
        "document_hash": document_hash,
        "identity": {"id": "legal001", "roles": ["legal"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=valid_context
    )
    
    print(f"Access with valid quantum verification: {access_result['granted']}")
    
    # Try to access with invalid quantum verification
    invalid_context = {
        "document_hash": hashlib.sha256(b"different document").hexdigest(),
        "identity": {"id": "legal001", "roles": ["legal"]}
    }
    
    access_result = tacs.access_resource(
        resource_id=encryption_result["resource_id"],
        context=invalid_context
    )
    
    print(f"Access with invalid quantum verification: {access_result['granted']}")
    print(f"Reason: {access_result['reason']}")
    
    return tacs


def main():
    """Run the demonstration examples."""
    print("\n=== Time-Based Access Control ===")
    create_simple_time_policy()
    
    print("\n=== Location-Based Access Control ===")
    create_location_based_policy()
    
    print("\n=== Role-Based Access Control ===")
    create_role_based_policy()
    
    print("\n=== Compound Policy Access Control ===")
    create_compound_policy()
    
    print("\n=== Quantum Ledger Integration ===")
    integrate_with_quantum_ledger()


if __name__ == "__main__":
    main()
