# Documentation for test_temporal_access_control.py

```python
"""
Unit tests for the Temporal Access Control System.
"""

import unittest
from datetime import datetime, timedelta
import base64
import hashlib
import json
import os
import sys

# Add the parent directory to the path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.revenue.temporal_access.temporal_access_control import (
    AccessCondition,
    TemporalAccessPolicy,
    TemporalAccessControlSystem
)


class TestAccessCondition(unittest.TestCase):
    """Test cases for AccessCondition class."""
    
    def test_temporal_condition(self):
        """Test temporal condition evaluation."""
        # Create a business hours condition
        condition = AccessCondition(
            condition_type="temporal",
            parameters={
                "hours_of_day": list(range(9, 18)),  # 9 AM to 6 PM
                "days_of_week": [0, 1, 2, 3, 4]      # Monday to Friday
            },
            description="Business hours only"
        )
        
        # Test during business hours (2 PM on a Wednesday)
        business_time = datetime.now().replace(hour=14, minute=0)
        # Make sure it's a weekday (0-4 = Monday-Friday)
        while business_time.weekday() > 4:
            business_time = business_time + timedelta(days=1)
            
        result, reason = condition.evaluate({"current_time": business_time})
        self.assertTrue(result, f"Should allow access during business hours: {reason}")
        
        # Test outside business hours (10 PM)
        after_hours = business_time.replace(hour=22)
        result, reason = condition.evaluate({"current_time": after_hours})
        self.assertFalse(result, "Should deny access outside business hours")
        
        # Test on weekend
        weekend = business_time
        # Move to Saturday or Sunday
        while weekend.weekday() < 5:
            weekend = weekend + timedelta(days=1)
            
        result, reason = condition.evaluate({"current_time": weekend})
        self.assertFalse(result, "Should deny access on weekends")
        
    def test_identity_condition(self):
        """Test identity condition evaluation."""
        # Create an admin-only condition
        condition = AccessCondition(
            condition_type="identity",
            parameters={
                "allowed_roles": ["admin", "security_officer"]
            },
            description="Admin access only"
        )
        
        # Test with admin role
        admin_context = {"identity": {"id": "user123", "roles": ["admin"]}}
        result, reason = condition.evaluate(admin_context)
        self.assertTrue(result, "Should allow access for admin role")
        
        # Test with security officer role
        security_context = {"identity": {"id": "user456", "roles": ["security_officer"]}}
        result, reason = condition.evaluate(security_context)
        self.assertTrue(result, "Should allow access for security_officer role")
        
        # Test with employee role
        employee_context = {"identity": {"id": "user789", "roles": ["employee"]}}
        result, reason = condition.evaluate(employee_context)
        self.assertFalse(result, "Should deny access for employee role")
        
    def test_compound_condition(self):
        """Test compound condition evaluation."""
        # Create a compound condition (admin OR during business hours)
        condition = AccessCondition(
            condition_type="compound",
            parameters={
                "operator": "OR",
                "conditions": [
                    {
                        "condition_type": "identity",
                        "parameters": {"allowed_roles": ["admin"]}
                    },
                    {
                        "condition_type": "temporal",
                        "parameters": {
                            "hours_of_day": list(range(9, 18)),
                            "days_of_week": [0, 1, 2, 3, 4]
                        }
                    }
                ]
            },
            description="Admin OR business hours"
        )
        
        # Test with admin outside business hours
        admin_after_hours = {
            "identity": {"id": "admin1", "roles": ["admin"]},
            "current_time": datetime.now().replace(hour=20)  # 8 PM
        }
        result, reason = condition.evaluate(admin_after_hours)
        self.assertTrue(result, "Should allow admin after hours")
        
        # Test with employee during business hours
        business_time = datetime.now().replace(hour=14, minute=0)
        # Make sure it's a weekday
        while business_time.weekday() > 4:
            business_time = business_time + timedelta(days=1)
            
        employee_business_hours = {
            "identity": {"id": "emp1", "roles": ["employee"]},
            "current_time": business_time
        }
        result, reason = condition.evaluate(employee_business_hours)
        self.assertTrue(result, "Should allow employee during business hours")
        
        # Test with employee outside business hours
        employee_after_hours = {
            "identity": {"id": "emp1", "roles": ["employee"]},
            "current_time": datetime.now().replace(hour=20)  # 8 PM
        }
        result, reason = condition.evaluate(employee_after_hours)
        self.assertFalse(result, "Should deny employee after hours")


class TestTemporalAccessPolicy(unittest.TestCase):
    """Test cases for TemporalAccessPolicy class."""
    
    def test_policy_with_and_operator(self):
        """Test policy with AND operator."""
        # Create conditions
        time_condition = AccessCondition(
            condition_type="temporal",
            parameters={"hours_of_day": list(range(9, 18))},
            description="Business hours"
        )
        
        role_condition = AccessCondition(
            condition_type="identity",
            parameters={"allowed_roles": ["admin", "manager"]},
            description="Admin or manager"
        )
        
        # Create policy with AND operator
        policy = TemporalAccessPolicy(
            name="Secure Admin Policy",
            conditions=[time_condition, role_condition],
            logical_operator="AND",
            description="Admin/manager during business hours"
        )
        
        # Test with admin during business hours
        admin_business_hours = {
            "identity": {"id": "admin1", "roles": ["admin"]},
            "current_time": datetime.now().replace(hour=14)  # 2 PM
        }
        result, reason = policy.evaluate(admin_business_hours)
        self.assertTrue(result, "Should allow admin during business hours")
        
        # Test with admin after hours
        admin_after_hours = {
            "identity": {"id": "admin1", "roles": ["admin"]},
            "current_time": datetime.now().replace(hour=20)  # 8 PM
        }
        result, reason = policy.evaluate(admin_after_hours)
        self.assertFalse(result, "Should deny admin after hours")
        
        # Test with employee during business hours
        employee_business_hours = {
            "identity": {"id": "emp1", "roles": ["employee"]},
            "current_time": datetime.now().replace(hour=14)  # 2 PM
        }
        result, reason = policy.evaluate(employee_business_hours)
        self.assertFalse(result, "Should deny employee during business hours")
        
    def test_policy_with_or_operator(self):
        """Test policy with OR operator."""
        # Create conditions
        weekend_condition = AccessCondition(
            condition_type="temporal",
            parameters={"days_of_week": [5, 6]},  # Saturday, Sunday
            description="Weekends only"
        )
        
        vip_condition = AccessCondition(
            condition_type="identity",
            parameters={"allowed_roles": ["vip"]},
            description="VIP users only"
        )
        
        # Create policy with OR operator
        policy = TemporalAccessPolicy(
            name="VIP or Weekend Policy",
            conditions=[weekend_condition, vip_condition],
            logical_operator="OR",
            description="VIP users anytime or anyone on weekends"
        )
        
        # Find a weekend day
        now = datetime.now()
        weekend = now
        while weekend.weekday() < 5:  # 5 = Saturday, 6 = Sunday
            weekend = weekend + timedelta(days=1)
        
        # Test with regular user on weekend
        regular_weekend = {
            "identity": {"id": "user1", "roles": ["user"]},
            "current_time": weekend
        }
        result, reason = policy.evaluate(regular_weekend)
        self.assertTrue(result, "Should allow regular user on weekend")
        
        # Test with VIP on weekday
        # Find a weekday
        weekday = now
        while weekday.weekday() >= 5:  # 0-4 = Monday-Friday
            weekday = weekday + timedelta(days=1)
            
        vip_weekday = {
            "identity": {"id": "vip1", "roles": ["vip"]},
            "current_time": weekday
        }
        result, reason = policy.evaluate(vip_weekday)
        self.assertTrue(result, "Should allow VIP on weekday")
        
        # Test with regular user on weekday
        regular_weekday = {
            "identity": {"id": "user1", "roles": ["user"]},
            "current_time": weekday
        }
        result, reason = policy.evaluate(regular_weekday)
        self.assertFalse(result, "Should deny regular user on weekday")


class TestTemporalAccessControlSystem(unittest.TestCase):
    """Test cases for TemporalAccessControlSystem class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.tacs = TemporalAccessControlSystem()
        
    def test_create_policy(self):
        """Test policy creation."""
        result = self.tacs.create_policy(
            name="Test Policy",
            conditions=[{
                "condition_type": "temporal",
                "parameters": {"hours_of_day": list(range(9, 18))}
            }],
            description="Test policy"
        )
        
        self.assertTrue(result["success"])
        self.assertIn(result["policy_id"], self.tacs.policies)
        
    def test_encrypt_and_access_resource(self):
        """Test resource encryption and access."""
        # Create a policy
        policy_result = self.tacs.create_policy(
            name="Test Access Policy",
            conditions=[{
                "condition_type": "identity",
                "parameters": {"allowed_roles": ["tester"]}
            }],
            description="Test access policy"
        )
        
        # Encrypt some data
        test_data = b"This is test data"
        encrypt_result = self.tacs.encrypt_resource(
            data=test_data,
            policy_id=policy_result["policy_id"],
            resource_type="test"
        )
        
        self.assertTrue(encrypt_result["success"])
        self.assertIn(encrypt_result["resource_id"], self.tacs.resources)
        
        # Try to access with allowed role
        allowed_context = {
            "identity": {"id": "test_user", "roles": ["tester"]}
        }
        
        access_result = self.tacs.access_resource(
            resource_id=encrypt_result["resource_id"],
            context=allowed_context
        )
        
        self.assertTrue(access_result["granted"])
        self.assertEqual(access_result["data"], test_data)
        
        # Try to access with disallowed role
        disallowed_context = {
            "identity": {"id": "other_user", "roles": ["other"]}
        }
        
        access_result = self.tacs.access_resource(
            resource_id=encrypt_result["resource_id"],
            context=disallowed_context
        )
        
        self.assertFalse(access_result["granted"])
        
    def test_access_logs(self):
        """Test access logging."""
        # Create a policy
        policy_result = self.tacs.create_policy(
            name="Log Test Policy",
            conditions=[{
                "condition_type": "identity",
                "parameters": {"allowed_roles": ["tester"]}
            }]
        )
        
        # Encrypt some data
        test_data = b"Log test data"
        encrypt_result = self.tacs.encrypt_resource(
            data=test_data,
            policy_id=policy_result["policy_id"],
            resource_type="test"
        )
        
        resource_id = encrypt_result["resource_id"]
        
        # Make some access attempts
        self.tacs.access_resource(
            resource_id=resource_id,
            context={"identity": {"id": "user1", "roles": ["tester"]}}
        )
        
        self.tacs.access_resource(
            resource_id=resource_id,
            context={"identity": {"id": "user2", "roles": ["other"]}}
        )
        
        # Get logs
        logs = self.tacs.get_access_logs(resource_id=resource_id)
        
        self.assertEqual(len(logs), 2, "Should have 2 log entries")
        self.assertTrue(logs[0]["granted"], "First access should be granted")
        self.assertFalse(logs[1]["granted"], "Second access should be denied")
        
        # Test filtering
        granted_logs = self.tacs.get_access_logs(
            resource_id=resource_id, 
            granted_only=True
        )
        
        self.assertEqual(len(granted_logs), 1, "Should have 1 granted log entry")
        self.assertEqual(granted_logs[0]["user_id"], "user1")
        
    def test_update_resource_policy(self):
        """Test updating a resource's policy."""
        # Create two policies
        policy1_result = self.tacs.create_policy(
            name="Original Policy",
            conditions=[{
                "condition_type": "identity",
                "parameters": {"allowed_roles": ["role1"]}
            }]
        )
        
        policy2_result = self.tacs.create_policy(
            name="New Policy",
            conditions=[{
                "condition_type": "identity",
                "parameters": {"allowed_roles": ["role2"]}
            }]
        )
        
        # Encrypt with first policy
        test_data = b"Policy update test"
        encrypt_result = self.tacs.encrypt_resource(
            data=test_data,
            policy_id=policy1_result["policy_id"],
            resource_type="test"
        )
        
        resource_id = encrypt_result["resource_id"]
        
        # Verify role1 can access but role2 cannot
        role1_context = {"identity": {"id": "user1", "roles": ["role1"]}}
        role2_context = {"identity": {"id": "user2", "roles": ["role2"]}}
        
        self.assertTrue(
            self.tacs.access_resource(resource_id, role1_context)["granted"],
            "role1 should have access initially"
        )
        
        self.assertFalse(
            self.tacs.access_resource(resource_id, role2_context)["granted"],
            "role2 should not have access initially"
        )
        
        # Update the policy
        update_result = self.tacs.update_resource_policy(
            resource_id=resource_id,
            policy_id=policy2_result["policy_id"]
        )
        
        self.assertTrue(update_result["success"])
        
        # Verify role2 can now access but role1 cannot
        self.assertFalse(
            self.tacs.access_resource(resource_id, role1_context)["granted"],
            "role1 should not have access after update"
        )
        
        self.assertTrue(
            self.tacs.access_resource(resource_id, role2_context)["granted"],
            "role2 should have access after update"
        )


if __name__ == "__main__":
    unittest.main()

```