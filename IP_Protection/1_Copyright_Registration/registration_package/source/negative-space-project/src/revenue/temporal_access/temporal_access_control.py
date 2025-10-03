"""
Temporal Access Control System (TACS)

This module implements a sophisticated access control system that grants or restricts
access to digital assets based on temporal, spatial, and astronomical conditions.
It creates time-locked content that can only be accessed under specific conditions,
providing a new level of security and controlled data disclosure.
"""

import uuid
import json
import hashlib
import time
import base64
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import hmac

# For astronomical calculations
try:
    import ephem  # You may need to install this: pip install pyephem
except ImportError:
    ephem = None

# Import from other modules
from ...negative_mapping.spatial_signature_generator import SpatialSignatureGenerator
from ..mnemonic_architecture.mnemonic_data_architecture import MnemonicDataArchitecture
from ..quantum_ledger.quantum_entangled_ledger import QuantumEntangledLedger, QuantumEntangledRecord


class AccessCondition:
    """
    Defines a condition under which access to a resource is granted.
    This can include temporal, spatial, astronomical, or identity-based conditions.
    """
    
    CONDITION_TYPES = [
        "temporal", "spatial", "astronomical", "identity", "quantum", "compound"
    ]
    
    def __init__(self,
                condition_type: str,
                parameters: Dict[str, Any],
                description: str = None):
        """
        Initialize an access condition.
        
        Args:
            condition_type: Type of condition (temporal, spatial, astronomical, etc.)
            parameters: Parameters specific to the condition type
            description: Human-readable description of the condition
        """
        if condition_type not in self.CONDITION_TYPES:
            raise ValueError(f"Unsupported condition type: {condition_type}")
            
        self.condition_id = str(uuid.uuid4())
        self.condition_type = condition_type
        self.parameters = parameters
        self.description = description or self._generate_description()
        
    def _generate_description(self) -> str:
        """
        Generate a human-readable description of the condition.
        
        Returns:
            Description string
        """
        if self.condition_type == "temporal":
            start_time = self.parameters.get("start_time", "any time")
            end_time = self.parameters.get("end_time", "indefinitely")
            return f"Access allowed from {start_time} until {end_time}"
            
        elif self.condition_type == "spatial":
            location = self.parameters.get("location", "any location")
            radius = self.parameters.get("radius", 0)
            return f"Access allowed within {radius}m of {location}"
            
        elif self.condition_type == "astronomical":
            celestial_event = self.parameters.get("event", "unspecified event")
            return f"Access allowed during {celestial_event}"
            
        elif self.condition_type == "identity":
            identity_type = self.parameters.get("identity_type", "user")
            return f"Access allowed for authorized {identity_type}"
            
        elif self.condition_type == "quantum":
            return f"Access allowed with quantum verification"
            
        elif self.condition_type == "compound":
            return f"Access allowed under multiple conditions"
            
        return "Access condition"
        
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate if this condition is satisfied given the context.
        
        Args:
            context: Current execution context (time, location, identity, etc.)
            
        Returns:
            Tuple of (is_satisfied, reason)
        """
        if self.condition_type == "temporal":
            return self._evaluate_temporal(context)
            
        elif self.condition_type == "spatial":
            return self._evaluate_spatial(context)
            
        elif self.condition_type == "astronomical":
            return self._evaluate_astronomical(context)
            
        elif self.condition_type == "identity":
            return self._evaluate_identity(context)
            
        elif self.condition_type == "quantum":
            return self._evaluate_quantum(context)
            
        elif self.condition_type == "compound":
            return self._evaluate_compound(context)
            
        return (False, "Unknown condition type")
        
    def _evaluate_temporal(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate a temporal condition."""
        current_time = context.get("current_time", datetime.now().isoformat())
        
        if isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(current_time)
            except ValueError:
                return (False, "Invalid time format in context")
        
        # Check start time
        if "start_time" in self.parameters:
            start_time = self.parameters["start_time"]
            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time)
                except ValueError:
                    return (False, "Invalid start time format in condition")
                    
            if current_time < start_time:
                return (False, f"Current time {current_time} is before start time {start_time}")
        
        # Check end time
        if "end_time" in self.parameters:
            end_time = self.parameters["end_time"]
            if isinstance(end_time, str):
                try:
                    end_time = datetime.fromisoformat(end_time)
                except ValueError:
                    return (False, "Invalid end time format in condition")
                    
            if current_time > end_time:
                return (False, f"Current time {current_time} is after end time {end_time}")
        
        # Check specific days of week
        if "days_of_week" in self.parameters:
            allowed_days = self.parameters["days_of_week"]
            current_day = current_time.weekday()  # 0 = Monday, 6 = Sunday
            if current_day not in allowed_days:
                return (False, f"Current day {current_day} is not in allowed days {allowed_days}")
        
        # Check specific hours of day
        if "hours_of_day" in self.parameters:
            allowed_hours = self.parameters["hours_of_day"]
            current_hour = current_time.hour
            if current_hour not in allowed_hours:
                return (False, f"Current hour {current_hour} is not in allowed hours {allowed_hours}")
        
        return (True, "Temporal condition satisfied")
        
    def _evaluate_spatial(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate a spatial condition."""
        current_location = context.get("current_location")
        if not current_location:
            return (False, "No location information in context")
            
        if "location" not in self.parameters or "radius" not in self.parameters:
            return (False, "Incomplete spatial condition parameters")
            
        target_location = self.parameters["location"]
        allowed_radius = self.parameters["radius"]
        
        # Calculate distance (simplified - assumes [lat, lon] format)
        # In a real implementation, use proper geospatial calculations
        try:
            from math import radians, sin, cos, sqrt, atan2
            
            lat1, lon1 = radians(current_location[0]), radians(current_location[1])
            lat2, lon2 = radians(target_location[0]), radians(target_location[1])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            R = 6371000  # Earth radius in meters
            distance = R * c
            
            if distance <= allowed_radius:
                return (True, f"Within {distance:.2f}m of target location (allowed: {allowed_radius}m)")
            else:
                return (False, f"Distance of {distance:.2f}m exceeds allowed radius of {allowed_radius}m")
                
        except Exception as e:
            return (False, f"Error calculating spatial distance: {e}")
        
    def _evaluate_astronomical(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate an astronomical condition."""
        if not ephem:
            return (False, "Astronomical library not available")
            
        event_type = self.parameters.get("event")
        if not event_type:
            return (False, "No astronomical event specified")
            
        current_time = context.get("current_time", datetime.now())
        if isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(current_time)
            except ValueError:
                return (False, "Invalid time format in context")
                
        observer_location = context.get("current_location")
        if not observer_location:
            # Default to Greenwich if no location provided
            observer = ephem.Observer()
            observer.lon, observer.lat = '0.0', '51.5'  # Greenwich
        else:
            observer = ephem.Observer()
            observer.lon, observer.lat = str(observer_location[1]), str(observer_location[0])
            
        observer.date = ephem.Date(current_time)
        
        try:
            # Check for various astronomical events
            if event_type == "full_moon":
                prev_full = ephem.previous_full_moon(observer.date)
                next_full = ephem.next_full_moon(observer.date)
                days_since = observer.date - prev_full
                days_until = next_full - observer.date
                
                # If within 1 day of full moon
                if min(days_since, days_until) <= 1:
                    return (True, "Full moon condition satisfied")
                else:
                    return (False, f"Not a full moon. {days_until:.2f} days until next full moon")
                    
            elif event_type == "new_moon":
                prev_new = ephem.previous_new_moon(observer.date)
                next_new = ephem.next_new_moon(observer.date)
                days_since = observer.date - prev_new
                days_until = next_new - observer.date
                
                # If within 1 day of new moon
                if min(days_since, days_until) <= 1:
                    return (True, "New moon condition satisfied")
                else:
                    return (False, f"Not a new moon. {days_until:.2f} days until next new moon")
                    
            elif event_type == "solstice":
                return self._evaluate_astronomical(context)
                next_summer = ephem.next_summer_solstice(observer.date)
                prev_winter = ephem.previous_winter_solstice(observer.date)
                next_winter = ephem.next_winter_solstice(observer.date)
        from ...shared.astronomical_engine import AstronomicalCalculationEngine
        astro_engine = AstronomicalCalculationEngine()
        astronomical = context.get("astronomical", {})
        event = self.parameters.get("event")
        location = astronomical.get("location")
        timestamp = astronomical.get("timestamp", datetime.now().isoformat())
        # Use astronomical engine for real event detection
        if event:
            if event == "full_moon":
                is_full_moon = astro_engine.is_event_occurring("full_moon", timestamp, location)
                if is_full_moon:
                    return True, "Full moon event detected (astronomical engine)"
                else:
                    return False, "Full moon event not detected (astronomical engine)"
            elif event == "solar_eclipse":
                is_eclipse = astro_engine.is_event_occurring("solar_eclipse", timestamp, location)
                if is_eclipse:
                    return True, "Solar eclipse detected (astronomical engine)"
                else:
                    return False, "Solar eclipse not detected (astronomical engine)"
            # Add more event types as needed
        # Alignment check example
        alignment = self.parameters.get("alignment")
        if alignment:
            obj1 = alignment.get("object1")
            obj2 = alignment.get("object2")
            min_angle = alignment.get("min_angle", 0)
            max_angle = alignment.get("max_angle", 360)
            angle = astro_engine.get_alignment_angle(obj1, obj2, timestamp)
            if angle is not None and min_angle <= angle <= max_angle:
                return True, f"Alignment between {obj1} and {obj2} detected: {angle} degrees (astronomical engine)"
            else:
                return False, f"Alignment between {obj1} and {obj2} not in range (astronomical engine)"
        return False, f"Astronomical event '{event}' not detected (astronomical engine)"
                
        if aligned:
                    return (True, f"Planetary alignment condition satisfied")
        else:
                    return (False, f"Planets not aligned within {max_angle} degrees")
                    
            else:
        return (False, f"Unsupported astronomical event: {event_type}")
                
        except Exception as e:
        return (False, f"Error evaluating astronomical condition: {e}")
        
    def _evaluate_identity(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate an identity-based condition."""
        current_identity = context.get("identity")
        if not current_identity:
            return (False, "No identity information in context")
            
        identity_type = self.parameters.get("identity_type", "user")
        required_ids = self.parameters.get("allowed_ids", [])
        required_roles = self.parameters.get("allowed_roles", [])
        required_attributes = self.parameters.get("required_attributes", {})
        
        # Check ID match
        if required_ids and current_identity.get("id") not in required_ids:
            return (False, f"Identity ID not in allowed list")
            
        # Check role match
        if required_roles:
            current_roles = current_identity.get("roles", [])
            if not any(role in current_roles for role in required_roles):
                return (False, f"Identity has none of the required roles")
                
        # Check attributes
        if required_attributes:
            current_attributes = current_identity.get("attributes", {})
            for key, value in required_attributes.items():
                if key not in current_attributes or current_attributes[key] != value:
                    return (False, f"Identity missing required attribute: {key}={value}")
                    
        return (True, "Identity condition satisfied")
        
    def _evaluate_quantum(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate a quantum verification condition."""
        quantum_ledger = context.get("quantum_ledger")
        if not quantum_ledger:
            return (False, "No quantum ledger available in context")
            
        verification_type = self.parameters.get("verification_type", "standard")
        
        if verification_type == "document":
            # Verify a document hash against a record
            document_hash = context.get("document_hash")
            record_id = self.parameters.get("record_id")
            
            if not document_hash or not record_id:
                return (False, "Missing document_hash or record_id for quantum verification")
                
            result = quantum_ledger.verify_document(document_hash, record_id)
            if result.get("verified", False):
                return (True, "Quantum document verification successful")
            else:
                return (False, f"Quantum verification failed: {result.get('reason', 'Unknown reason')}")
                
        elif verification_type == "historical":
            # Verify a historical record
            record_id = self.parameters.get("record_id")
            claimed_date = self.parameters.get("claimed_date")
            
            if not record_id or not claimed_date:
                return (False, "Missing record_id or claimed_date for historical verification")
                
            result = quantum_ledger.verify_historical_record(record_id, claimed_date)
            if result.get("verified", False):
                return (True, f"Historical verification successful (probability: {result.get('temporal_probability', 0):.2f})")
            else:
                return (False, f"Historical verification failed: {result.get('reason', 'Unknown reason')}")
                
        else:
            return (False, f"Unsupported quantum verification type: {verification_type}")
        
    def _evaluate_compound(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate a compound condition (multiple conditions with logical operators)."""
        subconditions = self.parameters.get("conditions", [])
        operator = self.parameters.get("operator", "AND").upper()
        
        if not subconditions:
            return (False, "No subconditions in compound condition")
            
        # Convert subconditions to AccessCondition objects if they aren't already
        condition_objects = []
        for subcond in subconditions:
            if isinstance(subcond, AccessCondition):
                condition_objects.append(subcond)
            elif isinstance(subcond, dict):
                try:
                    condition_objects.append(AccessCondition(
                        subcond.get("condition_type"),
                        subcond.get("parameters", {}),
                        subcond.get("description")
                    ))
                except Exception as e:
                    return (False, f"Invalid subcondition: {e}")
            else:
                return (False, f"Invalid subcondition type: {type(subcond)}")
        
        # Evaluate all subconditions
        results = [cond.evaluate(context) for cond in condition_objects]
        
        if operator == "AND":
            # All conditions must be satisfied
            all_satisfied = all(result[0] for result in results)
            if all_satisfied:
                return (True, "All subconditions satisfied")
            else:
                failed_reasons = [result[1] for result in results if not result[0]]
                return (False, f"AND condition failed: {'; '.join(failed_reasons)}")
                
        elif operator == "OR":
            # At least one condition must be satisfied
            any_satisfied = any(result[0] for result in results)
            if any_satisfied:
                satisfied_indices = [i for i, result in enumerate(results) if result[0]]
                return (True, f"OR condition satisfied by subcondition(s): {satisfied_indices}")
            else:
                return (False, "None of the subconditions were satisfied")
                
        elif operator == "XOR":
            # Exactly one condition must be satisfied
            satisfied = [result[0] for result in results]
            if sum(satisfied) == 1:
                index = satisfied.index(True)
                return (True, f"XOR condition satisfied by subcondition {index}")
            else:
                return (False, f"XOR condition failed: {sum(satisfied)} conditions satisfied instead of 1")
                
        else:
            return (False, f"Unsupported logical operator: {operator}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "condition_id": self.condition_id,
            "condition_type": self.condition_type,
            "parameters": self.parameters,
            "description": self.description
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccessCondition':
        """Create from dictionary representation."""
        condition = cls(
            condition_type=data["condition_type"],
            parameters=data["parameters"],
            description=data.get("description")
        )
        condition.condition_id = data.get("condition_id", str(uuid.uuid4()))
        return condition


class TemporalAccessPolicy:
    """
    A policy that defines under what conditions access to a resource is granted.
    A policy consists of one or more conditions that must be satisfied.
    """
    
    def __init__(self,
                name: str,
                conditions: List[AccessCondition],
                logical_operator: str = "AND",
                description: str = None,
                metadata: Dict[str, Any] = None):
        """
        Initialize a temporal access policy.
        
        Args:
            name: Policy name
            conditions: List of access conditions
            logical_operator: How to combine conditions ("AND", "OR", "XOR")
            description: Human-readable description
            metadata: Additional metadata
        """
        self.policy_id = str(uuid.uuid4())
        self.name = name
        self.conditions = conditions
        self.logical_operator = logical_operator.upper()
        self.description = description or f"Access policy: {name}"
        self.metadata = metadata or {}
        self.creation_time = datetime.now().isoformat()
        
        # Validate operator
        if self.logical_operator not in ["AND", "OR", "XOR"]:
            raise ValueError(f"Unsupported logical operator: {logical_operator}")
        
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate if this policy is satisfied given the context.
        
        Args:
            context: Current execution context (time, location, identity, etc.)
            
        Returns:
            Tuple of (is_satisfied, reason)
        """
        if not self.conditions:
            return (False, "Policy has no conditions")
            
        # Evaluate all conditions
        results = [condition.evaluate(context) for condition in self.conditions]
        
        if self.logical_operator == "AND":
            # All conditions must be satisfied
            all_satisfied = all(result[0] for result in results)
            if all_satisfied:
                return (True, "All conditions satisfied")
            else:
                failed_reasons = [result[1] for result in results if not result[0]]
                return (False, f"Policy failed: {'; '.join(failed_reasons)}")
                
        elif self.logical_operator == "OR":
            # At least one condition must be satisfied
            any_satisfied = any(result[0] for result in results)
            if any_satisfied:
                satisfied_indices = [i for i, result in enumerate(results) if result[0]]
                satisfied_conditions = [self.conditions[i].description for i in satisfied_indices]
                return (True, f"Policy satisfied by condition(s): {', '.join(satisfied_conditions)}")
            else:
                return (False, "None of the policy conditions were satisfied")
                
        elif self.logical_operator == "XOR":
            # Exactly one condition must be satisfied
            satisfied = [result[0] for result in results]
            if sum(satisfied) == 1:
                index = satisfied.index(True)
                return (True, f"Policy satisfied by condition: {self.conditions[index].description}")
            else:
                return (False, f"Policy failed: {sum(satisfied)} conditions satisfied instead of 1")
                
        return (False, f"Unsupported logical operator: {self.logical_operator}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "logical_operator": self.logical_operator,
            "description": self.description,
            "metadata": self.metadata,
            "creation_time": self.creation_time
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalAccessPolicy':
        """Create from dictionary representation."""
        conditions = [
            AccessCondition.from_dict(condition_data) 
            for condition_data in data["conditions"]
        ]
        
        policy = cls(
            name=data["name"],
            conditions=conditions,
            logical_operator=data["logical_operator"],
            description=data.get("description"),
            metadata=data.get("metadata", {})
        )
        
        policy.policy_id = data["policy_id"]
        policy.creation_time = data["creation_time"]
        
        return policy


class EncryptedResource:
    """
    Represents an encrypted resource that can only be accessed when
    its associated temporal access policy is satisfied.
    """
    
    def __init__(self,
                resource_id: str,
                encrypted_data: bytes,
                policy: TemporalAccessPolicy,
                resource_type: str,
                metadata: Dict[str, Any] = None):
        """
        Initialize an encrypted resource.
        
        Args:
            resource_id: Unique identifier for the resource
            encrypted_data: The encrypted data
            policy: The temporal access policy
            resource_type: Type of resource (file, text, image, etc.)
            metadata: Additional metadata
        """
        self.resource_id = resource_id
        self.encrypted_data = encrypted_data
        self.policy = policy
        self.resource_type = resource_type
        self.metadata = metadata or {}
        self.creation_time = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "resource_id": self.resource_id,
            "encrypted_data": base64.b64encode(self.encrypted_data).decode('utf-8'),
            "policy": self.policy.to_dict(),
            "resource_type": self.resource_type,
            "metadata": self.metadata,
            "creation_time": self.creation_time
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedResource':
        """Create from dictionary representation."""
        policy = TemporalAccessPolicy.from_dict(data["policy"])
        encrypted_data = base64.b64decode(data["encrypted_data"])
        
        resource = cls(
            resource_id=data["resource_id"],
            encrypted_data=encrypted_data,
            policy=policy,
            resource_type=data["resource_type"],
            metadata=data.get("metadata", {})
        )
        
        resource.creation_time = data["creation_time"]
        
        return resource


class AccessLogEntry:
    """
    Represents a log entry for an access attempt to a resource.
    """
    
    def __init__(self,
                resource_id: str,
                user_id: str,
                access_time: str,
                granted: bool,
                context: Dict[str, Any],
                reason: str = None):
        """
        Initialize an access log entry.
        
        Args:
            resource_id: ID of the accessed resource
            user_id: ID of the user attempting access
            access_time: Timestamp of the access attempt
            granted: Whether access was granted
            context: Context of the access attempt
            reason: Reason for granting/denying access
        """
        self.log_id = str(uuid.uuid4())
        self.resource_id = resource_id
        self.user_id = user_id
        self.access_time = access_time
        self.granted = granted
        self.context = context
        self.reason = reason
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "log_id": self.log_id,
            "resource_id": self.resource_id,
            "user_id": self.user_id,
            "access_time": self.access_time,
            "granted": self.granted,
            "context": self.context,
            "reason": self.reason
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccessLogEntry':
        """Create from dictionary representation."""
        entry = cls(
            resource_id=data["resource_id"],
            user_id=data["user_id"],
            access_time=data["access_time"],
            granted=data["granted"],
            context=data["context"],
            reason=data.get("reason")
        )
        
        entry.log_id = data["log_id"]
        
        return entry


class TemporalAccessControlSystem:
    """
    Main class for the Temporal Access Control System.
    This system controls access to digital resources based on temporal,
    spatial, and astronomical conditions.
    """
    
    def __init__(self, 
                encryption_key: Optional[str] = None,
                quantum_ledger: Optional[QuantumEntangledLedger] = None,
                mnemonic_data_architecture: Optional[MnemonicDataArchitecture] = None):
        """
        Initialize the temporal access control system.
        
        Args:
            encryption_key: Master key for encryption/decryption
            quantum_ledger: Optional connection to quantum ledger
            mnemonic_data_architecture: Optional connection to mnemonic data architecture
        """
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.quantum_ledger = quantum_ledger
        self.mnemonic_architecture = mnemonic_data_architecture
        
        # Storage
        self.policies = {}  # policy_id -> TemporalAccessPolicy
        self.resources = {}  # resource_id -> EncryptedResource
        self.access_logs = []  # List of AccessLogEntry
        
        # For astronomical calculations
        self.observer = ephem.Observer() if ephem else None
        
    def _generate_encryption_key(self) -> str:
        """Generate a secure encryption key."""
        return base64.b64encode(os.urandom(32)).decode('utf-8')
        
    def create_policy(self,
                     name: str,
                     conditions: List[Dict[str, Any]],
                     logical_operator: str = "AND",
                     description: str = None,
                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new temporal access policy.
        
        Args:
            name: Policy name
            conditions: List of condition specifications
            logical_operator: How to combine conditions ("AND", "OR", "XOR")
            description: Human-readable description
            metadata: Additional metadata
            
        Returns:
            Policy creation result
        """
        try:
            # Convert condition specs to AccessCondition objects
            condition_objects = []
            for cond_spec in conditions:
                condition = AccessCondition(
                    condition_type=cond_spec["condition_type"],
                    parameters=cond_spec["parameters"],
                    description=cond_spec.get("description")
                )
                condition_objects.append(condition)
                
            # Create the policy
            policy = TemporalAccessPolicy(
                name=name,
                conditions=condition_objects,
                logical_operator=logical_operator,
                description=description,
                metadata=metadata
            )
            
            # Store the policy
            self.policies[policy.policy_id] = policy
            
            return {
                "success": True,
                "policy_id": policy.policy_id,
                "name": policy.name,
                "condition_count": len(policy.conditions)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def encrypt_resource(self,
                        data: bytes,
                        policy_id: str,
                        resource_type: str,
                        resource_id: Optional[str] = None,
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Encrypt a resource under a temporal access policy.
        
        Args:
            data: The data to encrypt
            policy_id: ID of the policy to apply
            resource_type: Type of resource (file, text, image, etc.)
            resource_id: Optional custom resource ID
            metadata: Additional metadata
            
        Returns:
            Encryption result
        """
        # Check if the policy exists
        if policy_id not in self.policies:
            return {
                "success": False,
                "error": f"Policy not found: {policy_id}"
            }
            
        policy = self.policies[policy_id]
        
        try:
            # Generate a resource ID if not provided
            if not resource_id:
                resource_id = str(uuid.uuid4())
                
            # Generate a unique encryption key for this resource
            # derived from the master key and resource ID
            resource_key = self._derive_resource_key(resource_id)
            
            # Encrypt the data
            encrypted_data = self._encrypt_data(data, resource_key)
            
            # Create the encrypted resource
            resource = EncryptedResource(
                resource_id=resource_id,
                encrypted_data=encrypted_data,
                policy=policy,
                resource_type=resource_type,
                metadata=metadata
            )
            
            # Store the resource
            self.resources[resource_id] = resource
            
            # If quantum ledger is available, record the encryption
            if self.quantum_ledger:
                # Create a hash of the encrypted data
                data_hash = hashlib.sha256(encrypted_data).hexdigest()
                
                # Record in quantum ledger
                ledger_result = self.quantum_ledger.entangle_document(
                    document_hash=data_hash,
                    spatial_coordinates=self._get_current_astronomical_coordinates(),
                    entanglement_level=4,
                    metadata={
                        "type": "temporal_access_resource",
                        "resource_id": resource_id,
                        "resource_type": resource_type,
                        "policy_id": policy_id
                    }
                )
                
                # Store the record ID in the resource metadata
                if ledger_result.get("success", False):
                    resource.metadata["quantum_record_id"] = ledger_result["record"]["record_id"]
                    self.resources[resource_id] = resource
            
            return {
                "success": True,
                "resource_id": resource_id,
                "policy_id": policy_id,
                "resource_type": resource_type,
                "creation_time": resource.creation_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def access_resource(self,
                       resource_id: str,
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to access a resource based on the current context.
        
        Args:
            resource_id: ID of the resource to access
            context: Current execution context (time, location, identity, etc.)
            
        Returns:
            Access result including decrypted data if access is granted
        """
        # Check if the resource exists
        if resource_id not in self.resources:
            return {
                "success": False,
                "error": f"Resource not found: {resource_id}"
            }
            
        resource = self.resources[resource_id]
        
        # Add current time to context if not provided
        if "current_time" not in context:
            context["current_time"] = datetime.now()
            
        # Add quantum ledger to context if available
        if self.quantum_ledger:
            context["quantum_ledger"] = self.quantum_ledger
            
        # Evaluate the policy
        policy_satisfied, reason = resource.policy.evaluate(context)
        
        # Get user ID from context
        user_id = context.get("identity", {}).get("id", "anonymous")
        
        # Create a log entry
        log_entry = AccessLogEntry(
            resource_id=resource_id,
            user_id=user_id,
            access_time=datetime.now().isoformat(),
            granted=policy_satisfied,
            context={key: value for key, value in context.items() if key != "quantum_ledger"},
            reason=reason
        )
        
        self.access_logs.append(log_entry)
        
        if not policy_satisfied:
            return {
                "success": False,
                "granted": False,
                "reason": reason,
                "resource_id": resource_id,
                "log_id": log_entry.log_id
            }
            
        try:
            # Derive the resource key
            resource_key = self._derive_resource_key(resource_id)
            
            # Decrypt the data
            decrypted_data = self._decrypt_data(resource.encrypted_data, resource_key)
            
            return {
                "success": True,
                "granted": True,
                "resource_id": resource_id,
                "resource_type": resource.resource_type,
                "data": decrypted_data,
                "log_id": log_entry.log_id
            }
            
        except Exception as e:
            log_entry.granted = False
            log_entry.reason = f"Decryption error: {e}"
            
            return {
                "success": False,
                "granted": False,
                "reason": f"Decryption error: {e}",
                "resource_id": resource_id,
                "log_id": log_entry.log_id
            }
            
    def get_access_logs(self, 
                       resource_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       granted_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get access logs with optional filtering.
        
        Args:
            resource_id: Filter by resource ID
            user_id: Filter by user ID
            start_time: Filter by start time (inclusive)
            end_time: Filter by end time (inclusive)
            granted_only: Only include granted access attempts
            
        Returns:
            List of access log entries
        """
        filtered_logs = self.access_logs
        
        # Filter by resource ID
        if resource_id:
            filtered_logs = [log for log in filtered_logs if log.resource_id == resource_id]
            
        # Filter by user ID
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            
        # Filter by start time
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.access_time >= start_time]
            
        # Filter by end time
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.access_time <= end_time]
            
        # Filter by granted status
        if granted_only:
            filtered_logs = [log for log in filtered_logs if log.granted]
            
        return [log.to_dict() for log in filtered_logs]
        
    def update_resource_policy(self,
                             resource_id: str,
                             policy_id: str) -> Dict[str, Any]:
        """
        Update the policy associated with a resource.
        
        Args:
            resource_id: ID of the resource to update
            policy_id: ID of the new policy to apply
            
        Returns:
            Update result
        """
        # Check if the resource exists
        if resource_id not in self.resources:
            return {
                "success": False,
                "error": f"Resource not found: {resource_id}"
            }
            
        # Check if the policy exists
        if policy_id not in self.policies:
            return {
                "success": False,
                "error": f"Policy not found: {policy_id}"
            }
            
        resource = self.resources[resource_id]
        policy = self.policies[policy_id]
        
        # Update the resource
        resource.policy = policy
        self.resources[resource_id] = resource
        
        return {
            "success": True,
            "resource_id": resource_id,
            "policy_id": policy_id
        }
        
    def _derive_resource_key(self, resource_id: str) -> bytes:
        """
        Derive a unique encryption key for a resource from the master key.
        
        Args:
            resource_id: ID of the resource
            
        Returns:
            Derived key
        """
        # Use HMAC to derive a unique key for this resource
        return hmac.new(
            self.encryption_key.encode(),
            resource_id.encode(),
            hashlib.sha256
        ).digest()
        
    def _encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """
        Encrypt data with AES-GCM.
        
        Args:
            data: Data to encrypt
            key: Encryption key
            
        Returns:
            Encrypted data
        """
        try:
            # Try to use cryptography if available
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                
                # Generate a random 96-bit IV (12 bytes)
                iv = os.urandom(12)
                
                # Encrypt the data
                aesgcm = AESGCM(key)
                ciphertext = aesgcm.encrypt(iv, data, None)
                
                # Prepend the IV to the ciphertext
                return iv + ciphertext
            except ImportError:
                # Fallback encryption
                pass
            
            # Fallback to a simple XOR encryption for demo purposes
            # DO NOT USE THIS IN PRODUCTION
            import random
            random.seed(int.from_bytes(key, byteorder='big'))
            keystream = bytes([random.randint(0, 255) for _ in range(len(data))])
            return bytes([a ^ b for a, b in zip(data, keystream)])
            
        except Exception as e:
            raise ValueError(f"Encryption error: {e}")
        
    def _decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """
        Decrypt data with AES-GCM.
        
        Args:
            encrypted_data: Data to decrypt
            key: Decryption key
            
        Returns:
            Decrypted data
        """
        try:
            # Try to use cryptography if available
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                
                # Extract the IV (first 12 bytes)
                iv = encrypted_data[:12]
                ciphertext = encrypted_data[12:]
                
                # Decrypt the data
                aesgcm = AESGCM(key)
                return aesgcm.decrypt(iv, ciphertext, None)
            except ImportError:
                # Fallback decryption
                pass
            
            # Fallback to a simple XOR decryption for demo purposes
            # DO NOT USE THIS IN PRODUCTION
            import random
            random.seed(int.from_bytes(key, byteorder='big'))
            keystream = bytes([random.randint(0, 255) for _ in range(len(encrypted_data))])
            return bytes([a ^ b for a, b in zip(encrypted_data, keystream)])
            
        except Exception as e:
            raise ValueError(f"Decryption error: {e}")
        
    def _get_current_astronomical_coordinates(self) -> List[List[float]]:
        """
        Get current astronomical coordinates for use with the quantum ledger.
        
        Returns:
            List of spatial coordinates
        """
        if not self.observer or not ephem:
            # Fallback to random coordinates
            import random
            return [[random.uniform(-10, 10) for _ in range(3)] for _ in range(5)]
            
        # Use ephem to get current astronomical positions
        try:
            self.observer.date = ephem.now()
            
            # Get positions of planets
            planets = [ephem.Mercury(), ephem.Venus(), ephem.Mars(), 
                      ephem.Jupiter(), ephem.Saturn()]
                      
            for planet in planets:
                planet.compute(self.observer)
            
            # Create coordinates from the planets' positions
            coordinates = []
            for planet in planets:
                coordinates.append([
                    float(planet.ra), 
                    float(planet.dec), 
                    float(planet.earth_distance)
                ])
                
            return coordinates
            
        except Exception:
            # Fallback to random coordinates
            import random
            return [[random.uniform(-10, 10) for _ in range(3)] for _ in range(5)]
        
    def export_state(self) -> Dict[str, Any]:
        """
        Export the current state of the system.
        
        Returns:
            State dictionary
        """
        return {
            "policies": {pid: policy.to_dict() for pid, policy in self.policies.items()},
            "resources": {rid: resource.to_dict() for rid, resource in self.resources.items()},
            "access_logs": [log.to_dict() for log in self.access_logs],
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "policy_count": len(self.policies),
                "resource_count": len(self.resources),
                "log_count": len(self.access_logs)
            }
        }
        
    def import_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a state.
        
        Args:
            state: State dictionary from export_state
            
        Returns:
            Import result
        """
        try:
            # Import policies
            self.policies = {}
            for pid, policy_data in state.get("policies", {}).items():
                self.policies[pid] = TemporalAccessPolicy.from_dict(policy_data)
                
            # Import resources
            self.resources = {}
            for rid, resource_data in state.get("resources", {}).items():
                self.resources[rid] = EncryptedResource.from_dict(resource_data)
                
            # Import access logs
            self.access_logs = []
            for log_data in state.get("access_logs", []):
                self.access_logs.append(AccessLogEntry.from_dict(log_data))
                
            return {
                "success": True,
                "policy_count": len(self.policies),
                "resource_count": len(self.resources),
                "log_count": len(self.access_logs)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        
    def get_context_from_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a context dictionary from a request.
        This is a utility method to help build a context for access_resource.
        
        Args:
            request: Request data
            
        Returns:
            Context dictionary
        """
        context = {}
        
        # Add current time
        context["current_time"] = datetime.now()
        
        # Add identity information if available
        if "identity" in request:
            context["identity"] = request["identity"]
            
        # Add location information if available
        if "location" in request:
            context["current_location"] = request["location"]
            
        # Add document hash if available
        if "document_hash" in request:
            context["document_hash"] = request["document_hash"]
            
        # Add custom context variables
        if "context_variables" in request:
            context.update(request["context_variables"])
            
        return context
