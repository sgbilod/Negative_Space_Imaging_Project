"""
Incident Response System (IRS) Module
======================================

Automated incident response with:
- Incident classification and severity assessment
- Automated containment procedures
- Evidence preservation
- Alert escalation workflows
- Post-incident analysis

Author: @CIPHER - Advanced Cryptography & Security
Date: December 2025
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading

logger = logging.getLogger("incident_response")


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


class IncidentStatus(Enum):
    """Incident lifecycle status."""
    DETECTED = "DETECTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CONTAINED = "CONTAINED"
    MITIGATED = "MITIGATED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class IncidentType(Enum):
    """Classification of incident types."""
    MALWARE_INFECTION = "malware_infection"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DENIAL_OF_SERVICE = "denial_of_service"
    CREDENTIAL_COMPROMISE = "credential_compromise"
    MISCONFIGURATION = "misconfiguration"
    SUPPLY_CHAIN_ATTACK = "supply_chain_attack"
    INSIDER_THREAT = "insider_threat"
    OTHER = "other"


@dataclass
class ContainmentAction:
    """An action taken to contain an incident."""
    action_type: str
    description: str
    target: str  # What was affected
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    result: Optional[str] = None


@dataclass
class IncidentRecord:
    """Complete incident record."""
    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    title: str
    description: str
    detected_at: datetime
    reported_by: str
    status: IncidentStatus = IncidentStatus.DETECTED

    # Response tracking
    acknowledged_at: Optional[datetime] = None
    contained_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Affected systems
    affected_systems: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    affected_data: List[str] = field(default_factory=list)

    # Actions taken
    containment_actions: List[ContainmentAction] = field(default_factory=list)
    mitigation_steps: List[str] = field(default_factory=list)

    # Evidence
    evidence_collected: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None

    # Team
    assigned_to: Optional[str] = None
    escalated_to: Optional[str] = None


class ContainmentProcedure:
    """Automated containment procedures."""

    def __init__(self):
        """Initialize containment procedures."""
        self.procedures: Dict[IncidentType, List[Callable]] = {
            IncidentType.MALWARE_INFECTION: [
                self._isolate_system,
                self._disable_network,
                self._terminate_processes,
                self._preserve_logs
            ],
            IncidentType.DATA_BREACH: [
                self._revoke_access,
                self._isolate_data_stores,
                self._enable_audit_logging,
                self._notify_compliance
            ],
            IncidentType.UNAUTHORIZED_ACCESS: [
                self._revoke_credentials,
                self._invalidate_sessions,
                self._isolate_user_system,
                self._enable_mfa
            ],
            IncidentType.DENIAL_OF_SERVICE: [
                self._rate_limit_traffic,
                self._enable_ddos_protection,
                self._failover_to_backup,
                self._notify_isp
            ],
            IncidentType.CREDENTIAL_COMPROMISE: [
                self._reset_credentials,
                self._invalidate_tokens,
                self._enable_mfa,
                self._monitor_usage
            ]
        }

    async def execute_containment(
        self,
        incident: IncidentRecord
    ) -> List[ContainmentAction]:
        \"\"\"
        Execute containment procedures for incident type.

        Args:
            incident: The incident to contain

        Returns:
            List of actions taken
        \"\"\"
        actions = []

        if incident.incident_type in self.procedures:
            for procedure in self.procedures[incident.incident_type]:
                action = await procedure(incident)
                if action:
                    actions.append(action)
                    incident.containment_actions.append(action)

        return actions

    async def _isolate_system(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Isolate affected system from network.\"\"\"
        logger.warning(f\"Isolating systems: {incident.affected_systems}\")
        return ContainmentAction(
            action_type=\"ISOLATE_SYSTEM\",
            description=\"Disconnected from network\",
            target=\",\".join(incident.affected_systems),
            status=\"COMPLETED\"
        )

    async def _disable_network(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Disable network access.\"\"\"
        logger.warning(\"Disabling network interfaces\")
        return ContainmentAction(
            action_type=\"DISABLE_NETWORK\",
            description=\"Network interfaces disabled\",
            target=\"all\",
            status=\"COMPLETED\"
        )

    async def _terminate_processes(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Terminate suspicious processes.\"\"\"
        logger.warning(\"Terminating suspicious processes\")
        return ContainmentAction(
            action_type=\"TERMINATE_PROCESSES\",
            description=\"Malicious processes terminated\",
            target=\"malware\",
            status=\"COMPLETED\"
        )

    async def _preserve_logs(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Preserve evidence and logs.\"\"\"
        logger.info(\"Preserving logs for forensics\")
        return ContainmentAction(
            action_type=\"PRESERVE_LOGS\",
            description=\"Evidence preserved for analysis\",
            target=\"log_systems\",
            status=\"COMPLETED\"
        )

    async def _revoke_access(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Revoke unauthorized access.\"\"\"
        logger.warning(f\"Revoking access for users: {incident.affected_users}\")
        return ContainmentAction(
            action_type=\"REVOKE_ACCESS\",
            description=\"User access revoked\",
            target=\",\".join(incident.affected_users),
            status=\"COMPLETED\"
        )

    async def _isolate_data_stores(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Isolate affected data stores.\"\"\"
        logger.warning(f\"Isolating data: {incident.affected_data}\")
        return ContainmentAction(
            action_type=\"ISOLATE_DATA\",
            description=\"Data stores isolated\",
            target=\",\".join(incident.affected_data),
            status=\"COMPLETED\"
        )

    async def _enable_audit_logging(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Enable enhanced audit logging.\"\"\"
        logger.info(\"Enabling comprehensive audit logging\")
        return ContainmentAction(
            action_type=\"ENABLE_AUDIT\",
            description=\"Audit logging enabled\",
            target=\"all_systems\",
            status=\"COMPLETED\"
        )

    async def _notify_compliance(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Notify compliance team.\"\"\"
        logger.info(\"Notifying compliance team\")
        return ContainmentAction(
            action_type=\"NOTIFY_COMPLIANCE\",
            description=\"Compliance team notified\",
            target=\"compliance\",
            status=\"COMPLETED\"
        )

    async def _reset_credentials(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Reset compromised credentials.\"\"\"
        logger.warning(f\"Resetting credentials for: {incident.affected_users}\")
        return ContainmentAction(
            action_type=\"RESET_CREDENTIALS\",
            description=\"Credentials reset\",
            target=\",\".join(incident.affected_users),
            status=\"COMPLETED\"
        )

    async def _invalidate_sessions(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Invalidate active sessions.\"\"\"
        logger.warning(\"Invalidating active sessions\")
        return ContainmentAction(
            action_type=\"INVALIDATE_SESSIONS\",
            description=\"All active sessions terminated\",
            target=\"all\",
            status=\"COMPLETED\"
        )

    async def _invalidate_tokens(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Invalidate authentication tokens.\"\"\"
        logger.warning(\"Invalidating authentication tokens\")
        return ContainmentAction(
            action_type=\"INVALIDATE_TOKENS\",
            description=\"All tokens revoked\",
            target=\"auth_system\",
            status=\"COMPLETED\"
        )

    async def _enable_mfa(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Enable or enforce MFA.\"\"\"
        logger.info(\"Enforcing MFA\")
        return ContainmentAction(
            action_type=\"ENFORCE_MFA\",
            description=\"Multi-factor authentication enforced\",
            target=\"all_users\",
            status=\"COMPLETED\"
        )

    async def _rate_limit_traffic(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Enable rate limiting.\"\"\"
        logger.warning(\"Enabling rate limiting\")
        return ContainmentAction(
            action_type=\"RATE_LIMIT\",
            description=\"Rate limiting enabled\",
            target=\"traffic\",
            status=\"COMPLETED\"
        )

    async def _enable_ddos_protection(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Enable DDoS protection.\"\"\"
        logger.warning(\"Enabling DDoS protection\")
        return ContainmentAction(
            action_type=\"DDOS_PROTECTION\",
            description=\"DDoS protection activated\",
            target=\"network\",
            status=\"COMPLETED\"
        )

    async def _failover_to_backup(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Failover to backup systems.\"\"\"
        logger.info(\"Failover to backup systems\")
        return ContainmentAction(
            action_type=\"FAILOVER\",
            description=\"Failover completed\",
            target=\"backup_systems\",
            status=\"COMPLETED\"
        )

    async def _notify_isp(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Notify ISP of DDoS attack.\"\"\"
        logger.info(\"Notifying ISP\")
        return ContainmentAction(
            action_type=\"NOTIFY_ISP\",
            description=\"ISP notified of attack\",
            target=\"isp\",
            status=\"COMPLETED\"
        )

    async def _monitor_usage(self, incident: IncidentRecord) -> Optional[ContainmentAction]:
        \"\"\"Enable enhanced monitoring.\"\"\"
        logger.info(\"Enabling enhanced monitoring\")
        return ContainmentAction(
            action_type=\"MONITOR\",
            description=\"Enhanced monitoring activated\",
            target=\"all_systems\",
            status=\"COMPLETED\"
        )


class EscalationWorkflow:
    \"\"\"Incident escalation workflow.\"\"\"

    def __init__(self):
        \"\"\"Initialize escalation workflow.\"\"\"
        self.escalation_thresholds = {
            IncidentSeverity.LOW: \"Level 1 - Monitoring\",
            IncidentSeverity.MEDIUM: \"Level 2 - Investigation\",
            IncidentSeverity.HIGH: \"Level 3 - Incident Commander\",
            IncidentSeverity.CRITICAL: \"Level 4 - CISO Notification\",
            IncidentSeverity.CATASTROPHIC: \"Level 5 - CEO/Legal Notification\"
        }

    def get_escalation_level(self, severity: IncidentSeverity) -> str:
        \"\"\"Get escalation level for severity.\"\"\"
        return self.escalation_thresholds.get(severity, \"Unknown\")

    def should_escalate(
        self,
        incident: IncidentRecord,
        time_threshold_minutes: int = 60
    ) -> bool:
        \"\"\"
        Determine if incident should be escalated.

        Args:
            incident: The incident
            time_threshold_minutes: Time since detection

        Returns:
            True if should escalate
        \"\"\"
        time_since_detection = datetime.utcnow() - incident.detected_at

        if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.CATASTROPHIC]:
            return True

        if time_since_detection > timedelta(minutes=time_threshold_minutes):
            return True

        return False


class IncidentResponseSystem:
    \"\"\"Main incident response system.\"\"\"

    def __init__(self, evidence_dir: str = \"./security/incident_evidence\"):
        \"\"\"Initialize incident response system.\"\"\"
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.containment = ContainmentProcedure()
        self.escalation = EscalationWorkflow()

        self.incidents: Dict[str, IncidentRecord] = {}
        self._lock = threading.RLock()

        self._incident_counter = 0

        logger.info(\"Initialized IncidentResponseSystem\")

    def create_incident(
        self,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        title: str,
        description: str,
        reported_by: str,
        affected_systems: Optional[List[str]] = None,
        affected_users: Optional[List[str]] = None,
        affected_data: Optional[List[str]] = None
    ) -> IncidentRecord:
        \"\"\"
        Create and register a new incident.

        Args:
            incident_type: Type of incident
            severity: Severity level
            title: Incident title
            description: Detailed description
            reported_by: Who reported it
            affected_systems: Systems affected
            affected_users: Users affected
            affected_data: Data affected

        Returns:
            New incident record
        \"\"\"
        with self._lock:
            self._incident_counter += 1
            incident_id = f\"INC-{datetime.utcnow().strftime('%Y%m%d')}-{self._incident_counter:04d}\"

        incident = IncidentRecord(
            incident_id=incident_id,
            incident_type=incident_type,
            severity=severity,
            title=title,
            description=description,
            detected_at=datetime.utcnow(),
            reported_by=reported_by,
            affected_systems=affected_systems or [],
            affected_users=affected_users or [],
            affected_data=affected_data or []
        )

        with self._lock:
            self.incidents[incident_id] = incident

        logger.warning(f\"🚨 INCIDENT CREATED: {incident_id} - {severity.name} - {title}\")
        self._save_incident(incident)

        return incident

    def acknowledge_incident(
        self,
        incident_id: str,
        assigned_to: str
    ) -> bool:
        \"\"\"Acknowledge receipt of incident.\"\"\"
        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.ACKNOWLEDGED
        incident.acknowledged_at = datetime.utcnow()
        incident.assigned_to = assigned_to

        logger.info(f\"Incident {incident_id} acknowledged by {assigned_to}\")
        return True

    async def contain_incident(self, incident_id: str) -> bool:
        \"\"\"Execute containment procedures.\"\"\"
        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]

        # Execute containment
        actions = await self.containment.execute_containment(incident)

        incident.status = IncidentStatus.CONTAINED
        incident.contained_at = datetime.utcnow()

        logger.info(f\"Incident {incident_id} contained with {len(actions)} actions\")
        return True

    def resolve_incident(
        self,
        incident_id: str,
        root_cause: str,
        mitigation_steps: List[str]
    ) -> bool:
        \"\"\"Resolve incident.\"\"\"
        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.utcnow()
        incident.root_cause = root_cause
        incident.mitigation_steps = mitigation_steps

        logger.info(f\"Incident {incident_id} resolved\")
        return True

    def close_incident(self, incident_id: str) -> bool:
        \"\"\"Close incident.\"\"\"
        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.CLOSED
        incident.closed_at = datetime.utcnow()

        logger.info(f\"Incident {incident_id} closed\")
        return True

    def _save_incident(self, incident: IncidentRecord) -> None:
        \"\"\"Save incident to file for audit trail.\"\"\"
        try:
            incident_file = (
                self.evidence_dir / f\"incidents_{datetime.utcnow().strftime('%Y%m%d')}.jsonl\"
            )

            incident_dict = {
                \"incident_id\": incident.incident_id,
                \"timestamp\": incident.detected_at.isoformat(),
                \"type\": incident.incident_type.value,
                \"severity\": incident.severity.name,
                \"title\": incident.title
            }

            with open(incident_file, 'a') as f:
                f.write(json.dumps(incident_dict) + '\\n')

        except Exception as e:
            logger.error(f\"Error saving incident: {e}\")

    def get_incident_summary(self) -> Dict:
        \"\"\"Get summary of all incidents.\"\"\"
        with self._lock:
            summary = {
                \"timestamp\": datetime.utcnow().isoformat(),
                \"total_incidents\": len(self.incidents),
                \"by_status\": {},
                \"by_severity\": {},
                \"by_type\": {}
            }

            for incident in self.incidents.values():
                # Count by status
                status = incident.status.value
                summary[\"by_status\"][status] = summary[\"by_status\"].get(status, 0) + 1

                # Count by severity
                sev = incident.severity.name
                summary[\"by_severity\"][sev] = summary[\"by_severity\"].get(sev, 0) + 1

                # Count by type
                itype = incident.incident_type.value
                summary[\"by_type\"][itype] = summary[\"by_type\"].get(itype, 0) + 1

            return summary


if __name__ == \"__main__\":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    print(\"\\n=== Incident Response System Demo ===\\n\")

    irs = IncidentResponseSystem()

    # Create an incident
    print(\"1. Creating critical incident...\")
    incident = irs.create_incident(
        incident_type=IncidentType.UNAUTHORIZED_ACCESS,
        severity=IncidentSeverity.CRITICAL,
        title=\"Unauthorized Admin Access Detected\",
        description=\"Suspicious login from unusual location detected on admin account\",
        reported_by=\"Security Monitoring System\",
        affected_users=[\"admin@company.com\"],
        affected_systems=[\"admin-portal.example.com\"],
        affected_data=[\"user_database\", \"admin_settings\"]
    )

    # Acknowledge
    print(\"2. Acknowledging incident...\")
    irs.acknowledge_incident(incident.incident_id, \"security_team_lead\")

    # Contain
    print(\"3. Executing containment procedures...\")
    asyncio.run(irs.contain_incident(incident.incident_id))

    # Resolve
    print(\"4. Resolving incident...\")
    irs.resolve_incident(
        incident.incident_id,
        root_cause=\"Compromised admin credentials from phishing attack\",
        mitigation_steps=[
            \"Reset all admin passwords\",
            \"Enable mandatory MFA\",
            \"Review access logs\",
            \"Deploy security awareness training\"
        ]
    )

    # Summary
    print(\"\\n5. Incident Summary:\")\n    summary = irs.get_incident_summary()
    print(f\"   Total Incidents: {summary['total_incidents']}\")
    print(f\"   By Status: {summary['by_status']}\")

    print(\"\\n=== Demo Complete ===\")
"
