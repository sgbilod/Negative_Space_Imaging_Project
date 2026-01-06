"""
Compliance Reporter Module
============================

Generates compliance reports for:
- SOC 2 Type II (Trust Service Criteria)
- HIPAA (Health Insurance Portability and Accountability Act)
- PCI-DSS (Payment Card Industry Data Security Standard)
- GDPR (General Data Protection Regulation)

Author: @CIPHER - Advanced Cryptography & Security
Date: December 2025
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger("compliance_reporter")


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2_TYPE_II = "SOC 2 Type II"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"
    GDPR = "GDPR"


class ComplianceStatus(Enum):
    """Compliance status."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ComplianceControl:
    """Individual compliance control."""
    control_id: str
    description: str
    framework: ComplianceFramework
    status: ComplianceStatus
    evidence: str
    findings: List[str] = field(default_factory=list)
    remediation: Optional[str] = None
    assessment_date: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceSection:
    """Section of a compliance report."""
    title: str
    controls: List[ComplianceControl] = field(default_factory=list)
    summary: str = ""

    def calculate_compliance_percentage(self) -> float:
        """Calculate compliance percentage."""
        if not self.controls:
            return 0.0

        compliant_count = sum(
            1 for c in self.controls
            if c.status == ComplianceStatus.COMPLIANT
        )
        return (compliant_count / len(self.controls)) * 100


class SOC2Reporter:
    """Generates SOC 2 Type II reports."""

    TRUST_SERVICE_CRITERIA = {
        "Security": [
            ("CC1", "Governance", "Organization obtains or generates information"),
            ("CC2", "Objectives", "Board demonstrates independence"),
            ("CC3", "Responsibilities", "Entity obtains information for decisions"),
            ("CC4", "Competence", "Entity obtains information from sources"),
            ("CC5", "Responsibilities", "Entity holds individuals accountable"),
            ("CC6", "Fulfillment", "Entity specifies objectives clearly"),
            ("CC7", "Responsibility Assignment", "Responsibility clearly documented"),
            ("CC8", "Authority", "Competence levels documented"),
            ("CC9", "Accountability", "Accountability measures in place"),
        ],
        "Availability": [
            ("A1", "Availability Objectives", "System objectives established"),
            ("A2", "Capacity Planning", "Capacity monitoring performed"),
        ],
        "Processing Integrity": [
            ("PI1", "Objectives", "System processing objectives established"),
            ("PI2", "Completeness", "Complete and accurate processing"),
        ],
        "Confidentiality": [
            ("C1", "Objectives", "Confidentiality objectives established"),
            ("C2", "Processes", "Access policies restricted"),
        ],
        "Privacy": [
            ("P1", "Management", "Privacy policies established"),
            ("P2", "Notice", "Entity provides privacy notice"),
        ]
    }

    def __init__(self):
        """Initialize SOC 2 reporter."""
        self.sections: Dict[str, ComplianceSection] = {}
        self._initialize_sections()

    def _initialize_sections(self) -> None:
        """Initialize trust service criteria sections."""
        for category, criteria in self.TRUST_SERVICE_CRITERIA.items():
            controls = []
            for control_id, title, description in criteria:
                control = ComplianceControl(
                    control_id=control_id,
                    description=f"{title}: {description}",
                    framework=ComplianceFramework.SOC2_TYPE_II,
                    status=ComplianceStatus.UNKNOWN,
                    evidence=""
                )
                controls.append(control)

            self.sections[category] = ComplianceSection(title=category, controls=controls)

    def update_control(
        self,
        category: str,
        control_id: str,
        status: ComplianceStatus,
        evidence: str,
        findings: Optional[List[str]] = None
    ) -> bool:
        """
        Update control status.

        Args:
            category: Section category
            control_id: Control identifier
            status: Compliance status
            evidence: Evidence supporting status
            findings: Any findings

        Returns:
            True if successful
        """
        if category not in self.sections:
            return False

        for control in self.sections[category].controls:
            if control.control_id == control_id:
                control.status = status
                control.evidence = evidence
                control.findings = findings or []
                control.assessment_date = datetime.utcnow()
                return True

        return False

    def generate_report(self) -> Dict:
        """Generate SOC 2 report."""
        report = {
            "framework": "SOC 2 Type II",
            "report_date": datetime.utcnow().isoformat(),
            "sections": {}
        }

        total_controls = 0
        compliant_controls = 0

        for section_name, section in self.sections.items():
            section_data = {
                "title": section.title,
                "compliance_percentage": section.calculate_compliance_percentage(),
                "controls": []
            }

            for control in section.controls:
                total_controls += 1
                if control.status == ComplianceStatus.COMPLIANT:
                    compliant_controls += 1

                section_data["controls"].append({
                    "id": control.control_id,
                    "description": control.description,
                    "status": control.status.value,
                    "evidence": control.evidence,
                    "findings": control.findings
                })

            report["sections"][section_name] = section_data

        report["overall_compliance"] = (compliant_controls / total_controls * 100) if total_controls > 0 else 0
        return report


class HIPAAReporter:
    """Generates HIPAA compliance reports."""

    HIPAA_REQUIREMENTS = {
        "Administrative": [
            ("164.100", "Security Management Process", "Develop and implement security policies"),
            ("164.104", "Assigned Security Responsibility", "Designate security official"),
            ("164.106", "Workforce Security", "Implement user access controls"),
            ("164.308", "Administrative Safeguards", "Protect PHI with administrative controls"),
        ],
        "Physical": [
            ("164.310", "Physical Safeguards", "Control physical access to facilities"),
            ("164.312", "Facility Access Controls", "Implement facility access controls"),
        ],
        "Technical": [
            ("164.312", "Technical Safeguards", "Implement encryption and decryption"),
            ("164.314", "Implementation Specifications", "Audit controls and logging"),
        ],
        "Organizational": [
            ("164.400", "Notification Rule", "Notify individuals of breaches"),
            ("164.402", "Notification to Individuals", "Notify without unreasonable delay"),
        ]
    }

    def __init__(self):
        \"\"\"Initialize HIPAA reporter.\"\"\"
        self.sections: Dict[str, ComplianceSection] = {}
        self._initialize_sections()

    def _initialize_sections(self) -> None:
        \"\"\"Initialize HIPAA requirement sections.\"\"\"
        for category, requirements in self.HIPAA_REQUIREMENTS.items():
            controls = []
            for req_id, title, description in requirements:
                control = ComplianceControl(
                    control_id=req_id,
                    description=f\"{title}: {description}\",
                    framework=ComplianceFramework.HIPAA,
                    status=ComplianceStatus.UNKNOWN,
                    evidence=\"\"
                )
                controls.append(control)

            self.sections[category] = ComplianceSection(title=category, controls=controls)

    def update_requirement(
        self,
        category: str,
        req_id: str,
        status: ComplianceStatus,
        evidence: str
    ) -> bool:
        \"\"\"Update HIPAA requirement status.\"\"\"
        if category not in self.sections:
            return False

        for control in self.sections[category].controls:
            if control.control_id == req_id:
                control.status = status
                control.evidence = evidence
                control.assessment_date = datetime.utcnow()
                return True

        return False

    def generate_report(self) -> Dict:
        \"\"\"Generate HIPAA compliance report.\"\"\"
        report = {
            \"framework\": \"HIPAA\",
            \"report_date\": datetime.utcnow().isoformat(),
            \"sections\": {}
        }

        total_reqs = 0
        compliant_reqs = 0

        for section_name, section in self.sections.items():
            section_data = {
                \"title\": section.title,
                \"compliance_percentage\": section.calculate_compliance_percentage(),
                \"requirements\": []
            }

            for control in section.controls:
                total_reqs += 1
                if control.status == ComplianceStatus.COMPLIANT:
                    compliant_reqs += 1

                section_data[\"requirements\"].append({
                    \"id\": control.control_id,
                    \"description\": control.description,
                    \"status\": control.status.value,
                    \"evidence\": control.evidence
                })

            report[\"sections\"][section_name] = section_data

        report[\"overall_compliance\"] = (compliant_reqs / total_reqs * 100) if total_reqs > 0 else 0
        return report


class PCI_DSSReporter:
    \"\"\"Generates PCI-DSS compliance reports.\"\"\"

    PCI_DSS_REQUIREMENTS = {
        \"Network Security\": [
            (\"1\", \"Install and maintain firewall\", \"Firewall infrastructure in place\"),
            (\"2\", \"Default passwords changed\", \"All default passwords removed\"),
        ],
        \"Data Protection\": [
            (\"3\", \"Protect stored cardholder data\", \"Encryption for stored data\"),
            (\"4\", \"Protect data in transit\", \"Strong cryptography in use\"),
        ],
        \"Vulnerability Management\": [
            (\"5\", \"Protect against malware\", \"Antivirus software deployed\"),
            (\"6\", \"Maintain secure systems\", \"Security patches applied\"),
        ],
        \"Access Control\": [
            (\"7\", \"Restrict access to cardholder data\", \"Role-based access control\"),
            (\"8\", \"User identification and authentication\", \"Strong authentication implemented\"),
        ],
        \"Testing and Monitoring\": [
            (\"9\", \"Restrict physical access\", \"Physical security measures\"),
            (\"10\", \"Track and monitor access\", \"Comprehensive audit logging\"),
        ]
    }

    def __init__(self):
        \"\"\"Initialize PCI-DSS reporter.\"\"\"
        self.sections: Dict[str, ComplianceSection] = {}
        self._initialize_sections()

    def _initialize_sections(self) -> None:
        \"\"\"Initialize PCI-DSS requirement sections.\"\"\"
        for category, requirements in self.PCI_DSS_REQUIREMENTS.items():
            controls = []
            for req_num, title, description in requirements:
                control = ComplianceControl(
                    control_id=f\"REQ{req_num}\",
                    description=f\"{title}: {description}\",
                    framework=ComplianceFramework.PCI_DSS,
                    status=ComplianceStatus.UNKNOWN,
                    evidence=\"\"
                )
                controls.append(control)

            self.sections[category] = ComplianceSection(title=category, controls=controls)

    def generate_report(self) -> Dict:
        \"\"\"Generate PCI-DSS compliance report.\"\"\"
        report = {
            \"framework\": \"PCI-DSS 3.2.1\",
            \"report_date\": datetime.utcnow().isoformat(),
            \"sections\": {}
        }

        total_reqs = 0
        compliant_reqs = 0

        for section_name, section in self.sections.items():
            section_data = {
                \"title\": section.title,
                \"compliance_percentage\": section.calculate_compliance_percentage(),
                \"requirements\": []
            }

            for control in section.controls:
                total_reqs += 1
                if control.status == ComplianceStatus.COMPLIANT:
                    compliant_reqs += 1

                section_data[\"requirements\"].append({
                    \"id\": control.control_id,
                    \"description\": control.description,
                    \"status\": control.status.value
                })

            report[\"sections\"][section_name] = section_data

        report[\"overall_compliance\"] = (compliant_reqs / total_reqs * 100) if total_reqs > 0 else 0
        return report


class ComplianceReporter:
    \"\"\"Master compliance reporting system.\"\"\"

    def __init__(self, reports_dir: str = \"./security/compliance_reports\"):
        \"\"\"Initialize compliance reporter.\"\"\"
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.soc2_reporter = SOC2Reporter()
        self.hipaa_reporter = HIPAAReporter()
        self.pci_dss_reporter = PCI_DSSReporter()

        logger.info(\"Initialized ComplianceReporter\")

    def generate_all_reports(self) -> Dict:
        \"\"\"Generate all compliance reports.\"\"\"
        reports = {
            \"timestamp\": datetime.utcnow().isoformat(),
            \"soc2\": self.soc2_reporter.generate_report(),
            \"hipaa\": self.hipaa_reporter.generate_report(),
            \"pci_dss\": self.pci_dss_reporter.generate_report()
        }

        # Save to file
        report_file = self.reports_dir / f\"compliance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json\"
        with open(report_file, 'w') as f:
            json.dump(reports, f, indent=2, default=str)

        logger.info(f\"Compliance report saved to {report_file}\")
        return reports

    def get_executive_summary(self) -> Dict:
        \"\"\"Get executive summary of compliance status.\"\"\"
        return {
            \"timestamp\": datetime.utcnow().isoformat(),
            \"frameworks\": [
                {
                    \"name\": \"SOC 2 Type II\",
                    \"compliance\": self.soc2_reporter.generate_report()[\"overall_compliance\"]
                },
                {
                    \"name\": \"HIPAA\",
                    \"compliance\": self.hipaa_reporter.generate_report()[\"overall_compliance\"]
                },
                {
                    \"name\": \"PCI-DSS\",
                    \"compliance\": self.pci_dss_reporter.generate_report()[\"overall_compliance\"]
                }
            ]
        }


if __name__ == \"__main__\":
    logging.basicConfig(level=logging.INFO)

    print(\"\\n=== Compliance Reporter Demo ===\\n\")

    reporter = ComplianceReporter()

    # Update some controls as example
    reporter.soc2_reporter.update_control(
        \"Security\",
        \"CC1\",
        ComplianceStatus.COMPLIANT,
        \"Governance policy implemented and documented\"
    )

    reporter.hipaa_reporter.update_requirement(
        \"Technical\",
        \"164.312\",
        ComplianceStatus.COMPLIANT,
        \"AES-256 encryption implemented for PHI\"
    )

    reporter.pci_dss_reporter.update_requirement(
        \"Data Protection\",
        \"4\",
        ComplianceStatus.COMPLIANT,
        \"TLS 1.3 enforced for all data in transit\"
    )

    # Generate summary
    print(\"\\nCompliance Executive Summary:\")
    summary = reporter.get_executive_summary()
    for framework in summary[\"frameworks\"]:
        print(f\"  {framework['name']}: {framework['compliance']:.1f}% compliant\")

    print(\"\\n=== Demo Complete ===\")
"
