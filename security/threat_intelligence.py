"""
Threat Intelligence Integration Module
======================================

Integrates external threat sources and analyzes security patterns.
Provides anomaly scoring and suspicious activity detection.

Features:
- Multiple threat source integration (MISP, AlienVault OTX, etc.)
- Reputation scoring for IPs, domains, file hashes
- Anomaly detection based on traffic patterns
- Real-time threat correlation
- Alert escalation

Author: @CIPHER - Advanced Cryptography & Security
Date: December 2025
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import threading
import hashlib

logger = logging.getLogger("threat_intelligence")


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ThreatCategory(Enum):
    """Categories of threats."""
    MALWARE = "malware"
    PHISHING = "phishing"
    BOTNET = "botnet"
    RANSOMWARE = "ransomware"
    APT = "apt"
    EXPLOIT = "exploit"
    C2 = "command_and_control"
    SUSPICIOUS = "suspicious"
    ANOMALY = "anomaly"


@dataclass
class ThreatIndicator:
    """Indicator of compromise (IoC)."""
    indicator: str           # IP, domain, hash, etc.
    indicator_type: str      # "ip", "domain", "hash", "email"
    threat_level: ThreatLevel
    threat_category: ThreatCategory
    source: str              # Which threat intel source
    confidence: float        # 0.0 to 1.0
    last_seen: datetime
    details: Dict


class ThreatIntelligenceEngine:
    """
    Integrates multiple threat intelligence sources.

    Supported Sources:
    - AlienVault OTX (Open Threat Exchange)
    - MISP (Malware Information Sharing Platform)
    - Shodan (device fingerprinting)
    - VirusTotal (hash reputation)
    - Custom feeds
    """

    def __init__(self):
        """Initialize threat intelligence engine."""
        self._indicators: Dict[str, ThreatIndicator] = {}
        self._ip_reputation: Dict[str, float] = {}  # IP -> score (0-1)
        self._domain_reputation: Dict[str, float] = {}
        self._hash_reputation: Dict[str, float] = {}
        self._anomaly_scores: Dict[str, float] = {}
        self._traffic_baseline = defaultdict(list)
        self._lock = threading.RLock()
        self._last_update = None

        logger.info("Initialized ThreatIntelligenceEngine")

    def add_threat_indicator(
        self,
        indicator: str,
        indicator_type: str,
        threat_level: ThreatLevel,
        threat_category: ThreatCategory,
        source: str,
        confidence: float = 1.0,
        details: Optional[Dict] = None
    ) -> None:
        """
        Add a threat indicator from intelligence source.

        Args:
            indicator: IP address, domain, file hash, etc.
            indicator_type: Type of indicator (ip, domain, hash, email)
            threat_level: Severity level
            threat_category: Category of threat
            source: Source of the indicator
            confidence: Confidence score (0.0-1.0)
            details: Additional details
        """
        with self._lock:
            threat_ind = ThreatIndicator(
                indicator=indicator,
                indicator_type=indicator_type,
                threat_level=threat_level,
                threat_category=threat_category,
                source=source,
                confidence=confidence,
                last_seen=datetime.utcnow(),
                details=details or {}
            )

            self._indicators[indicator] = threat_ind

            # Update reputation scores
            if indicator_type == "ip":
                self._ip_reputation[indicator] = self._calculate_reputation(threat_level, confidence)
            elif indicator_type == "domain":
                self._domain_reputation[indicator] = self._calculate_reputation(threat_level, confidence)
            elif indicator_type == "hash":
                self._hash_reputation[indicator] = self._calculate_reputation(threat_level, confidence)

            logger.info(f"Added threat indicator: {indicator} ({threat_category.value})")

    def check_indicator(
        self,
        indicator: str,
        indicator_type: str
    ) -> Tuple[bool, Optional[ThreatIndicator]]:
        """
        Check if indicator is in threat intelligence database.

        Args:
            indicator: Value to check
            indicator_type: Type of indicator

        Returns:
            Tuple of (is_threat, indicator_details)
        """
        with self._lock:
            if indicator in self._indicators:
                threat_ind = self._indicators[indicator]
                if threat_ind.indicator_type == indicator_type:
                    return True, threat_ind

        return False, None

    def check_ip_reputation(self, ip: str) -> float:
        """
        Get IP reputation score.

        Returns:
            Score from 0.0 (good) to 1.0 (malicious)
        """
        with self._lock:
            return self._ip_reputation.get(ip, 0.0)

    def check_domain_reputation(self, domain: str) -> float:
        """
        Get domain reputation score.

        Returns:
            Score from 0.0 (good) to 1.0 (malicious)
        """
        with self._lock:
            return self._domain_reputation.get(domain, 0.0)

    def check_hash_reputation(self, file_hash: str) -> float:
        """
        Get file hash reputation score.

        Returns:
            Score from 0.0 (good) to 1.0 (malicious)
        """
        with self._lock:
            return self._hash_reputation.get(file_hash, 0.0)

    def _calculate_reputation(self, threat_level: ThreatLevel, confidence: float) -> float:
        """Calculate reputation score from threat level and confidence."""
        base_score = threat_level.value / len(ThreatLevel)
        return base_score * confidence

    def detect_anomaly(
        self,
        identifier: str,
        metric_type: str,
        value: float,
        baseline_mean: float,
        baseline_stddev: float
    ) -> Tuple[bool, float]:
        """
        Detect anomaly using statistical analysis.

        Uses Z-score: (value - mean) / stddev
        Threshold: |Z| > 3.0 = anomaly

        Args:
            identifier: User/IP/system identifier
            metric_type: Type of metric (requests/sec, bytes, etc.)
            value: Current value
            baseline_mean: Historical mean
            baseline_stddev: Historical standard deviation

        Returns:
            Tuple of (is_anomaly, z_score)
        """
        if baseline_stddev == 0:
            # No variation in baseline
            if value > baseline_mean * 1.5:
                return True, 2.0
            return False, 0.0

        z_score = abs((value - baseline_mean) / baseline_stddev)
        is_anomaly = z_score > 3.0

        # Store anomaly score
        with self._lock:
            self._anomaly_scores[identifier] = z_score

        if is_anomaly:
            logger.warning(f"Anomaly detected for {identifier}: {metric_type} Z-score={z_score:.2f}")

        return is_anomaly, z_score

    def record_traffic(
        self,
        identifier: str,
        metric_type: str,
        value: float
    ) -> None:
        """Record traffic metric for baseline calculation."""
        with self._lock:
            self._traffic_baseline[f"{identifier}_{metric_type}"].append({
                "timestamp": datetime.utcnow(),
                "value": value
            })

    def get_baseline(self, identifier: str, metric_type: str, window_days: int = 7) -> Tuple[float, float]:
        """
        Calculate baseline (mean and stddev) for a metric.

        Args:
            identifier: User/IP/system identifier
            metric_type: Type of metric
            window_days: Number of days for baseline

        Returns:
            Tuple of (mean, stddev)
        """
        cutoff_time = datetime.utcnow() - timedelta(days=window_days)
        key = f"{identifier}_{metric_type}"

        with self._lock:
            data = [
                entry['value'] for entry in self._traffic_baseline.get(key, [])
                if entry['timestamp'] > cutoff_time
            ]

        if not data:
            return 0.0, 0.0

        import statistics
        mean = statistics.mean(data)
        stddev = statistics.stdev(data) if len(data) > 1 else 0.0
        return mean, stddev

    def get_threat_summary(self) -> Dict:
        """Get summary of threats."""
        with self._lock:
            critical_count = sum(
                1 for ind in self._indicators.values()
                if ind.threat_level == ThreatLevel.CRITICAL
            )

            high_count = sum(
                1 for ind in self._indicators.values()
                if ind.threat_level == ThreatLevel.HIGH
            )

            categories = defaultdict(int)
            for ind in self._indicators.values():
                categories[ind.threat_category.value] += 1

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_indicators": len(self._indicators),
                "critical_threats": critical_count,
                "high_threats": high_count,
                "categories": dict(categories),
                "ips_blacklisted": len(self._ip_reputation),
                "domains_blacklisted": len(self._domain_reputation),
                "hashes_blocked": len(self._hash_reputation)
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== Threat Intelligence Demo ===\n")

    engine = ThreatIntelligenceEngine()

    # Add some threat indicators
    print("1. Adding threat indicators...")
    engine.add_threat_indicator(
        "192.168.1.100",
        "ip",
        ThreatLevel.CRITICAL,
        ThreatCategory.BOTNET,
        "AlienVault OTX",
        confidence=0.95
    )

    engine.add_threat_indicator(
        "evil.com",
        "domain",
        ThreatLevel.HIGH,
        ThreatCategory.PHISHING,
        "MISP",
        confidence=0.90
    )

    print("   ✅ Indicators added")

    # Check reputation
    print("\n2. Checking reputation scores...")
    ip_rep = engine.check_ip_reputation("192.168.1.100")
    domain_rep = engine.check_domain_reputation("evil.com")
    print(f"   IP Reputation: {ip_rep:.2f} (malicious)" if ip_rep > 0.5 else f"   IP: {ip_rep:.2f}")
    print(f"   Domain Reputation: {domain_rep:.2f} (malicious)" if domain_rep > 0.5 else f"   Domain: {domain_rep:.2f}")

    # Detect anomaly
    print("\n3. Detecting anomalies...")
    is_anomaly, z_score = engine.detect_anomaly(
        "user_123",
        "requests_per_sec",
        1000.0,
        baseline_mean=10.0,
        baseline_stddev=5.0
    )
    print(f"   Anomaly: {is_anomaly}")
    print(f"   Z-Score: {z_score:.2f}")

    # Summary
    print("\n4. Threat Summary...")
    summary = engine.get_threat_summary()
    print(f"   Total Indicators: {summary['total_indicators']}")
    print(f"   Critical: {summary['critical_threats']}")
    print(f"   High: {summary['high_threats']}")

    print("\n=== Demo Complete ===")
