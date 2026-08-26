import unittest
from datetime import datetime, timedelta
from src.revenue.quantum_ledger.quantum_ledger_audit import QuantumLedgerAudit

class TestQuantumLedgerAudit(unittest.TestCase):

    def setUp(self):
        """Set up a mock ledger with audit trail capabilities."""
        class MockLedger(QuantumLedgerAudit):
            def _get_current_spatial_coordinates(self):
                return [[0.0, 0.0, 0.0]]

            def _get_celestial_state_digest(self):
                return "mock_digest"

        self.ledger = MockLedger()
        self.ledger.initialize_audit_trail()

    def test_add_audit_event(self):
        """Test adding an audit event."""
        event = self.ledger._add_audit_event("test_event", {"key": "value"})
        self.assertEqual(event["event_type"], "test_event")
        self.assertIn("event_hash", event)
        self.assertEqual(len(self.ledger.audit_trail), 2)  # Includes ledger_created event

    def test_get_audit_trail(self):
        """Test retrieving filtered audit trail events."""
        self.ledger._add_audit_event("event_1", {"key": "value1"})
        self.ledger._add_audit_event("event_2", {"key": "value2"})

        start_time = (datetime.now() - timedelta(days=1)).isoformat()
        end_time = datetime.now().isoformat()

        filtered_events = self.ledger.get_audit_trail(start_time=start_time, end_time=end_time, event_types=["event_1"])
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["event_type"], "event_1")

    def test_audit_ledger(self):
        """Test performing a comprehensive audit of the ledger."""
        audit_results = self.ledger.audit_ledger()
        self.assertIn("integrity_validation", audit_results)
        self.assertIn("audit_trail_analysis", audit_results)
        self.assertTrue(audit_results["audit_summary"]["ledger_valid"])
    
    def setUp(self):
        """Set up a mock ledger with audit trail capabilities."""
    class MockLedger(QuantumLedgerAudit):
        def _get_current_spatial_coordinates(self):
            return [[0.0, 0.0, 0.0]]

        def _get_celestial_state_digest(self):
            return "mock_digest"

    self.ledger = MockLedger()
    self.ledger.initialize_audit_trail()

    # Add a record to the ledger with entanglement attributes
    self.ledger.records["record_1"] = QuantumEntangledRecord(
        document_hash="mock_document_hash",
        spatial_signature="mock_spatial_signature",
        timestamp="2023-01-01T00:00:00",
        entanglement_level=3,
        metadata={}
    )

    # Add blockchain metadata to the record
    self.ledger.records["record_1"].metadata["blockchain"] = {
        "transaction_id": "mock_transaction_id"
    }