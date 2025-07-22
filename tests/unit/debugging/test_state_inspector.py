"""
Tests for the state inspection framework.
"""

import json
import time
from datetime import datetime
from threading import Thread
from unittest.mock import MagicMock, Mock

import pytest

from src.mcp_server_git.debugging.state_inspector import (
    ComponentStateInspector,
    StateSnapshot,
)
from src.mcp_server_git.protocols.debugging_protocol import (
    ComponentState,
    DebuggableComponent,
    DebugInfo,
    ValidationResult,
)


class MockComponentState:
    """Mock implementation of ComponentState protocol."""

    def __init__(self, component_id: str, component_type: str, state_data: dict):
        self._component_id = component_id
        self._component_type = component_type
        self._state_data = state_data
        self._last_updated = datetime.now()

    @property
    def component_id(self) -> str:
        return self._component_id

    @property
    def component_type(self) -> str:
        return self._component_type

    @property
    def state_data(self) -> dict:
        return self._state_data

    @property
    def last_updated(self) -> datetime:
        return self._last_updated


class MockValidationResult:
    """Mock implementation of ValidationResult protocol."""

    def __init__(self, is_valid: bool = True, errors: list = None, warnings: list = None):
        self._is_valid = is_valid
        self._validation_errors = errors or []
        self._validation_warnings = warnings or []
        self._validation_timestamp = datetime.now()

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def validation_errors(self) -> list:
        return self._validation_errors

    @property
    def validation_warnings(self) -> list:
        return self._validation_warnings

    @property
    def validation_timestamp(self) -> datetime:
        return self._validation_timestamp


class MockDebugInfo:
    """Mock implementation of DebugInfo protocol."""

    def __init__(self, debug_level: str = "INFO", debug_data: dict = None,
                 stack_trace: list = None, performance_metrics: dict = None):
        self._debug_level = debug_level
        self._debug_data = debug_data or {}
        self._stack_trace = stack_trace
        self._performance_metrics = performance_metrics or {}

    @property
    def debug_level(self) -> str:
        return self._debug_level

    @property
    def debug_data(self) -> dict:
        return self._debug_data

    @property
    def stack_trace(self) -> list:
        return self._stack_trace

    @property
    def performance_metrics(self) -> dict:
        return self._performance_metrics


class MockDebuggableComponent:
    """Mock implementation of DebuggableComponent protocol."""

    def __init__(self, component_id: str, component_type: str = "MockComponent",
                 state_data: dict = None, is_valid: bool = True):
        self.component_id = component_id
        self.component_type = component_type
        self.state_data = state_data or {"status": "active", "value": 42}
        self.is_valid = is_valid
        self.call_counts = {
            'get_component_state': 0,
            'validate_component': 0,
            'get_debug_info': 0,
        }

    def get_component_state(self) -> MockComponentState:
        self.call_counts['get_component_state'] += 1
        return MockComponentState(self.component_id, self.component_type, self.state_data)

    def validate_component(self) -> MockValidationResult:
        self.call_counts['validate_component'] += 1
        errors = [] if self.is_valid else ["Mock validation error"]
        return MockValidationResult(self.is_valid, errors)

    def get_debug_info(self, debug_level: str = "INFO") -> MockDebugInfo:
        self.call_counts['get_debug_info'] += 1
        return MockDebugInfo(
            debug_level=debug_level,
            debug_data={"mock_debug": True, "level": debug_level},
            performance_metrics={"response_time_ms": 123.45, "memory_usage_mb": 67.89}
        )

    def inspect_state(self, path: str = None) -> dict:
        return self.state_data

    def get_component_dependencies(self) -> list:
        return []

    def export_state_json(self) -> str:
        return json.dumps(self.state_data)

    def health_check(self) -> dict:
        return {"healthy": True, "status": "ok"}


class TestStateSnapshot:
    """Test StateSnapshot dataclass functionality."""

    def test_state_snapshot_creation(self):
        """Test creating a state snapshot."""
        timestamp = datetime.now()
        snapshot = StateSnapshot(
            component_id="test_comp",
            component_type="TestComponent",
            state_data={"key": "value"},
            snapshot_timestamp=timestamp,
        )

        assert snapshot.component_id == "test_comp"
        assert snapshot.component_type == "TestComponent"
        assert snapshot.state_data == {"key": "value"}
        assert snapshot.snapshot_timestamp == timestamp
        assert snapshot.validation_result is None
        assert snapshot.debug_metadata == {}
        assert snapshot.performance_metrics == {}

    def test_state_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        timestamp = datetime.now()
        snapshot = StateSnapshot(
            component_id="test_comp",
            component_type="TestComponent",
            state_data={"key": "value"},
            snapshot_timestamp=timestamp,
            validation_result={"is_valid": True},
            debug_metadata={"debug": True},
            performance_metrics={"time": 1.23},
        )

        snapshot_dict = snapshot.to_dict()

        assert snapshot_dict["component_id"] == "test_comp"
        assert snapshot_dict["component_type"] == "TestComponent"
        assert snapshot_dict["state_data"] == {"key": "value"}
        assert snapshot_dict["snapshot_timestamp"] == timestamp.isoformat()
        assert snapshot_dict["validation_result"] == {"is_valid": True}
        assert snapshot_dict["debug_metadata"] == {"debug": True}
        assert snapshot_dict["performance_metrics"] == {"time": 1.23}

    def test_state_snapshot_to_json(self):
        """Test converting snapshot to JSON."""
        timestamp = datetime.now()
        snapshot = StateSnapshot(
            component_id="test_comp",
            component_type="TestComponent",
            state_data={"key": "value"},
            snapshot_timestamp=timestamp,
        )

        json_str = snapshot.to_json()
        parsed = json.loads(json_str)

        assert parsed["component_id"] == "test_comp"
        assert parsed["component_type"] == "TestComponent"
        assert parsed["state_data"] == {"key": "value"}
        assert parsed["snapshot_timestamp"] == timestamp.isoformat()


class TestComponentStateInspector:
    """Test ComponentStateInspector functionality."""

    def test_inspector_initialization(self):
        """Test creating a state inspector."""
        inspector = ComponentStateInspector(max_history_per_component=100)

        assert inspector._max_history == 100
        assert len(inspector._components) == 0
        assert len(inspector._state_history) == 0
        assert len(inspector._component_metadata) == 0

    def test_register_component(self):
        """Test registering a component."""
        inspector = ComponentStateInspector()
        component = MockDebuggableComponent("test_comp")

        inspector.register_component("test_comp", component)

        assert "test_comp" in inspector._components
        assert inspector._components["test_comp"] == component
        assert "test_comp" in inspector._component_metadata

        metadata = inspector._component_metadata["test_comp"]
        assert metadata["component_type"] == "MockDebuggableComponent"
        assert metadata["snapshot_count"] == 0
        assert "registration_time" in metadata

    def test_unregister_component(self):
        """Test unregistering a component."""
        inspector = ComponentStateInspector()
        component = MockDebuggableComponent("test_comp")

        inspector.register_component("test_comp", component)
        assert "test_comp" in inspector._components

        result = inspector.unregister_component("test_comp")
        assert result is True
        assert "test_comp" not in inspector._components

        # Metadata should be preserved but marked as unregistered
        assert "test_comp" in inspector._component_metadata
        assert "unregistration_time" in inspector._component_metadata["test_comp"]

        # Unregistering non-existent component should return False
        result = inspector.unregister_component("nonexistent")
        assert result is False

    def test_get_registered_components(self):
        """Test getting list of registered components."""
        inspector = ComponentStateInspector()

        assert inspector.get_registered_components() == set()

        inspector.register_component("comp1", MockDebuggableComponent("comp1"))
        inspector.register_component("comp2", MockDebuggableComponent("comp2"))

        registered = inspector.get_registered_components()
        assert registered == {"comp1", "comp2"}

    def test_capture_state_snapshot(self):
        """Test capturing a state snapshot."""
        inspector = ComponentStateInspector()
        component = MockDebuggableComponent("test_comp", state_data={"status": "running"})

        inspector.register_component("test_comp", component)
        snapshot = inspector.capture_state_snapshot("test_comp")

        assert snapshot is not None
        assert snapshot.component_id == "test_comp"
        assert snapshot.component_type == "MockComponent"
        assert snapshot.state_data == {"status": "running"}
        assert snapshot.validation_result is not None
        assert snapshot.validation_result["is_valid"] is True
        assert snapshot.performance_metrics == {"response_time_ms": 123.45, "memory_usage_mb": 67.89}

        # Check that component methods were called
        assert component.call_counts["get_component_state"] == 1
        assert component.call_counts["validate_component"] == 1
        assert component.call_counts["get_debug_info"] == 1

        # Check that snapshot was added to history
        history = inspector.get_state_history("test_comp")
        assert len(history) == 1
        assert history[0] == snapshot

        # Check metadata was updated
        metadata = inspector._component_metadata["test_comp"]
        assert metadata["snapshot_count"] == 1
        assert metadata["last_snapshot_time"] == snapshot.snapshot_timestamp

    def test_capture_snapshot_nonexistent_component(self):
        """Test capturing snapshot for non-existent component."""
        inspector = ComponentStateInspector()

        snapshot = inspector.capture_state_snapshot("nonexistent")
        assert snapshot is None

    def test_capture_snapshot_with_error(self):
        """Test capturing snapshot when component throws error."""
        inspector = ComponentStateInspector()

        # Create a component that throws an error
        component = Mock()
        component.get_component_state.side_effect = Exception("Test error")

        inspector.register_component("error_comp", component)
        snapshot = inspector.capture_state_snapshot("error_comp")

        assert snapshot is not None
        assert snapshot.component_id == "error_comp"
        assert "error" in snapshot.state_data
        assert snapshot.state_data["error"] == "Test error"
        assert snapshot.debug_metadata["capture_error"] is True

    def test_get_state_history(self):
        """Test getting state history."""
        inspector = ComponentStateInspector()
        component = MockDebuggableComponent("test_comp")

        inspector.register_component("test_comp", component)

        # Capture multiple snapshots
        snapshot1 = inspector.capture_state_snapshot("test_comp")
        time.sleep(0.01)  # Ensure different timestamps
        snapshot2 = inspector.capture_state_snapshot("test_comp")
        time.sleep(0.01)
        snapshot3 = inspector.capture_state_snapshot("test_comp")

        # Get all history
        history = inspector.get_state_history("test_comp")
        assert len(history) == 3
        assert history[0] == snapshot3  # Newest first
        assert history[1] == snapshot2
        assert history[2] == snapshot1

        # Get limited history
        limited_history = inspector.get_state_history("test_comp", limit=2)
        assert len(limited_history) == 2
        assert limited_history[0] == snapshot3
        assert limited_history[1] == snapshot2

    def test_compare_states(self):
        """Test comparing two state snapshots."""
        inspector = ComponentStateInspector()

        timestamp1 = datetime.now()
        snapshot1 = StateSnapshot(
            component_id="test_comp",
            component_type="TestComponent",
            state_data={"status": "running", "value": 10},
            snapshot_timestamp=timestamp1,
            validation_result={"is_valid": True},
            performance_metrics={"cpu": 50.0, "memory": 100.0},
        )

        timestamp2 = datetime.now()
        snapshot2 = StateSnapshot(
            component_id="test_comp",
            component_type="TestComponent",
            state_data={"status": "stopped", "value": 20, "new_key": "new_value"},
            snapshot_timestamp=timestamp2,
            validation_result={"is_valid": False},
            performance_metrics={"cpu": 75.0, "memory": 150.0},
        )

        comparison = inspector.compare_states("test_comp", snapshot1, snapshot2)

        assert comparison["component_id"] == "test_comp"
        assert comparison["time_delta_seconds"] == pytest.approx((timestamp2 - timestamp1).total_seconds(), rel=1e-3)

        differences = comparison["differences"]
        assert "new_key" in differences["added_keys"]
        assert differences["removed_keys"] == []
        assert set(differences["changed_keys"]) == {"status", "value"}

        changes = differences["changes"]
        assert changes["status"]["old"] == "running"
        assert changes["status"]["new"] == "stopped"
        assert changes["value"]["old"] == 10
        assert changes["value"]["new"] == 20

        assert comparison["validation_changed"] is True

        perf_changes = comparison["performance_changes"]
        assert "cpu" in perf_changes
        assert perf_changes["cpu"]["old"] == 50.0
        assert perf_changes["cpu"]["new"] == 75.0
        assert perf_changes["cpu"]["delta"] == 25.0

        summary = comparison["summary"]
        assert summary["has_changes"] is True
        assert summary["change_count"] == 3  # added + changed keys
        assert summary["validation_status_changed"] is True

    def test_generate_llm_friendly_report(self):
        """Test generating LLM-friendly report."""
        inspector = ComponentStateInspector()

        # Add components with different states
        comp1 = MockDebuggableComponent("comp1", state_data={"status": "healthy"})
        comp2 = MockDebuggableComponent("comp2", state_data={"status": "warning"}, is_valid=False)

        inspector.register_component("comp1", comp1)
        inspector.register_component("comp2", comp2)

        # Generate report for all components
        report = inspector.generate_llm_friendly_report()

        assert "Component State Inspection Report" in report
        assert "comp1" in report
        assert "comp2" in report
        assert "Summary" in report
        assert "Total registered components: 2" in report
        assert "Analysis" in report

        # Generate report for specific component
        comp1_report = inspector.generate_llm_friendly_report("comp1")
        assert "comp1" in comp1_report
        assert "comp2" not in comp1_report

    def test_generate_report_no_components(self):
        """Test generating report when no components are registered."""
        inspector = ComponentStateInspector()

        report = inspector.generate_llm_friendly_report()
        assert "No components available for reporting" in report

    def test_export_full_state(self):
        """Test exporting full state."""
        inspector = ComponentStateInspector()
        component = MockDebuggableComponent("test_comp")

        inspector.register_component("test_comp", component)
        inspector.capture_state_snapshot("test_comp")

        export_data = inspector.export_full_state()

        assert "export_timestamp" in export_data
        assert export_data["registered_components"] == ["test_comp"]
        assert "test_comp" in export_data["component_metadata"]
        assert "test_comp" in export_data["current_snapshots"]
        assert "test_comp" in export_data["state_history_summary"]

        history_summary = export_data["state_history_summary"]["test_comp"]
        assert history_summary["total_snapshots"] == 2  # One from capture_state_snapshot, one from export_full_state

    def test_thread_safety(self):
        """Test thread safety of the inspector."""
        inspector = ComponentStateInspector()
        component = MockDebuggableComponent("test_comp")
        inspector.register_component("test_comp", component)

        snapshots = []
        errors = []

        def capture_snapshots():
            try:
                for _ in range(10):
                    snapshot = inspector.capture_state_snapshot("test_comp")
                    if snapshot:
                        snapshots.append(snapshot)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Run multiple threads capturing snapshots
        threads = []
        for _ in range(5):
            thread = Thread(target=capture_snapshots)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors and multiple snapshots
        assert len(errors) == 0
        assert len(snapshots) == 50  # 5 threads * 10 snapshots each

        # History should contain all snapshots
        history = inspector.get_state_history("test_comp", limit=100)
        assert len(history) == 50

    def test_max_history_limit(self):
        """Test that history is limited to max_history_per_component."""
        inspector = ComponentStateInspector(max_history_per_component=3)
        component = MockDebuggableComponent("test_comp")

        inspector.register_component("test_comp", component)

        # Capture more snapshots than the limit
        snapshots = []
        for i in range(5):
            component.state_data = {"iteration": i}
            snapshot = inspector.capture_state_snapshot("test_comp")
            snapshots.append(snapshot)
            time.sleep(0.01)

        # History should be limited to 3
        history = inspector.get_state_history("test_comp", limit=10)
        assert len(history) == 3

        # Should contain the most recent snapshots
        assert history[0] == snapshots[4]  # Most recent
        assert history[1] == snapshots[3]
        assert history[2] == snapshots[2]
