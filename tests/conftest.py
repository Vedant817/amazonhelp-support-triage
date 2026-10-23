"""
Pytest configuration and fixtures.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def test_data_dir():
    """Return test data directory."""
    return os.path.join(os.path.dirname(__file__), "data")

@pytest.fixture
def sample_query():
    """Sample customer query for testing."""
    return "My package shows delivered but I never received it"

@pytest.fixture
def sample_exemplars():
    """Sample exemplars for testing."""
    return [
        {
            "reply_id": "1",
            "customer_text": "Package late",
            "reply_text": "Check amazon.com/orders for status",
            "intent_label": "ORDER_TRACKING_AND_DELAY",
            "resolution_act": "LINK_ORDERS",
        },
        {
            "reply_id": "2",
            "customer_text": "Item damaged",
            "reply_text": "Check amazon.com/returns for replacement",
            "intent_label": "DAMAGED_OR_WRONG_ITEM",
            "resolution_act": "LINK_RETURNS",
        },
    ]


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "safety: marks tests as safety-critical")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Mark tests based on name patterns
        if "slow" in item.name or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        if "safety" in str(item.fspath) or "hard_gate" in item.name:
            item.add_marker(pytest.mark.safety)