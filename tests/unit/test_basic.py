"""Basic smoke tests to verify test infrastructure."""

import pytest


def test_basic_assertion():
    """Test that basic assertions work."""
    assert True


def test_basic_math():
    """Test basic math operations."""
    assert 1 + 1 == 2


def test_import_main_module():
    """Test that we can import the main module."""
    from mcp_server_cheap_llm import __version__

    assert __version__ == "0.1.0"


def test_import_core_models():
    """Test that we can import core models."""
    from mcp_server_cheap_llm.core.models import ProviderStatus, ProviderType

    assert ProviderType.GEMINI == "gemini"
    assert ProviderStatus.AVAILABLE == "available"
