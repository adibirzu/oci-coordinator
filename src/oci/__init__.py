"""
OCI Integration Module.

Provides multi-tenancy support and OCI resource management.
"""

from src.oci.discovery import DiscoveryService, initialize_discovery
from src.oci.tenancy_manager import Compartment, TenancyConfig, TenancyManager

__all__ = [
    "TenancyManager",
    "TenancyConfig",
    "Compartment",
    "DiscoveryService",
    "initialize_discovery",
]
