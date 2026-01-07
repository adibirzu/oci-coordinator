"""
OCI Integration Module.

Provides multi-tenancy support and OCI resource management.
"""

from src.oci.discovery import DiscoveryService, initialize_discovery
from src.oci.profile_manager import ProfileInfo, ProfileManager, get_profile_manager
from src.oci.tenancy_manager import Compartment, TenancyConfig, TenancyManager

__all__ = [
    "Compartment",
    "DiscoveryService",
    "ProfileInfo",
    "ProfileManager",
    "TenancyConfig",
    "TenancyManager",
    "get_profile_manager",
    "initialize_discovery",
]
