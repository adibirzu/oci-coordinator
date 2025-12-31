"""
ShowOCI Integration Module.

Provides a wrapper around Oracle's ShowOCI tool for comprehensive
OCI resource discovery and caching.

ShowOCI is part of the OCI Python SDK:
https://github.com/oracle/oci-python-sdk/tree/master/examples/showoci

Usage:
    from src.showoci import ShowOCIRunner

    runner = ShowOCIRunner(profile="DEFAULT")
    result = await runner.run_discovery(resource_types=["compute", "network"])

    # Access discovered resources
    instances = result.get_instances()
    vcns = result.get_vcns()
"""

from src.showoci.runner import ShowOCIRunner
from src.showoci.parser import ShowOCIParser
from src.showoci.cache_loader import ShowOCICacheLoader

__all__ = [
    "ShowOCIRunner",
    "ShowOCIParser",
    "ShowOCICacheLoader",
]
