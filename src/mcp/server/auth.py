import oci
import os
from typing import Optional

def get_oci_config() -> dict:
    """Get OCI configuration from environment or file."""
    config_file = os.getenv("OCI_CONFIG_FILE", "~/.oci/config")
    profile = os.getenv("OCI_CLI_PROFILE", "DEFAULT")
    
    try:
        return oci.config.from_file(file_location=config_file, profile_name=profile)
    except Exception:
        # Fallback to instance principals or other methods if needed
        return {}

def get_compute_client():
    """Get OCI Compute client."""
    config = get_oci_config()
    return oci.core.ComputeClient(config)

def get_network_client():
    """Get OCI Virtual Network client."""
    config = get_oci_config()
    return oci.core.VirtualNetworkClient(config)

def get_usage_client():
    """Get OCI Usage client."""
    config = get_oci_config()
    return oci.usage_api.UsageapiClient(config)

def get_identity_client():
    """Get OCI Identity client."""
    config = get_oci_config()
    return oci.identity.IdentityClient(config)

def get_monitoring_client():
    """Get OCI Monitoring client."""
    config = get_oci_config()
    return oci.monitoring.MonitoringClient(config)

def get_logging_client():
    """Get OCI Logging Management client."""
    config = get_oci_config()
    return oci.loggingingestion.LoggingClient(config) # For ingestion
    # Note: Logging search uses loggingsearch client
    
def get_logging_search_client():
    """Get OCI Logging Search client."""
    config = get_oci_config()
    return oci.loggingsearch.LogSearchClient(config)