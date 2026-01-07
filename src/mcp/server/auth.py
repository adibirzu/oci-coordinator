import configparser
import os

import oci


def _find_profile_case_insensitive(config_file: str, profile: str) -> str | None:
    """Find a profile name in OCI config file with case-insensitive matching.

    Args:
        config_file: Path to OCI config file
        profile: Profile name to find (case-insensitive)

    Returns:
        The actual profile name from config file, or None if not found
    """
    expanded_path = os.path.expanduser(config_file)
    if not os.path.exists(expanded_path):
        return None

    try:
        parser = configparser.ConfigParser()
        parser.read(expanded_path)
        profile_lower = profile.lower()
        for section in parser.sections():
            if section.lower() == profile_lower:
                return section
    except Exception:
        pass
    return None


def get_oci_config(profile: str | None = None) -> dict:
    """Get OCI configuration from environment or file.

    Args:
        profile: OCI config profile name. Defaults to OCI_CLI_PROFILE env var.
                 Profile lookup is case-insensitive.

    Returns:
        OCI configuration dictionary
    """
    config_file = os.getenv("OCI_CONFIG_FILE", "~/.oci/config")
    profile = profile or os.getenv("OCI_CLI_PROFILE", "DEFAULT")

    try:
        return oci.config.from_file(file_location=config_file, profile_name=profile)
    except oci.exceptions.ProfileNotFound:
        # Try case-insensitive lookup
        actual_profile = _find_profile_case_insensitive(config_file, profile)
        if actual_profile:
            try:
                return oci.config.from_file(
                    file_location=config_file, profile_name=actual_profile
                )
            except Exception:
                return {}
        return {}
    except Exception:
        # Fallback to instance principals or other methods if needed
        return {}


def get_oci_config_with_region(profile: str | None = None, region: str | None = None) -> dict:
    """Get OCI configuration with optional region override.

    Args:
        profile: OCI config profile name
        region: OCI region (e.g., 'us-ashburn-1'). Overrides profile region.

    Returns:
        OCI configuration dictionary with region set
    """
    config = get_oci_config(profile)
    if region:
        config["region"] = region
    return config


# Client cache for performance (keyed by profile+region)
_client_cache: dict[str, object] = {}

def get_compute_client():
    """Get OCI Compute client."""
    config = get_oci_config()
    return oci.core.ComputeClient(config)

def get_network_client():
    """Get OCI Virtual Network client."""
    config = get_oci_config()
    return oci.core.VirtualNetworkClient(config)

def get_usage_client(profile: str | None = None):
    """Get OCI Usage client."""
    config = get_oci_config(profile)
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


# ─────────────────────────────────────────────────────────────────────────────
# Database Observability Clients (DB Management, OpsInsights, Log Analytics)
# ─────────────────────────────────────────────────────────────────────────────


def get_database_management_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.database_management.DbManagementClient:
    """Get OCI Database Management client.

    Args:
        profile: OCI config profile (e.g., 'EMDEMO'). Defaults to OCI_CLI_PROFILE.
        region: OCI region override (e.g., 'us-ashburn-1').

    Returns:
        DbManagementClient for managed database operations
    """
    # Normalize profile to lowercase for consistent cache keys
    profile_key = (profile or "default").lower()
    cache_key = f"dbmgmt:{profile_key}:{region or 'default'}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    config = get_oci_config_with_region(profile, region)
    client = oci.database_management.DbManagementClient(config)
    _client_cache[cache_key] = client
    return client


def get_opsi_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.opsi.OperationsInsightsClient:
    """Get OCI Operations Insights client.

    Args:
        profile: OCI config profile (e.g., 'EMDEMO'). Defaults to OCI_CLI_PROFILE.
        region: OCI region override (e.g., 'uk-london-1').

    Returns:
        OperationsInsightsClient for OPSI operations
    """
    # Normalize profile to lowercase for consistent cache keys
    profile_key = (profile or "default").lower()
    cache_key = f"opsi:{profile_key}:{region or 'default'}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    config = get_oci_config_with_region(profile, region)
    client = oci.opsi.OperationsInsightsClient(config)
    _client_cache[cache_key] = client
    return client


def get_log_analytics_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.log_analytics.LogAnalyticsClient:
    """Get OCI Log Analytics client.

    Args:
        profile: OCI config profile (e.g., 'EMDEMO'). Defaults to OCI_CLI_PROFILE.
        region: OCI region override (e.g., 'us-ashburn-1').

    Returns:
        LogAnalyticsClient for log analytics operations
    """
    # Normalize profile to lowercase for consistent cache keys
    profile_key = (profile or "default").lower()
    cache_key = f"logan:{profile_key}:{region or 'default'}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    config = get_oci_config_with_region(profile, region)
    client = oci.log_analytics.LogAnalyticsClient(config)
    _client_cache[cache_key] = client
    return client


def get_diagnosability_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.database_management.DiagnosabilityClient:
    """Get OCI Diagnosability client for AWR/ADDM reports.

    Args:
        profile: OCI config profile (e.g., 'EMDEMO'). Defaults to OCI_CLI_PROFILE.
        region: OCI region override.

    Returns:
        DiagnosabilityClient for AWR/ADDM operations
    """
    # Normalize profile to lowercase for consistent cache keys
    profile_key = (profile or "default").lower()
    cache_key = f"diag:{profile_key}:{region or 'default'}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    config = get_oci_config_with_region(profile, region)
    client = oci.database_management.DiagnosabilityClient(config)
    _client_cache[cache_key] = client
    return client


def clear_client_cache():
    """Clear the client cache. Useful for testing or profile changes."""
    global _client_cache
    _client_cache.clear()
