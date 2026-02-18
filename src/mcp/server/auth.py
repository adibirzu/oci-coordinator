from __future__ import annotations

import configparser
import os
from typing import Any

import structlog

import oci

logger = structlog.get_logger(__name__)


def _find_profile_case_insensitive(config_file: str, profile: str) -> str | None:
    """Find a profile name in OCI config file with case-insensitive matching."""
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
        return None
    return None


def _create_signer(auth_mode: str) -> Any:
    """Create OCI signer for non-file auth modes."""
    mode = auth_mode.strip().lower()

    if mode in {"resource_principal", "resource-principal"}:
        return oci.auth.signers.get_resource_principals_signer()

    if mode in {"instance_principal", "instance-principal"}:
        return oci.auth.signers.InstancePrincipalsSecurityTokenSigner()

    if mode in {"oke_workload_identity", "workload_identity", "oke-workload-identity"}:
        # SDK method availability depends on OCI SDK version.
        getter = getattr(
            oci.auth.signers,
            "get_oke_workload_identity_resource_principal_signer",
            None,
        )
        if getter is None:
            msg = (
                "OCI SDK does not expose OKE workload identity signer. "
                "Upgrade OCI Python SDK."
            )
            raise RuntimeError(msg)
        return getter()

    msg = f"Unsupported OCI_CLI_AUTH mode: {auth_mode}"
    raise ValueError(msg)


def _effective_auth_mode() -> str:
    """Return normalized auth mode from OCI_CLI_AUTH."""
    mode = (os.getenv("OCI_CLI_AUTH") or "").strip().lower()
    if not mode or mode in {"api_key", "api-key", "config_file", "none"}:
        return "api_key"
    return mode


def _build_signer_config(
    signer: Any,
    profile: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Build minimal OCI config dict for signer-based auth."""
    config: dict[str, Any] = {}

    effective_region = (
        region
        or os.getenv("OCI_REGION")
        or getattr(signer, "region", None)
        or getattr(signer, "_region", None)
    )
    if effective_region:
        config["region"] = effective_region

    tenancy = (
        getattr(signer, "tenancy_id", None)
        or getattr(signer, "tenancy", None)
        or os.getenv("OCI_TENANCY_OCID")
    )
    if tenancy:
        config["tenancy"] = tenancy

    if profile:
        config["profile"] = profile

    return config


def _resolve_oci_auth(
    profile: str | None = None,
    region: str | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Resolve OCI config + optional signer from runtime environment."""
    auth_mode = _effective_auth_mode()

    if auth_mode == "api_key":
        config = _load_oci_config_from_file(profile)
        if region:
            config["region"] = region
        return config, None

    try:
        signer = _create_signer(auth_mode)
        config = _build_signer_config(signer, profile=profile, region=region)
        logger.info("Using OCI signer auth", auth_mode=auth_mode, region=config.get("region"))
        return config, signer
    except Exception as e:
        # Graceful fallback to file-based config so local/dev keeps working.
        logger.warning(
            "Failed to initialize signer auth, falling back to OCI config file",
            auth_mode=auth_mode,
            error=str(e),
        )
        config = _load_oci_config_from_file(profile)
        if region:
            config["region"] = region
        return config, None


def _build_client(
    client_cls: Any,
    profile: str | None = None,
    region: str | None = None,
) -> Any:
    """Create OCI client using either config-file auth or signer auth."""
    config, signer = _resolve_oci_auth(profile=profile, region=region)
    if signer is not None:
        return client_cls(config, signer=signer)
    return client_cls(config)


def _load_oci_config_from_file(profile: str | None = None) -> dict[str, Any]:
    """Load OCI configuration from file-based profile."""
    config_file = os.getenv("OCI_CONFIG_FILE", "~/.oci/config")
    profile = profile or os.getenv("OCI_CLI_PROFILE", "DEFAULT")

    try:
        return oci.config.from_file(file_location=config_file, profile_name=profile)
    except oci.exceptions.ProfileNotFound:
        actual_profile = _find_profile_case_insensitive(config_file, profile)
        if actual_profile:
            try:
                return oci.config.from_file(
                    file_location=config_file,
                    profile_name=actual_profile,
                )
            except Exception:
                return {}
        return {}
    except Exception:
        return {}


def get_oci_config(profile: str | None = None) -> dict[str, Any]:
    """Get OCI configuration for the active auth mode.

    For `api_key`, reads from OCI config file/profile.
    For signer modes, returns signer-derived config (region/tenancy when available).
    """
    auth_mode = _effective_auth_mode()
    if auth_mode == "api_key":
        return _load_oci_config_from_file(profile)

    try:
        signer = _create_signer(auth_mode)
        return _build_signer_config(signer, profile=profile)
    except Exception as e:
        logger.warning(
            "Failed to build auth config from signer, falling back to file config",
            auth_mode=auth_mode,
            error=str(e),
        )
        return _load_oci_config_from_file(profile)


def get_oci_config_with_region(
    profile: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Get OCI config resolved for active auth mode plus optional region override."""
    config, _ = _resolve_oci_auth(profile=profile, region=region)
    if region:
        config["region"] = region
    return config


# Client cache for performance (keyed by profile+region+auth mode)
_client_cache: dict[str, object] = {}


def _cache_key(service: str, profile: str | None, region: str | None) -> str:
    profile_key = (profile or "default").lower()
    auth_key = _effective_auth_mode()
    return f"{service}:{profile_key}:{region or 'default'}:{auth_key}"


def get_compute_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.core.ComputeClient:
    """Get OCI Compute client."""
    key = _cache_key("compute", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(oci.core.ComputeClient, profile=profile, region=region)
    _client_cache[key] = client
    return client


def get_network_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.core.VirtualNetworkClient:
    """Get OCI Virtual Network client."""
    key = _cache_key("network", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(oci.core.VirtualNetworkClient, profile=profile, region=region)
    _client_cache[key] = client
    return client


def get_usage_client(profile: str | None = None):
    """Get OCI Usage client."""
    key = _cache_key("usage", profile, None)
    if key in _client_cache:
        return _client_cache[key]

    client = _build_client(oci.usage_api.UsageapiClient, profile=profile)
    _client_cache[key] = client
    return client


def get_identity_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.identity.IdentityClient:
    """Get OCI Identity client."""
    key = _cache_key("identity", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(oci.identity.IdentityClient, profile=profile, region=region)
    _client_cache[key] = client
    return client


def get_monitoring_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.monitoring.MonitoringClient:
    """Get OCI Monitoring client."""
    key = _cache_key("monitoring", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(oci.monitoring.MonitoringClient, profile=profile, region=region)
    _client_cache[key] = client
    return client


def get_logging_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.loggingingestion.LoggingClient:
    """Get OCI Logging ingestion client."""
    key = _cache_key("logging", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(oci.loggingingestion.LoggingClient, profile=profile, region=region)
    _client_cache[key] = client
    return client


def get_logging_search_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.loggingsearch.LogSearchClient:
    """Get OCI Logging Search client."""
    key = _cache_key("loggingsearch", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(oci.loggingsearch.LogSearchClient, profile=profile, region=region)
    _client_cache[key] = client
    return client


def get_database_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.database.DatabaseClient:
    """Get OCI Database client for Autonomous Database operations."""
    key = _cache_key("database", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(oci.database.DatabaseClient, profile=profile, region=region)
    _client_cache[key] = client
    return client


def get_database_management_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.database_management.DbManagementClient:
    """Get OCI Database Management client."""
    key = _cache_key("dbmgmt", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(
        oci.database_management.DbManagementClient,
        profile=profile,
        region=region,
    )
    _client_cache[key] = client
    return client


def get_opsi_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.opsi.OperationsInsightsClient:
    """Get OCI Operations Insights client."""
    key = _cache_key("opsi", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(
        oci.opsi.OperationsInsightsClient,
        profile=profile,
        region=region,
    )
    _client_cache[key] = client
    return client


def get_log_analytics_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.log_analytics.LogAnalyticsClient:
    """Get OCI Log Analytics client."""
    key = _cache_key("logan", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(
        oci.log_analytics.LogAnalyticsClient,
        profile=profile,
        region=region,
    )
    _client_cache[key] = client
    return client


def get_diagnosability_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.database_management.DiagnosabilityClient:
    """Get OCI Diagnosability client for AWR/ADDM reports."""
    key = _cache_key("diag", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(
        oci.database_management.DiagnosabilityClient,
        profile=profile,
        region=region,
    )
    _client_cache[key] = client
    return client


def get_cloud_guard_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.cloud_guard.CloudGuardClient:
    """Get OCI Cloud Guard client."""
    key = _cache_key("cloudguard", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(
        oci.cloud_guard.CloudGuardClient,
        profile=profile,
        region=region,
    )
    _client_cache[key] = client
    return client


def get_audit_client(
    profile: str | None = None,
    region: str | None = None,
) -> oci.audit.AuditClient:
    """Get OCI Audit client."""
    key = _cache_key("audit", profile, region)
    if key in _client_cache:
        return _client_cache[key]  # type: ignore[return-value]

    client = _build_client(oci.audit.AuditClient, profile=profile, region=region)
    _client_cache[key] = client
    return client


def clear_client_cache():
    """Clear the client cache. Useful for testing or profile changes."""
    _client_cache.clear()
