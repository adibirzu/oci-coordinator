"""
ShowOCI Runner - Python API wrapper for ShowOCI discovery.

This module provides a Python interface to ShowOCI functionality
by importing and using ShowOCI classes directly (no subprocess).

ShowOCI Source: https://github.com/oracle/oci-python-sdk/tree/master/examples/showoci
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ShowOCIConfig:
    """Configuration for ShowOCI execution."""

    profile: str = "DEFAULT"
    """OCI config profile to use."""

    regions: list[str] | None = None
    """Specific regions to scan (None = all subscribed regions)."""

    compartment_ocid: str | None = None
    """Specific compartment OCID to scan."""

    resource_types: list[str] | None = None
    """Resource types to discover: compute, network, database, storage, all."""

    use_instance_principal: bool = False
    """Use instance principal authentication."""

    threads: int = 8
    """Number of parallel threads for discovery."""

    cache_dir: str = "/tmp/showoci_cache"
    """Directory for caching ShowOCI output."""


@dataclass
class ShowOCIResult:
    """Result of ShowOCI execution."""

    success: bool
    """Whether execution completed successfully."""

    profile: str
    """OCI profile used."""

    data: dict[str, Any] = field(default_factory=dict)
    """Parsed discovery data."""

    regions_scanned: list[str] = field(default_factory=list)
    """Regions that were scanned."""

    compartments_scanned: int = 0
    """Number of compartments scanned."""

    resource_counts: dict[str, int] = field(default_factory=dict)
    """Count of resources by type."""

    duration_seconds: float = 0.0
    """Execution duration."""

    error: str | None = None
    """Error message if failed."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    """Execution timestamp."""

    def get_instances(self) -> list[dict[str, Any]]:
        """Get all discovered compute instances."""
        return self.data.get("instances", [])

    def get_autonomous_databases(self) -> list[dict[str, Any]]:
        """Get all discovered autonomous databases."""
        return self.data.get("autonomous_databases", [])

    def get_db_systems(self) -> list[dict[str, Any]]:
        """Get all discovered DB systems."""
        return self.data.get("db_systems", [])

    def get_vcns(self) -> list[dict[str, Any]]:
        """Get all discovered VCNs."""
        return self.data.get("vcns", [])

    def get_subnets(self) -> list[dict[str, Any]]:
        """Get all discovered subnets."""
        return self.data.get("subnets", [])

    def get_compartments(self) -> list[dict[str, Any]]:
        """Get all discovered compartments."""
        return self.data.get("compartments", [])

    def get_block_volumes(self) -> list[dict[str, Any]]:
        """Get all discovered block volumes."""
        return self.data.get("block_volumes", [])

    def get_buckets(self) -> list[dict[str, Any]]:
        """Get all discovered object storage buckets."""
        return self.data.get("buckets", [])


class ShowOCIRunner:
    """
    Runner for ShowOCI-style discovery using OCI SDK directly.

    Instead of invoking ShowOCI as a subprocess, this class implements
    similar discovery logic using the OCI Python SDK directly for safety.

    Example:
        runner = ShowOCIRunner(profile="DEFAULT")
        result = await runner.run_discovery(resource_types=["compute"])

        for instance in result.get_instances():
            print(f"{instance['display_name']}: {instance['lifecycle_state']}")
    """

    def __init__(self, config: ShowOCIConfig | None = None, **kwargs):
        """
        Initialize ShowOCI runner.

        Args:
            config: ShowOCIConfig instance
            **kwargs: Override config values
        """
        self.config = config or ShowOCIConfig(**kwargs)
        self._logger = logger.bind(component="ShowOCIRunner", profile=self.config.profile)
        self._oci_config = None
        self._clients: dict[str, Any] = {}

    def _init_oci_clients(self) -> None:
        """Initialize OCI SDK clients."""
        import oci

        if self.config.use_instance_principal:
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            self._oci_config = {"region": signer.region}
            self._signer = signer
        else:
            self._oci_config = oci.config.from_file(profile_name=self.config.profile)
            self._signer = None

    def _get_client(self, client_class: type) -> Any:
        """Get or create an OCI client."""

        client_name = client_class.__name__
        if client_name not in self._clients:
            if self._signer:
                self._clients[client_name] = client_class(
                    config=self._oci_config,
                    signer=self._signer,
                )
            else:
                self._clients[client_name] = client_class(self._oci_config)
        return self._clients[client_name]

    async def _discover_compartments(self, tenancy_id: str) -> list[dict[str, Any]]:
        """Discover all compartments in tenancy."""
        import oci

        client = self._get_client(oci.identity.IdentityClient)
        compartments = []

        try:
            # Get root compartment (tenancy)
            tenancy = client.get_tenancy(tenancy_id).data
            compartments.append({
                "id": tenancy.id,
                "name": tenancy.name,
                "description": tenancy.description,
                "lifecycle_state": "ACTIVE",
                "is_root": True,
            })

            # Get all child compartments
            response = oci.pagination.list_call_get_all_results(
                client.list_compartments,
                compartment_id=tenancy_id,
                compartment_id_in_subtree=True,
                access_level="ACCESSIBLE",
                lifecycle_state="ACTIVE",
            )

            for comp in response.data:
                compartments.append({
                    "id": comp.id,
                    "name": comp.name,
                    "description": comp.description,
                    "lifecycle_state": comp.lifecycle_state,
                    "parent_id": comp.compartment_id,
                    "time_created": str(comp.time_created) if comp.time_created else None,
                })

        except Exception as e:
            self._logger.error("Failed to discover compartments", error=str(e))

        return compartments

    async def _discover_instances(
        self, compartment_id: str, region: str | None = None
    ) -> list[dict[str, Any]]:
        """Discover compute instances in a compartment."""
        import oci

        client = self._get_client(oci.core.ComputeClient)
        instances = []

        try:
            response = oci.pagination.list_call_get_all_results(
                client.list_instances,
                compartment_id=compartment_id,
            )

            for inst in response.data:
                instances.append({
                    "id": inst.id,
                    "display_name": inst.display_name,
                    "lifecycle_state": inst.lifecycle_state,
                    "shape": inst.shape,
                    "availability_domain": inst.availability_domain,
                    "fault_domain": inst.fault_domain,
                    "compartment_id": inst.compartment_id,
                    "region": inst.region,
                    "time_created": str(inst.time_created) if inst.time_created else None,
                })

        except Exception as e:
            self._logger.debug("Failed to discover instances", compartment=compartment_id[:20], error=str(e))

        return instances

    async def _discover_vcns(self, compartment_id: str) -> list[dict[str, Any]]:
        """Discover VCNs in a compartment."""
        import oci

        client = self._get_client(oci.core.VirtualNetworkClient)
        vcns = []

        try:
            response = oci.pagination.list_call_get_all_results(
                client.list_vcns,
                compartment_id=compartment_id,
            )

            for vcn in response.data:
                vcns.append({
                    "id": vcn.id,
                    "display_name": vcn.display_name,
                    "cidr_block": vcn.cidr_block,
                    "cidr_blocks": vcn.cidr_blocks,
                    "lifecycle_state": vcn.lifecycle_state,
                    "compartment_id": vcn.compartment_id,
                    "dns_label": vcn.dns_label,
                    "time_created": str(vcn.time_created) if vcn.time_created else None,
                })

        except Exception as e:
            self._logger.debug("Failed to discover VCNs", compartment=compartment_id[:20], error=str(e))

        return vcns

    async def _discover_subnets(self, compartment_id: str) -> list[dict[str, Any]]:
        """Discover subnets in a compartment."""
        import oci

        client = self._get_client(oci.core.VirtualNetworkClient)
        subnets = []

        try:
            response = oci.pagination.list_call_get_all_results(
                client.list_subnets,
                compartment_id=compartment_id,
            )

            for subnet in response.data:
                subnets.append({
                    "id": subnet.id,
                    "display_name": subnet.display_name,
                    "cidr_block": subnet.cidr_block,
                    "vcn_id": subnet.vcn_id,
                    "availability_domain": subnet.availability_domain,
                    "lifecycle_state": subnet.lifecycle_state,
                    "compartment_id": subnet.compartment_id,
                    "prohibit_public_ip_on_vnic": subnet.prohibit_public_ip_on_vnic,
                })

        except Exception as e:
            self._logger.debug("Failed to discover subnets", compartment=compartment_id[:20], error=str(e))

        return subnets

    async def _discover_autonomous_databases(self, compartment_id: str) -> list[dict[str, Any]]:
        """Discover autonomous databases in a compartment."""
        import oci

        client = self._get_client(oci.database.DatabaseClient)
        databases = []

        try:
            response = oci.pagination.list_call_get_all_results(
                client.list_autonomous_databases,
                compartment_id=compartment_id,
            )

            for db in response.data:
                databases.append({
                    "id": db.id,
                    "display_name": db.display_name,
                    "db_name": db.db_name,
                    "lifecycle_state": db.lifecycle_state,
                    "db_workload": db.db_workload,
                    "cpu_core_count": db.cpu_core_count,
                    "data_storage_size_in_tbs": db.data_storage_size_in_tbs,
                    "compartment_id": db.compartment_id,
                    "is_free_tier": db.is_free_tier,
                    "time_created": str(db.time_created) if db.time_created else None,
                })

        except Exception as e:
            self._logger.debug("Failed to discover autonomous DBs", compartment=compartment_id[:20], error=str(e))

        return databases

    async def _discover_block_volumes(self, compartment_id: str) -> list[dict[str, Any]]:
        """Discover block volumes in a compartment."""
        import oci

        client = self._get_client(oci.core.BlockstorageClient)
        volumes = []

        try:
            response = oci.pagination.list_call_get_all_results(
                client.list_volumes,
                compartment_id=compartment_id,
            )

            for vol in response.data:
                volumes.append({
                    "id": vol.id,
                    "display_name": vol.display_name,
                    "size_in_gbs": vol.size_in_gbs,
                    "lifecycle_state": vol.lifecycle_state,
                    "availability_domain": vol.availability_domain,
                    "compartment_id": vol.compartment_id,
                    "vpus_per_gb": vol.vpus_per_gb,
                    "time_created": str(vol.time_created) if vol.time_created else None,
                })

        except Exception as e:
            self._logger.debug("Failed to discover block volumes", compartment=compartment_id[:20], error=str(e))

        return volumes

    async def run_discovery(
        self,
        resource_types: list[str] | None = None,
        compartment_ids: list[str] | None = None,
    ) -> ShowOCIResult:
        """
        Run resource discovery.

        Args:
            resource_types: Types to discover (compute, network, database, storage, all)
            compartment_ids: Specific compartments to scan (None = all)

        Returns:
            ShowOCIResult with discovered resources
        """
        start_time = datetime.now(UTC)
        resource_types = resource_types or self.config.resource_types or ["all"]

        # Normalize resource types
        if "all" in resource_types:
            resource_types = ["compute", "network", "database", "storage"]

        self._logger.info("Starting discovery", resource_types=resource_types)

        try:
            # Initialize OCI clients
            self._init_oci_clients()

            # Get tenancy ID
            tenancy_id = self._oci_config.get("tenancy")

            # Discover compartments
            compartments = await self._discover_compartments(tenancy_id)

            # Filter compartments if specified
            if compartment_ids:
                compartments = [c for c in compartments if c["id"] in compartment_ids]
            elif self.config.compartment_ocid:
                compartments = [c for c in compartments if c["id"] == self.config.compartment_ocid]

            # Initialize result data
            data: dict[str, list] = {
                "compartments": compartments,
                "instances": [],
                "vcns": [],
                "subnets": [],
                "autonomous_databases": [],
                "db_systems": [],
                "block_volumes": [],
                "buckets": [],
            }

            # Discover resources in each compartment
            for comp in compartments:
                comp_id = comp["id"]

                if "compute" in resource_types:
                    instances = await self._discover_instances(comp_id)
                    data["instances"].extend(instances)

                if "network" in resource_types:
                    vcns = await self._discover_vcns(comp_id)
                    data["vcns"].extend(vcns)
                    subnets = await self._discover_subnets(comp_id)
                    data["subnets"].extend(subnets)

                if "database" in resource_types:
                    adbs = await self._discover_autonomous_databases(comp_id)
                    data["autonomous_databases"].extend(adbs)

                if "storage" in resource_types:
                    volumes = await self._discover_block_volumes(comp_id)
                    data["block_volumes"].extend(volumes)

            duration = (datetime.now(UTC) - start_time).total_seconds()

            result = ShowOCIResult(
                success=True,
                profile=self.config.profile,
                data=data,
                compartments_scanned=len(compartments),
                duration_seconds=duration,
                resource_counts={
                    "compartments": len(data["compartments"]),
                    "instances": len(data["instances"]),
                    "vcns": len(data["vcns"]),
                    "subnets": len(data["subnets"]),
                    "autonomous_databases": len(data["autonomous_databases"]),
                    "block_volumes": len(data["block_volumes"]),
                },
            )

            self._logger.info(
                "Discovery completed",
                duration_seconds=duration,
                compartments=result.compartments_scanned,
                resources=result.resource_counts,
            )

            return result

        except Exception as e:
            self._logger.error("Discovery failed", error=str(e))
            return ShowOCIResult(
                success=False,
                profile=self.config.profile,
                error=str(e),
                duration_seconds=(datetime.now(UTC) - start_time).total_seconds(),
            )

    async def run_quick_discovery(self) -> ShowOCIResult:
        """Run quick discovery (compute + network only)."""
        return await self.run_discovery(resource_types=["compute", "network"])

    async def run_full_discovery(self) -> ShowOCIResult:
        """Run full discovery (all resources)."""
        return await self.run_discovery(resource_types=["all"])
