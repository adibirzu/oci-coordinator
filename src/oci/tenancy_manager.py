"""
OCI Tenancy Manager for multi-tenancy support.

Handles multiple OCI profiles and provides compartment discovery
across tenancies. Caches compartment information in Redis for fast access.

Usage:
    from src.oci.tenancy_manager import TenancyManager

    manager = TenancyManager()
    await manager.initialize()

    # Get compartment OCID by name
    ocid = await manager.get_compartment_ocid("adrian_birzu")

    # List all compartments
    compartments = await manager.list_compartments()
"""

import os
from dataclasses import dataclass, field
from typing import Any

import structlog
from opentelemetry import trace

logger = structlog.get_logger(__name__)


@dataclass
class TenancyConfig:
    """Configuration for a single OCI tenancy."""

    profile_name: str
    tenancy_ocid: str
    region: str
    is_default: bool = False
    display_name: str | None = None

    @property
    def name(self) -> str:
        return self.display_name or self.profile_name


@dataclass
class Compartment:
    """OCI Compartment information."""

    id: str  # OCID
    name: str
    description: str | None = None
    parent_id: str | None = None
    tenancy_profile: str = "DEFAULT"
    lifecycle_state: str = "ACTIVE"
    path: str = ""  # Full path like "root/parent/child"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "tenancy_profile": self.tenancy_profile,
            "lifecycle_state": self.lifecycle_state,
            "path": self.path,
        }


class TenancyManager:
    """
    Manages multiple OCI tenancies and compartment discovery.

    Supports:
    - Multiple OCI profiles from ~/.oci/config
    - Automatic compartment hierarchy discovery
    - Name-to-OCID resolution
    - Redis caching for fast lookups
    """

    _instance: "TenancyManager | None" = None

    def __init__(
        self,
        config_file: str | None = None,
        redis_url: str | None = None,
    ):
        self.config_file = config_file or os.path.expanduser("~/.oci/config")
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")

        self._tenancies: dict[str, TenancyConfig] = {}
        self._compartments: dict[str, Compartment] = {}  # OCID -> Compartment
        self._compartments_by_name: dict[str, list[Compartment]] = {}  # name -> [Compartment]
        self._initialized = False
        self._redis: Any = None
        self._logger = logger.bind(component="TenancyManager")

    @classmethod
    def get_instance(cls) -> "TenancyManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self) -> None:
        """Initialize the tenancy manager."""
        if self._initialized:
            return

        self._logger.info("Initializing TenancyManager")

        # Load OCI profiles
        self._load_oci_profiles()

        # Try to connect to Redis for caching
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            self._logger.info("Redis connected for tenancy cache")
        except Exception as e:
            self._logger.warning("Redis not available, using memory cache", error=str(e))
            self._redis = None

        # Discover compartments from all tenancies
        await self._discover_all_compartments()

        self._initialized = True
        self._logger.info(
            "TenancyManager initialized",
            tenancies=len(self._tenancies),
            compartments=len(self._compartments),
        )

    def _load_oci_profiles(self) -> None:
        """Load OCI profiles from config file."""
        try:
            import oci

            # Get list of profiles from config file
            config_path = self.config_file

            # Read config file to find all profile sections
            profiles = []
            if os.path.exists(config_path):
                with open(config_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("[") and line.endswith("]"):
                            profile = line[1:-1]
                            profiles.append(profile)

            # Load each profile
            for profile_name in profiles:
                try:
                    config = oci.config.from_file(config_path, profile_name)

                    tenancy = TenancyConfig(
                        profile_name=profile_name,
                        tenancy_ocid=config.get("tenancy", ""),
                        region=config.get("region", ""),
                        is_default=(profile_name == "DEFAULT"),
                        display_name=profile_name,
                    )

                    self._tenancies[profile_name] = tenancy
                    self._logger.debug(
                        "Loaded OCI profile",
                        profile=profile_name,
                        region=tenancy.region,
                    )

                except Exception as e:
                    self._logger.warning(
                        "Failed to load OCI profile",
                        profile=profile_name,
                        error=str(e),
                    )

            self._logger.info("OCI profiles loaded", count=len(self._tenancies))

        except Exception as e:
            self._logger.error("Failed to load OCI profiles", error=str(e))

    async def _discover_all_compartments(self) -> None:
        """Discover compartments from all tenancies."""
        from src.observability import get_tracer
        tracer = get_tracer("coordinator")

        with tracer.start_as_current_span("tenancy.discover_compartments") as span:
            total_compartments = 0

            for profile_name, tenancy in self._tenancies.items():
                try:
                    count = await self._discover_compartments_for_tenancy(tenancy)
                    total_compartments += count
                    span.set_attribute(f"compartments.{profile_name}", count)
                except Exception as e:
                    self._logger.error(
                        "Failed to discover compartments",
                        profile=profile_name,
                        error=str(e),
                    )

            span.set_attribute("compartments.total", total_compartments)
            self._logger.info(
                "Compartment discovery complete",
                total=total_compartments,
            )

    async def _discover_compartments_for_tenancy(
        self, tenancy: TenancyConfig
    ) -> int:
        """Discover compartments for a single tenancy."""
        try:
            import oci

            config = oci.config.from_file(self.config_file, tenancy.profile_name)
            identity_client = oci.identity.IdentityClient(config)

            # Get root compartment (tenancy)
            root_ocid = tenancy.tenancy_ocid

            # List all compartments recursively
            compartments_response = identity_client.list_compartments(
                compartment_id=root_ocid,
                compartment_id_in_subtree=True,
                access_level="ACCESSIBLE",
                lifecycle_state="ACTIVE",
            )

            count = 0
            compartment_map = {}  # For building paths

            # First pass: collect all compartments
            for comp in compartments_response.data:
                compartment = Compartment(
                    id=comp.id,
                    name=comp.name,
                    description=comp.description,
                    parent_id=comp.compartment_id,
                    tenancy_profile=tenancy.profile_name,
                    lifecycle_state=comp.lifecycle_state,
                )
                compartment_map[comp.id] = compartment

            # Add root compartment
            try:
                root_comp = identity_client.get_compartment(root_ocid).data
                root = Compartment(
                    id=root_ocid,
                    name=root_comp.name or "root",
                    description="Tenancy root compartment",
                    parent_id=None,
                    tenancy_profile=tenancy.profile_name,
                    lifecycle_state="ACTIVE",
                    path="root",
                )
                compartment_map[root_ocid] = root
            except Exception:
                # Root might not be accessible
                pass

            # Second pass: build paths
            for comp in compartment_map.values():
                path_parts = [comp.name]
                current = comp
                while current.parent_id and current.parent_id in compartment_map:
                    current = compartment_map[current.parent_id]
                    path_parts.insert(0, current.name)
                comp.path = "/".join(path_parts)

            # Store compartments
            for comp in compartment_map.values():
                self._compartments[comp.id] = comp

                # Index by name (multiple compartments can have same name)
                name_lower = comp.name.lower()
                if name_lower not in self._compartments_by_name:
                    self._compartments_by_name[name_lower] = []
                self._compartments_by_name[name_lower].append(comp)

                count += 1

            # Cache in Redis if available
            if self._redis:
                try:
                    import json
                    for comp in compartment_map.values():
                        key = f"compartment:{comp.id}"
                        await self._redis.setex(
                            key,
                            3600,  # 1 hour TTL
                            json.dumps(comp.to_dict()),
                        )
                        # Also index by name
                        name_key = f"compartment:name:{comp.name.lower()}"
                        await self._redis.sadd(name_key, comp.id)
                        await self._redis.expire(name_key, 3600)
                except Exception as e:
                    self._logger.warning("Failed to cache compartments", error=str(e))

            self._logger.info(
                "Compartments discovered for tenancy",
                profile=tenancy.profile_name,
                count=count,
            )

            return count

        except Exception as e:
            self._logger.error(
                "Compartment discovery failed",
                profile=tenancy.profile_name,
                error=str(e),
            )
            return 0

    async def get_compartment_ocid(
        self,
        name: str,
        tenancy_profile: str | None = None,
    ) -> str | None:
        """
        Get compartment OCID by name.

        Args:
            name: Compartment name (case-insensitive)
            tenancy_profile: Optional tenancy profile to filter

        Returns:
            Compartment OCID or None if not found
        """
        if not self._initialized:
            await self.initialize()

        name_lower = name.lower()

        # Look up in memory cache
        compartments = self._compartments_by_name.get(name_lower, [])

        if tenancy_profile:
            compartments = [c for c in compartments if c.tenancy_profile == tenancy_profile]

        if compartments:
            # Return first match (or could return all if multiple)
            return compartments[0].id

        # Try Redis cache
        if self._redis:
            try:
                name_key = f"compartment:name:{name_lower}"
                ocids = await self._redis.smembers(name_key)
                if ocids:
                    return list(ocids)[0].decode() if isinstance(list(ocids)[0], bytes) else list(ocids)[0]
            except Exception:
                pass

        return None

    async def get_compartment(self, ocid: str) -> Compartment | None:
        """Get compartment by OCID."""
        if not self._initialized:
            await self.initialize()

        return self._compartments.get(ocid)

    async def list_compartments(
        self,
        tenancy_profile: str | None = None,
    ) -> list[Compartment]:
        """
        List all compartments.

        Args:
            tenancy_profile: Optional filter by tenancy profile

        Returns:
            List of compartments
        """
        if not self._initialized:
            await self.initialize()

        compartments = list(self._compartments.values())

        if tenancy_profile:
            compartments = [c for c in compartments if c.tenancy_profile == tenancy_profile]

        return sorted(compartments, key=lambda c: c.path)

    async def search_compartments(self, query: str) -> list[Compartment]:
        """
        Search compartments by name pattern.

        Args:
            query: Search query (partial match)

        Returns:
            Matching compartments
        """
        if not self._initialized:
            await self.initialize()

        query_lower = query.lower()
        matches = []

        for comp in self._compartments.values():
            if query_lower in comp.name.lower() or query_lower in comp.path.lower():
                matches.append(comp)

        return sorted(matches, key=lambda c: c.path)

    def list_tenancies(self) -> list[TenancyConfig]:
        """List configured tenancies."""
        return list(self._tenancies.values())

    async def get_compartment_context(self, name: str) -> str:
        """
        Get context string for a compartment to include in agent prompts.

        Args:
            name: Compartment name

        Returns:
            Context string with OCID and path
        """
        ocid = await self.get_compartment_ocid(name)
        if not ocid:
            return f"Compartment '{name}' not found in any configured tenancy."

        comp = await self.get_compartment(ocid)
        if not comp:
            return f"Compartment OCID: {ocid}"

        return f"Compartment '{comp.name}' (OCID: {comp.id}, Path: {comp.path}, Tenancy: {comp.tenancy_profile})"

    async def get_all_compartments_summary(self) -> str:
        """Get a summary of all compartments for agent context."""
        if not self._initialized:
            await self.initialize()

        lines = ["## Available Compartments\n"]

        for profile_name in sorted(self._tenancies.keys()):
            tenancy = self._tenancies[profile_name]
            compartments = [c for c in self._compartments.values() if c.tenancy_profile == profile_name]

            lines.append(f"\n### Tenancy: {profile_name} ({tenancy.region})")
            lines.append(f"Root OCID: `{tenancy.tenancy_ocid[:30]}...`\n")

            for comp in sorted(compartments, key=lambda c: c.path)[:20]:
                lines.append(f"- **{comp.name}**: `{comp.id[:40]}...`")

            if len(compartments) > 20:
                lines.append(f"- ... and {len(compartments) - 20} more")

        return "\n".join(lines)
