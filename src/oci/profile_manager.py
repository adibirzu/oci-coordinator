"""
OCI Profile Manager for Human-in-the-Loop profile selection.

Manages active OCI profile selection per user/session with Redis persistence.
Provides Slack UI components for interactive profile selection.

Usage:
    from src.oci.profile_manager import ProfileManager

    manager = ProfileManager.get_instance()
    await manager.initialize()

    # Get user's active profile
    profile = await manager.get_active_profile(user_id)

    # Set user's profile
    await manager.set_active_profile(user_id, "emdemo")

    # Build Slack profile selection UI
    blocks = await manager.build_profile_selection_blocks(user_id)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ProfileInfo:
    """Information about an OCI profile."""

    name: str
    region: str
    tenancy_ocid: str
    display_name: str | None = None
    compartment_count: int = 0
    is_default: bool = False
    last_discovery: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "region": self.region,
            "tenancy_ocid": self.tenancy_ocid,
            "display_name": self.display_name or self.name,
            "compartment_count": self.compartment_count,
            "is_default": self.is_default,
            "last_discovery": self.last_discovery,
        }


class ProfileManager:
    """
    Manages OCI profile selection with human-in-the-loop support.

    Features:
    - Auto-discovery of available profiles from ~/.oci/config
    - Per-user active profile persistence in Redis
    - Slack Block Kit UI for profile selection
    - Profile-aware context for coordinator
    """

    _instance: "ProfileManager | None" = None

    def __init__(
        self,
        config_file: str | None = None,
        redis_url: str | None = None,
    ):
        self.config_file = config_file or os.path.expanduser("~/.oci/config")
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")

        self._profiles: dict[str, ProfileInfo] = {}
        self._initialized = False
        self._redis: Any = None
        self._logger = logger.bind(component="ProfileManager")

    @classmethod
    def get_instance(cls) -> "ProfileManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    async def initialize(self) -> None:
        """Initialize the profile manager."""
        if self._initialized:
            return

        self._logger.info("Initializing ProfileManager")

        # Load profiles from ~/.oci/config
        self._load_profiles()

        # Connect to Redis for persistence
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            self._logger.info("Redis connected for profile persistence")
        except Exception as e:
            self._logger.warning("Redis not available, using memory only", error=str(e))
            self._redis = None

        # Load cached discovery data if available (fast startup)
        self.load_cached_discovery()

        self._initialized = True
        self._logger.info(
            "ProfileManager initialized",
            profiles=list(self._profiles.keys()),
        )

    def _load_profiles(self) -> None:
        """Load OCI profiles from config file."""
        try:
            import oci

            if not os.path.exists(self.config_file):
                self._logger.warning("OCI config file not found", path=self.config_file)
                return

            # Parse config file for profile sections
            profiles: list[str] = []
            with open(self.config_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        profile = line[1:-1]
                        profiles.append(profile)

            # Load each profile
            for profile_name in profiles:
                try:
                    config = oci.config.from_file(self.config_file, profile_name)

                    info = ProfileInfo(
                        name=profile_name,
                        region=config.get("region", "unknown"),
                        tenancy_ocid=config.get("tenancy", ""),
                        display_name=profile_name.upper() if profile_name != "DEFAULT" else "Default",
                        is_default=(profile_name == "DEFAULT"),
                    )

                    self._profiles[profile_name] = info
                    self._logger.debug(
                        "Loaded OCI profile",
                        profile=profile_name,
                        region=info.region,
                    )

                except Exception as e:
                    self._logger.warning(
                        "Failed to load OCI profile",
                        profile=profile_name,
                        error=str(e),
                    )

            self._logger.info("OCI profiles loaded", count=len(self._profiles))

        except Exception as e:
            self._logger.error("Failed to load OCI profiles", error=str(e))

    def list_profiles(self) -> list[ProfileInfo]:
        """List all available OCI profiles."""
        return list(self._profiles.values())

    def get_profile(self, name: str) -> ProfileInfo | None:
        """Get profile by name."""
        return self._profiles.get(name)

    def has_multiple_profiles(self) -> bool:
        """Check if multiple profiles are configured."""
        return len(self._profiles) > 1

    async def get_active_profile(self, user_id: str) -> str:
        """
        Get user's active profile.

        Args:
            user_id: Slack user ID or session identifier

        Returns:
            Active profile name (defaults to DEFAULT or first available)
        """
        if not self._initialized:
            await self.initialize()

        # Try Redis cache first
        if self._redis:
            try:
                key = f"oci:profile:{user_id}"
                profile = await self._redis.get(key)
                if profile:
                    profile_str = profile.decode() if isinstance(profile, bytes) else profile
                    if profile_str in self._profiles:
                        return profile_str
            except Exception as e:
                self._logger.warning("Redis get failed", error=str(e))

        # Default to DEFAULT profile or first available
        if "DEFAULT" in self._profiles:
            return "DEFAULT"

        if self._profiles:
            return list(self._profiles.keys())[0]

        return "DEFAULT"

    async def set_active_profile(self, user_id: str, profile: str) -> bool:
        """
        Set user's active profile.

        Args:
            user_id: Slack user ID or session identifier
            profile: Profile name to set

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        if profile not in self._profiles:
            self._logger.warning("Invalid profile", profile=profile)
            return False

        # Store in Redis
        if self._redis:
            try:
                key = f"oci:profile:{user_id}"
                await self._redis.setex(
                    key,
                    86400 * 7,  # 7 day TTL
                    profile,
                )
                self._logger.info(
                    "Active profile set",
                    user_id=user_id,
                    profile=profile,
                )
                return True
            except Exception as e:
                self._logger.warning("Redis set failed", error=str(e))

        return False

    async def clear_active_profile(self, user_id: str) -> None:
        """Clear user's active profile preference."""
        if self._redis:
            try:
                key = f"oci:profile:{user_id}"
                await self._redis.delete(key)
            except Exception:
                pass

    async def get_profile_context(self, user_id: str) -> dict[str, Any]:
        """
        Get profile context for coordinator.

        Args:
            user_id: User identifier

        Returns:
            Dict with profile information for coordinator context
        """
        if not self._initialized:
            await self.initialize()

        profile_name = await self.get_active_profile(user_id)
        profile = self._profiles.get(profile_name)

        if not profile:
            return {
                "profile": "DEFAULT",
                "region": "unknown",
                "needs_selection": self.has_multiple_profiles(),
            }

        return {
            "profile": profile.name,
            "region": profile.region,
            "tenancy_ocid": profile.tenancy_ocid,
            "display_name": profile.display_name,
            "needs_selection": False,
        }

    def build_profile_selection_blocks(
        self,
        current_profile: str | None = None,
        include_header: bool = True,
    ) -> list[dict]:
        """
        Build Slack Block Kit blocks for profile selection.

        Args:
            current_profile: Currently active profile (if any)
            include_header: Whether to include a header section

        Returns:
            List of Slack block dictionaries
        """
        blocks = []

        if include_header:
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": ":gear: Select OCI Profile",
                    "emoji": True,
                }
            })
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Multiple OCI profiles are available. Please select which tenancy you want to work with:",
                }
            })

        # Build profile buttons
        profile_buttons = []
        for profile in self._profiles.values():
            # Show indicator for current profile
            label = profile.display_name or profile.name
            if profile.name == current_profile:
                label = f"{label} (Active)"
                style = "primary"
            else:
                style = None

            # Region hint
            region_short = profile.region.replace("-", " ").title() if profile.region else ""

            button = {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": f"{label}",
                    "emoji": True,
                },
                "action_id": f"select_profile_{profile.name}",
                "value": profile.name,
            }

            if style:
                button["style"] = style

            profile_buttons.append(button)

        if profile_buttons:
            blocks.append({
                "type": "actions",
                "elements": profile_buttons[:5],  # Slack max 5 buttons per action
            })

        # Show profile details
        for profile in self._profiles.values():
            emoji = ":white_check_mark:" if profile.name == current_profile else ":cloud:"
            blocks.append({
                "type": "context",
                "elements": [{
                    "type": "mrkdwn",
                    "text": f"{emoji} *{profile.display_name or profile.name}*: `{profile.region}` | Tenancy: `{profile.tenancy_ocid[:30]}...`",
                }]
            })

        return blocks

    def build_profile_indicator_blocks(
        self,
        current_profile: str,
    ) -> list[dict]:
        """
        Build compact profile indicator for messages.

        Args:
            current_profile: Active profile name

        Returns:
            Slack blocks showing current profile
        """
        profile = self._profiles.get(current_profile)
        if not profile:
            return []

        # Compact indicator with change button
        return [{
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": f":cloud: *Profile:* {profile.display_name or profile.name} ({profile.region})",
            }]
        }, {
            "type": "actions",
            "elements": [{
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "Change Profile",
                    "emoji": True,
                },
                "action_id": "show_profile_selector",
                "value": "show",
            }]
        }]

    async def update_compartment_counts(self) -> None:
        """Update compartment counts from TenancyManager."""
        try:
            from src.oci.tenancy_manager import TenancyManager
            manager = TenancyManager.get_instance()

            if not manager._initialized:
                await manager.initialize()

            for profile_name, profile in self._profiles.items():
                compartments = await manager.list_compartments(tenancy_profile=profile_name)
                profile.compartment_count = len(compartments)
                profile.last_discovery = datetime.now().isoformat()

            self._logger.info("Compartment counts updated")

        except Exception as e:
            self._logger.warning("Failed to update compartment counts", error=str(e))

    def load_cached_discovery(self) -> bool:
        """
        Load cached discovery data from JSON files.

        This provides fast startup by loading pre-computed compartment counts
        from the discovery script cache files.

        Returns:
            True if any cache was loaded
        """
        import json
        from pathlib import Path

        cache_dir = Path.home() / ".oci_coordinator_cache"
        loaded = False

        for profile_name, profile in self._profiles.items():
            cache_file = cache_dir / f"discovery_{profile_name}.json"

            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        data = json.load(f)

                    profile.compartment_count = len(data.get("compartments", []))
                    profile.last_discovery = data.get("discovered_at")

                    self._logger.debug(
                        "Loaded cached discovery",
                        profile=profile_name,
                        compartments=profile.compartment_count,
                    )
                    loaded = True

                except Exception as e:
                    self._logger.warning(
                        "Failed to load cache",
                        profile=profile_name,
                        error=str(e),
                    )

        if loaded:
            self._logger.info("Cached discovery data loaded")

        return loaded

    async def get_cached_compartments(self, profile_name: str) -> list[dict[str, Any]]:
        """
        Get cached compartments for a profile.

        Args:
            profile_name: OCI profile name

        Returns:
            List of compartment dictionaries from cache
        """
        import json
        from pathlib import Path

        cache_file = Path.home() / ".oci_coordinator_cache" / f"discovery_{profile_name}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                return data.get("compartments", [])
            except Exception as e:
                self._logger.warning("Failed to read cache", error=str(e))

        return []

    async def get_cached_databases(self, profile_name: str) -> list[dict[str, Any]]:
        """
        Get cached databases for a profile.

        Args:
            profile_name: OCI profile name

        Returns:
            List of database dictionaries from cache
        """
        import json
        from pathlib import Path

        cache_file = Path.home() / ".oci_coordinator_cache" / f"discovery_{profile_name}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                return data.get("databases", []) + data.get("opsi_databases", [])
            except Exception as e:
                self._logger.warning("Failed to read cache", error=str(e))

        return []

    async def search_cached_resources(
        self,
        query: str,
        profile_name: str | None = None,
        resource_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search cached resources by name.

        Args:
            query: Search query (partial match)
            profile_name: Optional profile filter
            resource_type: Optional type filter (compartment, database)

        Returns:
            Matching resources from cache
        """
        results = []
        query_lower = query.lower()

        profiles = [profile_name] if profile_name else list(self._profiles.keys())

        for profile in profiles:
            if resource_type in (None, "compartment"):
                compartments = await self.get_cached_compartments(profile)
                for comp in compartments:
                    if query_lower in comp.get("name", "").lower():
                        comp["_profile"] = profile
                        comp["_type"] = "compartment"
                        results.append(comp)

            if resource_type in (None, "database"):
                databases = await self.get_cached_databases(profile)
                for db in databases:
                    name = db.get("display_name") or db.get("database_name") or ""
                    if query_lower in name.lower():
                        db["_profile"] = profile
                        db["_type"] = "database"
                        results.append(db)

        return results


async def get_profile_manager() -> ProfileManager:
    """Get initialized ProfileManager instance."""
    manager = ProfileManager.get_instance()
    await manager.initialize()
    return manager
