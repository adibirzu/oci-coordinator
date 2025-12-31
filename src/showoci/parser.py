"""
ShowOCI Parser - Utilities for parsing ShowOCI output formats.

Handles parsing of ShowOCI JSON output and converting to
standardized resource formats for caching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ShowOCIParser:
    """
    Parser for ShowOCI JSON output files.

    Handles the nested structure of ShowOCI output and extracts
    resources in a flat, cacheable format.

    Example:
        parser = ShowOCIParser()
        data = parser.parse_file("/tmp/showoci_output.json")

        instances = parser.extract_instances(data)
        vcns = parser.extract_vcns(data)
    """

    def __init__(self):
        self._logger = logger.bind(component="ShowOCIParser")

    def parse_file(self, file_path: str | Path) -> dict[str, Any]:
        """
        Parse ShowOCI JSON output file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed data dictionary
        """
        path = Path(file_path)
        if not path.exists():
            self._logger.error("File not found", path=str(path))
            return {}

        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._logger.info("Parsed ShowOCI file", path=str(path))
            return data
        except json.JSONDecodeError as e:
            self._logger.error("Invalid JSON", path=str(path), error=str(e))
            return {}

    def parse_json(self, json_str: str) -> dict[str, Any]:
        """Parse ShowOCI JSON string."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self._logger.error("Invalid JSON string", error=str(e))
            return {}

    def extract_compartments(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract compartments from ShowOCI data.

        ShowOCI structure: data["identity"]["compartments"]
        """
        identity = data.get("identity", {})
        raw_compartments = identity.get("compartments", [])

        compartments = []
        for comp in raw_compartments:
            compartments.append({
                "id": comp.get("id"),
                "name": comp.get("name"),
                "description": comp.get("description"),
                "lifecycle_state": comp.get("lifecycle_state", "ACTIVE"),
                "parent_id": comp.get("compartment_id"),
                "path": comp.get("path"),
                "time_created": comp.get("time_created"),
            })

        return compartments

    def extract_instances(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract compute instances from ShowOCI data.

        ShowOCI structure: data["data"][region]["compartments"][comp_path]["compute"]["instances"]
        """
        instances = []

        for region, region_data in data.get("data", {}).items():
            for comp_path, comp_data in region_data.get("compartments", {}).items():
                compute = comp_data.get("compute", {})
                for inst in compute.get("instances", []):
                    instances.append({
                        "id": inst.get("id"),
                        "display_name": inst.get("display_name") or inst.get("name"),
                        "lifecycle_state": inst.get("lifecycle_state") or inst.get("status"),
                        "shape": inst.get("shape"),
                        "availability_domain": inst.get("availability_domain"),
                        "fault_domain": inst.get("fault_domain"),
                        "compartment_id": inst.get("compartment_id"),
                        "compartment_path": comp_path,
                        "region": region,
                        "time_created": inst.get("time_created"),
                        # Extended info if available
                        "image_id": inst.get("image_id"),
                        "public_ips": inst.get("public_ips", []),
                        "private_ips": inst.get("private_ips", []),
                        "ocpu_count": inst.get("ocpus") or inst.get("shape_config_ocpus"),
                        "memory_gb": inst.get("memory_in_gbs") or inst.get("shape_config_memory_in_gbs"),
                    })

        return instances

    def extract_vcns(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract VCNs from ShowOCI data.

        ShowOCI structure: data["data"][region]["compartments"][comp_path]["network"]["vcn"]
        """
        vcns = []

        for region, region_data in data.get("data", {}).items():
            for comp_path, comp_data in region_data.get("compartments", {}).items():
                network = comp_data.get("network", {})
                for vcn in network.get("vcn", []):
                    vcns.append({
                        "id": vcn.get("id"),
                        "display_name": vcn.get("display_name") or vcn.get("name"),
                        "cidr_block": vcn.get("cidr_block"),
                        "cidr_blocks": vcn.get("cidr_blocks", []),
                        "lifecycle_state": vcn.get("lifecycle_state"),
                        "compartment_id": vcn.get("compartment_id"),
                        "compartment_path": comp_path,
                        "region": region,
                        "dns_label": vcn.get("dns_label"),
                        "time_created": vcn.get("time_created"),
                        # Subnet count if available
                        "subnet_count": len(vcn.get("subnets", [])),
                    })

        return vcns

    def extract_subnets(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract subnets from ShowOCI data."""
        subnets = []

        for region, region_data in data.get("data", {}).items():
            for comp_path, comp_data in region_data.get("compartments", {}).items():
                network = comp_data.get("network", {})
                for subnet in network.get("subnet", []):
                    subnets.append({
                        "id": subnet.get("id"),
                        "display_name": subnet.get("display_name") or subnet.get("name"),
                        "cidr_block": subnet.get("cidr_block"),
                        "vcn_id": subnet.get("vcn_id"),
                        "availability_domain": subnet.get("availability_domain"),
                        "lifecycle_state": subnet.get("lifecycle_state"),
                        "compartment_id": subnet.get("compartment_id"),
                        "compartment_path": comp_path,
                        "region": region,
                        "prohibit_public_ip": subnet.get("prohibit_public_ip_on_vnic"),
                    })

        return subnets

    def extract_autonomous_databases(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract autonomous databases from ShowOCI data."""
        databases = []

        for region, region_data in data.get("data", {}).items():
            for comp_path, comp_data in region_data.get("compartments", {}).items():
                db_data = comp_data.get("database", {})
                for db in db_data.get("autonomous", []):
                    databases.append({
                        "id": db.get("id"),
                        "display_name": db.get("display_name") or db.get("name"),
                        "db_name": db.get("db_name"),
                        "lifecycle_state": db.get("lifecycle_state") or db.get("status"),
                        "db_workload": db.get("db_workload"),
                        "cpu_core_count": db.get("cpu_core_count"),
                        "data_storage_size_in_tbs": db.get("data_storage_size_in_tbs"),
                        "compartment_id": db.get("compartment_id"),
                        "compartment_path": comp_path,
                        "region": region,
                        "is_free_tier": db.get("is_free_tier", False),
                        "time_created": db.get("time_created"),
                    })

        return databases

    def extract_db_systems(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract DB systems from ShowOCI data."""
        db_systems = []

        for region, region_data in data.get("data", {}).items():
            for comp_path, comp_data in region_data.get("compartments", {}).items():
                db_data = comp_data.get("database", {})
                for dbs in db_data.get("db_system", []):
                    db_systems.append({
                        "id": dbs.get("id"),
                        "display_name": dbs.get("display_name") or dbs.get("name"),
                        "lifecycle_state": dbs.get("lifecycle_state") or dbs.get("status"),
                        "shape": dbs.get("shape"),
                        "database_edition": dbs.get("database_edition"),
                        "availability_domain": dbs.get("availability_domain"),
                        "compartment_id": dbs.get("compartment_id"),
                        "compartment_path": comp_path,
                        "region": region,
                        "node_count": dbs.get("node_count"),
                        "time_created": dbs.get("time_created"),
                    })

        return db_systems

    def extract_block_volumes(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract block volumes from ShowOCI data."""
        volumes = []

        for region, region_data in data.get("data", {}).items():
            for comp_path, comp_data in region_data.get("compartments", {}).items():
                storage = comp_data.get("block_storage", {})
                for vol in storage.get("volumes", []):
                    volumes.append({
                        "id": vol.get("id"),
                        "display_name": vol.get("display_name") or vol.get("name"),
                        "size_in_gbs": vol.get("size_in_gbs"),
                        "lifecycle_state": vol.get("lifecycle_state"),
                        "availability_domain": vol.get("availability_domain"),
                        "compartment_id": vol.get("compartment_id"),
                        "compartment_path": comp_path,
                        "region": region,
                        "vpus_per_gb": vol.get("vpus_per_gb"),
                        "is_hydrated": vol.get("is_hydrated"),
                        "time_created": vol.get("time_created"),
                    })

        return volumes

    def extract_buckets(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract object storage buckets from ShowOCI data."""
        buckets = []

        for region, region_data in data.get("data", {}).items():
            for comp_path, comp_data in region_data.get("compartments", {}).items():
                storage = comp_data.get("object_storage", {})
                for bucket in storage.get("buckets", []):
                    buckets.append({
                        "id": bucket.get("id"),
                        "name": bucket.get("name"),
                        "namespace": bucket.get("namespace"),
                        "compartment_id": bucket.get("compartment_id"),
                        "compartment_path": comp_path,
                        "region": region,
                        "storage_tier": bucket.get("storage_tier"),
                        "public_access_type": bucket.get("public_access_type"),
                        "approximate_count": bucket.get("approximate_count"),
                        "approximate_size": bucket.get("approximate_size"),
                        "time_created": bucket.get("time_created"),
                    })

        return buckets

    def get_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Get a summary of all resources in ShowOCI data.

        Returns:
            Summary with counts by resource type and region
        """
        instances = self.extract_instances(data)
        vcns = self.extract_vcns(data)
        subnets = self.extract_subnets(data)
        adbs = self.extract_autonomous_databases(data)
        db_systems = self.extract_db_systems(data)
        volumes = self.extract_block_volumes(data)
        buckets = self.extract_buckets(data)
        compartments = self.extract_compartments(data)

        # Count by region
        regions = set()
        for inst in instances:
            if inst.get("region"):
                regions.add(inst["region"])

        return {
            "total_compartments": len(compartments),
            "total_instances": len(instances),
            "total_vcns": len(vcns),
            "total_subnets": len(subnets),
            "total_autonomous_databases": len(adbs),
            "total_db_systems": len(db_systems),
            "total_block_volumes": len(volumes),
            "total_buckets": len(buckets),
            "regions": list(regions),
            "instances_by_state": self._count_by_field(instances, "lifecycle_state"),
            "databases_by_state": self._count_by_field(adbs, "lifecycle_state"),
        }

    def _count_by_field(
        self, items: list[dict[str, Any]], field: str
    ) -> dict[str, int]:
        """Count items by a field value."""
        counts: dict[str, int] = {}
        for item in items:
            value = item.get(field, "UNKNOWN")
            counts[value] = counts.get(value, 0) + 1
        return counts
