"""
OCI Compute MCP Tools.

Provides tools for compute instance management including listing,
starting, stopping, and getting instance details.
"""

import json
from typing import Any

from opentelemetry import trace

from src.mcp.server.auth import get_compute_client

# Get tracer for compute tools
_tracer = trace.get_tracer("mcp-oci-compute")


async def _list_instances_logic(
    compartment_id: str,
    limit: int = 50,
    lifecycle_state: str | None = None,
    format: str = "json",
) -> str:
    """Internal logic for listing instances."""
    with _tracer.start_as_current_span("mcp.compute.list_instances") as span:
        span.set_attribute("compartment_id", compartment_id)
        span.set_attribute("limit", limit)
        if lifecycle_state:
            span.set_attribute("lifecycle_state", lifecycle_state)

        client = get_compute_client()

        try:
            kwargs = {"compartment_id": compartment_id, "limit": limit}
            if lifecycle_state:
                kwargs["lifecycle_state"] = lifecycle_state

            response = client.list_instances(**kwargs)
            instances = response.data
            span.set_attribute("instance_count", len(instances))

            if format == "json":
                return json.dumps(
                    [
                        {
                            "name": i.display_name,
                            "id": i.id,
                            "state": i.lifecycle_state,
                            "shape": i.shape,
                            "availability_domain": i.availability_domain,
                            "fault_domain": i.fault_domain,
                            "time_created": str(i.time_created) if i.time_created else None,
                        }
                        for i in instances
                    ],
                    indent=2,
                )

            # Markdown table for display
            if not instances:
                return "No compute instances found in this compartment."

            lines = [
                f"Found {len(instances)} compute instances:\n",
                "| Name | State | Shape | Availability Domain |",
                "| --- | --- | --- | --- |",
            ]
            for i in instances:
                ad_short = i.availability_domain.split(":")[-1] if i.availability_domain else "N/A"
                lines.append(f"| {i.display_name} | {i.lifecycle_state} | {i.shape} | {ad_short} |")

            return "\n".join(lines)

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return f"Error listing instances: {e}"


async def _get_instance_logic(instance_id: str, format: str = "json") -> str:
    """Get details of a specific instance."""
    client = get_compute_client()

    try:
        response = client.get_instance(instance_id=instance_id)
        i = response.data

        if format == "json":
            return json.dumps(
                {
                    "name": i.display_name,
                    "id": i.id,
                    "state": i.lifecycle_state,
                    "shape": i.shape,
                    "availability_domain": i.availability_domain,
                    "fault_domain": i.fault_domain,
                    "compartment_id": i.compartment_id,
                    "time_created": str(i.time_created) if i.time_created else None,
                    "image_id": i.image_id,
                    "region": i.region,
                },
                indent=2,
            )

        return f"""## Instance: {i.display_name}

- **OCID**: `{i.id}`
- **State**: {i.lifecycle_state}
- **Shape**: {i.shape}
- **Availability Domain**: {i.availability_domain}
- **Fault Domain**: {i.fault_domain}
- **Created**: {i.time_created}
"""

    except Exception as e:
        return f"Error getting instance: {e}"


async def _instance_action_logic(instance_id: str, action: str) -> str:
    """Perform action on an instance (start, stop, softreset, reset)."""
    client = get_compute_client()

    try:
        response = client.instance_action(instance_id=instance_id, action=action)
        i = response.data
        return json.dumps(
            {
                "success": True,
                "action": action,
                "instance_name": i.display_name,
                "instance_id": i.id,
                "new_state": i.lifecycle_state,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "action": action, "error": str(e)}, indent=2)


async def _find_instance_by_name(
    instance_name: str,
    compartment_id: str,
    exact_match: bool = False,
) -> list[dict[str, Any]]:
    """
    Find instances by name.

    Args:
        instance_name: Name or partial name to search for
        compartment_id: Compartment to search in
        exact_match: If True, requires exact name match

    Returns:
        List of matching instances
    """
    client = get_compute_client()

    try:
        # List all instances (not terminated)
        response = client.list_instances(
            compartment_id=compartment_id,
            lifecycle_state="RUNNING",
        )
        running = response.data or []

        response = client.list_instances(
            compartment_id=compartment_id,
            lifecycle_state="STOPPED",
        )
        stopped = response.data or []

        all_instances = running + stopped

        # Filter by name
        matches = []
        search_name = instance_name.lower()

        for i in all_instances:
            display_name = (i.display_name or "").lower()
            if exact_match:
                if display_name == search_name:
                    matches.append({
                        "name": i.display_name,
                        "id": i.id,
                        "state": i.lifecycle_state,
                        "shape": i.shape,
                        "compartment_id": i.compartment_id,
                    })
            elif search_name in display_name:
                matches.append({
                    "name": i.display_name,
                    "id": i.id,
                    "state": i.lifecycle_state,
                    "shape": i.shape,
                    "compartment_id": i.compartment_id,
                })

        return matches

    except Exception as e:
        return [{"error": str(e)}]


async def _instance_action_by_name(
    instance_name: str,
    compartment_id: str,
    action: str,
) -> str:
    """
    Perform action on instance(s) by name.

    Args:
        instance_name: Name of the instance
        compartment_id: Compartment containing the instance
        action: Action to perform (START, SOFTSTOP, SOFTRESET)

    Returns:
        JSON result of the action
    """
    matches = await _find_instance_by_name(instance_name, compartment_id, exact_match=True)

    if not matches:
        # Try partial match
        matches = await _find_instance_by_name(instance_name, compartment_id, exact_match=False)

    if not matches:
        return json.dumps({
            "success": False,
            "error": f"No instance found matching '{instance_name}' in compartment",
        }, indent=2)

    if len(matches) > 1:
        return json.dumps({
            "success": False,
            "error": f"Multiple instances match '{instance_name}'. Please be more specific.",
            "matches": [{"name": m["name"], "id": m["id"], "state": m["state"]} for m in matches],
        }, indent=2)

    # Single match - perform action
    instance = matches[0]
    if "error" in instance:
        return json.dumps({"success": False, "error": instance["error"]}, indent=2)

    return await _instance_action_logic(instance["id"], action)


async def _list_shapes_logic(
    compartment_id: str,
    availability_domain: str | None = None,
) -> str:
    """List available compute shapes."""
    client = get_compute_client()

    try:
        kwargs = {"compartment_id": compartment_id}
        if availability_domain:
            kwargs["availability_domain"] = availability_domain

        response = client.list_shapes(**kwargs)
        shapes = response.data

        # Filter to common shapes and format nicely
        result = []
        for s in shapes[:30]:  # Limit to avoid too much output
            shape_info = {
                "name": s.shape,
                "ocpus": s.ocpus if hasattr(s, "ocpus") else None,
                "memory_gb": s.memory_in_gbs if hasattr(s, "memory_in_gbs") else None,
                "is_flexible": s.is_flexible if hasattr(s, "is_flexible") else False,
            }
            if shape_info["is_flexible"]:
                shape_info["max_ocpus"] = s.ocpu_options.max if hasattr(s, "ocpu_options") else None
                shape_info["max_memory_gb"] = s.memory_options.max_in_gbs if hasattr(s, "memory_options") else None
            result.append(shape_info)

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


async def _list_images_logic(
    compartment_id: str,
    operating_system: str = "Oracle Linux",
    limit: int = 10,
) -> str:
    """List available compute images."""
    client = get_compute_client()

    try:
        response = client.list_images(
            compartment_id=compartment_id,
            operating_system=operating_system,
            limit=limit,
            sort_by="TIMECREATED",
            sort_order="DESC",
        )
        images = response.data

        result = []
        for img in images:
            result.append({
                "id": img.id,
                "name": img.display_name,
                "operating_system": img.operating_system,
                "operating_system_version": img.operating_system_version,
                "size_gb": round(img.size_in_mbs / 1024, 1) if img.size_in_mbs else None,
                "time_created": str(img.time_created) if img.time_created else None,
            })

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


async def _launch_instance_logic(
    display_name: str,
    compartment_id: str,
    availability_domain: str,
    subnet_id: str,
    shape: str = "VM.Standard.E4.Flex",
    ocpus: int = 1,
    memory_gb: int = 8,
    image_id: str | None = None,
    ssh_public_key: str | None = None,
    boot_volume_size_gb: int = 50,
    assign_public_ip: bool = True,
) -> str:
    """Launch a new compute instance."""
    import os

    import oci

    # Check if mutations are allowed
    allow_mutations = os.getenv("ALLOW_MUTATIONS", "").lower() in {"1", "true", "yes"}
    if not allow_mutations:
        return json.dumps({
            "error": "Write operations are disabled",
            "message": "Set ALLOW_MUTATIONS=true in environment to enable instance creation",
            "suggestion": "This is a safety feature. Enable mutations only in controlled environments.",
        })

    client = get_compute_client()

    try:
        # If no image_id provided, find the latest Oracle Linux 8 image
        if not image_id:
            images_response = client.list_images(
                compartment_id=compartment_id,
                operating_system="Oracle Linux",
                operating_system_version="8",
                sort_by="TIMECREATED",
                sort_order="DESC",
                limit=1,
            )
            if images_response.data:
                image_id = images_response.data[0].id
            else:
                return json.dumps({"error": "No Oracle Linux 8 image found. Please provide an image_id."})

        # Build shape config for flex shapes
        shape_config = None
        if "Flex" in shape:
            shape_config = oci.core.models.LaunchInstanceShapeConfigDetails(
                ocpus=float(ocpus),
                memory_in_gbs=float(memory_gb),
            )

        # Build source details
        source_details = oci.core.models.InstanceSourceViaImageDetails(
            image_id=image_id,
            boot_volume_size_in_gbs=boot_volume_size_gb,
        )

        # Build VNIC details
        vnic_details = oci.core.models.CreateVnicDetails(
            subnet_id=subnet_id,
            assign_public_ip=assign_public_ip,
            display_name=f"{display_name}-vnic",
        )

        # Build metadata (SSH key if provided)
        metadata = {}
        if ssh_public_key:
            metadata["ssh_authorized_keys"] = ssh_public_key

        # Build launch details
        launch_details = oci.core.models.LaunchInstanceDetails(
            compartment_id=compartment_id,
            availability_domain=availability_domain,
            display_name=display_name,
            shape=shape,
            shape_config=shape_config,
            source_details=source_details,
            create_vnic_details=vnic_details,
            metadata=metadata if metadata else None,
        )

        # Launch the instance
        response = client.launch_instance(launch_details)
        instance = response.data

        return json.dumps({
            "success": True,
            "message": f"Instance '{display_name}' is being launched",
            "instance": {
                "id": instance.id,
                "name": instance.display_name,
                "state": instance.lifecycle_state,
                "shape": instance.shape,
                "availability_domain": instance.availability_domain,
                "image_id": image_id,
            },
            "note": "Instance provisioning may take 2-5 minutes. Use oci_compute_get_instance to check status.",
        }, indent=2)

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OCI API error: {e.message}",
            "code": e.code,
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def register_compute_tools(mcp: Any) -> None:
    """Register compute tools with the MCP server."""

    @mcp.tool()
    async def oci_compute_list_instances(
        compartment_id: str,
        limit: int = 50,
        lifecycle_state: str | None = None,
        format: str = "json",
    ) -> str:
        """List OCI compute instances in a compartment.

        Args:
            compartment_id: OCID of the compartment to list instances from
            limit: Maximum number of instances to return (default 50)
            lifecycle_state: Filter by state (RUNNING, STOPPED, TERMINATED, etc.)
            format: Output format ('json' or 'markdown')

        Returns:
            List of compute instances with name, state, shape, and availability domain
        """
        return await _list_instances_logic(compartment_id, limit, lifecycle_state, format)

    @mcp.tool()
    async def oci_compute_get_instance(instance_id: str, format: str = "json") -> str:
        """Get details of a specific compute instance.

        Args:
            instance_id: OCID of the instance
            format: Output format ('json' or 'markdown')

        Returns:
            Instance details including shape, state, and configuration
        """
        return await _get_instance_logic(instance_id, format)

    @mcp.tool()
    async def oci_compute_start_instance(instance_id: str) -> str:
        """Start a stopped compute instance.

        Args:
            instance_id: OCID of the instance to start

        Returns:
            Action result with new state
        """
        return await _instance_action_logic(instance_id, "START")

    @mcp.tool()
    async def oci_compute_stop_instance(instance_id: str) -> str:
        """Stop a running compute instance (graceful shutdown).

        Args:
            instance_id: OCID of the instance to stop

        Returns:
            Action result with new state
        """
        return await _instance_action_logic(instance_id, "SOFTSTOP")

    @mcp.tool()
    async def oci_compute_restart_instance(instance_id: str) -> str:
        """Restart a compute instance (graceful reboot).

        Args:
            instance_id: OCID of the instance to restart

        Returns:
            Action result with new state
        """
        return await _instance_action_logic(instance_id, "SOFTRESET")

    @mcp.tool()
    async def oci_compute_find_instance(
        instance_name: str,
        compartment_id: str,
        exact_match: bool = False,
    ) -> str:
        """Find compute instances by name.

        Use this to look up an instance OCID when you only know the name.

        Args:
            instance_name: Name or partial name to search for
            compartment_id: OCID of the compartment to search in
            exact_match: If True, requires exact name match (default: False)

        Returns:
            List of matching instances with their OCIDs and states
        """
        matches = await _find_instance_by_name(instance_name, compartment_id, exact_match)
        return json.dumps(matches, indent=2)

    @mcp.tool()
    async def oci_compute_start_by_name(
        instance_name: str,
        compartment_id: str,
    ) -> str:
        """Start a compute instance by its display name.

        Finds the instance by name and starts it. If multiple instances
        match the name, returns an error listing the matches.

        Args:
            instance_name: Display name of the instance to start
            compartment_id: OCID of the compartment containing the instance

        Returns:
            Action result with new state, or error if name is ambiguous
        """
        return await _instance_action_by_name(instance_name, compartment_id, "START")

    @mcp.tool()
    async def oci_compute_stop_by_name(
        instance_name: str,
        compartment_id: str,
    ) -> str:
        """Stop a compute instance by its display name (graceful shutdown).

        Finds the instance by name and stops it. If multiple instances
        match the name, returns an error listing the matches.

        Args:
            instance_name: Display name of the instance to stop
            compartment_id: OCID of the compartment containing the instance

        Returns:
            Action result with new state, or error if name is ambiguous
        """
        return await _instance_action_by_name(instance_name, compartment_id, "SOFTSTOP")

    @mcp.tool()
    async def oci_compute_restart_by_name(
        instance_name: str,
        compartment_id: str,
    ) -> str:
        """Restart a compute instance by its display name (graceful reboot).

        Finds the instance by name and restarts it. If multiple instances
        match the name, returns an error listing the matches.

        Args:
            instance_name: Display name of the instance to restart
            compartment_id: OCID of the compartment containing the instance

        Returns:
            Action result with new state, or error if name is ambiguous
        """
        return await _instance_action_by_name(instance_name, compartment_id, "SOFTRESET")

    @mcp.tool()
    async def oci_compute_list_shapes(
        compartment_id: str,
        availability_domain: str | None = None,
    ) -> str:
        """List available compute shapes in a compartment.

        Returns shapes with their OCPU, memory, and type information.
        Useful for selecting a shape before launching an instance.

        Args:
            compartment_id: OCID of the compartment
            availability_domain: Optional AD to filter shapes by availability

        Returns:
            List of available shapes with their specifications
        """
        return await _list_shapes_logic(compartment_id, availability_domain)

    @mcp.tool()
    async def oci_compute_list_images(
        compartment_id: str,
        operating_system: str = "Oracle Linux",
        limit: int = 10,
    ) -> str:
        """List available compute images for launching instances.

        Returns images with their OS, version, and type information.
        Useful for selecting an image before launching an instance.

        Args:
            compartment_id: OCID of the compartment
            operating_system: Filter by OS (e.g., "Oracle Linux", "CentOS", "Windows")
            limit: Maximum number of images to return

        Returns:
            List of available images with their specifications
        """
        return await _list_images_logic(compartment_id, operating_system, limit)

    @mcp.tool()
    async def oci_compute_launch_instance(
        display_name: str,
        compartment_id: str,
        availability_domain: str,
        subnet_id: str,
        shape: str = "VM.Standard.E4.Flex",
        ocpus: int = 1,
        memory_gb: int = 8,
        image_id: str | None = None,
        ssh_public_key: str | None = None,
        boot_volume_size_gb: int = 50,
        assign_public_ip: bool = True,
    ) -> str:
        """Launch a new compute instance (requires ALLOW_MUTATIONS=true).

        Creates a new VM instance with the specified configuration.
        Uses Oracle Linux 8 by default if no image_id is provided.

        Args:
            display_name: Name for the new instance
            compartment_id: OCID of the compartment to create instance in
            availability_domain: Full AD name (e.g., "Uocm:EU-FRANKFURT-1-AD-1")
            subnet_id: OCID of the subnet for the instance's VNIC
            shape: Compute shape (default: VM.Standard.E4.Flex)
            ocpus: Number of OCPUs for flex shapes (default: 1)
            memory_gb: Memory in GB for flex shapes (default: 8)
            image_id: OCID of the image to use (defaults to latest Oracle Linux 8)
            ssh_public_key: SSH public key for instance access
            boot_volume_size_gb: Boot volume size in GB (default: 50)
            assign_public_ip: Whether to assign a public IP (default: True)

        Returns:
            Details of the launched instance or error message
        """
        return await _launch_instance_logic(
            display_name=display_name,
            compartment_id=compartment_id,
            availability_domain=availability_domain,
            subnet_id=subnet_id,
            shape=shape,
            ocpus=ocpus,
            memory_gb=memory_gb,
            image_id=image_id,
            ssh_public_key=ssh_public_key,
            boot_volume_size_gb=boot_volume_size_gb,
            assign_public_ip=assign_public_ip,
        )
