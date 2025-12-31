"""
OCI Network MCP Tools.

Provides tools for VCN, subnet, and security list management.
"""

import json
from typing import Any

from src.mcp.server.auth import get_network_client


async def _list_vcns_logic(
    compartment_id: str,
    limit: int = 50,
    lifecycle_state: str | None = None,
    format: str = "json",
) -> str:
    """Internal logic for listing VCNs."""
    client = get_network_client()

    try:
        kwargs = {"compartment_id": compartment_id, "limit": limit}
        if lifecycle_state:
            kwargs["lifecycle_state"] = lifecycle_state

        response = client.list_vcns(**kwargs)
        vcns = response.data

        if format == "json":
            return json.dumps(
                [
                    {
                        "name": v.display_name,
                        "id": v.id,
                        "cidr_block": v.cidr_block,
                        "cidr_blocks": v.cidr_blocks,
                        "state": v.lifecycle_state,
                        "dns_label": v.dns_label,
                        "default_route_table_id": v.default_route_table_id,
                        "default_security_list_id": v.default_security_list_id,
                    }
                    for v in vcns
                ],
                indent=2,
            )

        if not vcns:
            return "No VCNs found in this compartment."

        lines = [
            f"Found {len(vcns)} VCNs:\n",
            "| Name | State | CIDR Block | DNS Label |",
            "| --- | --- | --- | --- |",
        ]
        for v in vcns:
            dns = v.dns_label or "N/A"
            lines.append(f"| {v.display_name} | {v.lifecycle_state} | {v.cidr_block} | {dns} |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing VCNs: {e}"


async def _list_subnets_logic(
    compartment_id: str,
    vcn_id: str | None = None,
    limit: int = 50,
    format: str = "json",
) -> str:
    """Internal logic for listing subnets."""
    client = get_network_client()

    try:
        kwargs = {"compartment_id": compartment_id, "limit": limit}
        if vcn_id:
            kwargs["vcn_id"] = vcn_id

        response = client.list_subnets(**kwargs)
        subnets = response.data

        if format == "json":
            return json.dumps(
                [
                    {
                        "name": s.display_name,
                        "id": s.id,
                        "cidr_block": s.cidr_block,
                        "state": s.lifecycle_state,
                        "vcn_id": s.vcn_id,
                        "availability_domain": s.availability_domain,
                        "dns_label": s.dns_label,
                        "prohibit_public_ip": s.prohibit_public_ip_on_vnic,
                    }
                    for s in subnets
                ],
                indent=2,
            )

        if not subnets:
            return "No subnets found."

        lines = [
            f"Found {len(subnets)} subnets:\n",
            "| Name | State | CIDR | Type | AD |",
            "| --- | --- | --- | --- | --- |",
        ]
        for s in subnets:
            subnet_type = "Private" if s.prohibit_public_ip_on_vnic else "Public"
            ad = s.availability_domain.split(":")[-1] if s.availability_domain else "Regional"
            lines.append(f"| {s.display_name} | {s.lifecycle_state} | {s.cidr_block} | {subnet_type} | {ad} |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing subnets: {e}"


async def _list_security_lists_logic(
    compartment_id: str,
    vcn_id: str | None = None,
    limit: int = 50,
    format: str = "json",
) -> str:
    """Internal logic for listing security lists."""
    client = get_network_client()

    try:
        kwargs = {"compartment_id": compartment_id, "limit": limit}
        if vcn_id:
            kwargs["vcn_id"] = vcn_id

        response = client.list_security_lists(**kwargs)
        sec_lists = response.data

        if format == "json":
            return json.dumps(
                [
                    {
                        "name": sl.display_name,
                        "id": sl.id,
                        "state": sl.lifecycle_state,
                        "vcn_id": sl.vcn_id,
                        "ingress_rules_count": len(sl.ingress_security_rules or []),
                        "egress_rules_count": len(sl.egress_security_rules or []),
                    }
                    for sl in sec_lists
                ],
                indent=2,
            )

        if not sec_lists:
            return "No security lists found."

        lines = [
            f"Found {len(sec_lists)} security lists:\n",
            "| Name | State | Ingress Rules | Egress Rules |",
            "| --- | --- | --- | --- |",
        ]
        for sl in sec_lists:
            ingress = len(sl.ingress_security_rules or [])
            egress = len(sl.egress_security_rules or [])
            lines.append(f"| {sl.display_name} | {sl.lifecycle_state} | {ingress} | {egress} |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing security lists: {e}"


def register_network_tools(mcp: Any) -> None:
    """Register network tools with the MCP server."""

    @mcp.tool()
    async def oci_network_list_vcns(
        compartment_id: str,
        limit: int = 50,
        lifecycle_state: str | None = None,
        format: str = "json",
    ) -> str:
        """List OCI Virtual Cloud Networks (VCNs) in a compartment.

        Args:
            compartment_id: OCID of the compartment
            limit: Maximum number of VCNs to return (default 50)
            lifecycle_state: Filter by state (AVAILABLE, PROVISIONING, etc.)
            format: Output format ('json' or 'markdown')

        Returns:
            List of VCNs with name, CIDR block, and state
        """
        return await _list_vcns_logic(compartment_id, limit, lifecycle_state, format)

    @mcp.tool()
    async def oci_network_list_subnets(
        compartment_id: str,
        vcn_id: str | None = None,
        limit: int = 50,
        format: str = "json",
    ) -> str:
        """List OCI subnets in a compartment or VCN.

        Args:
            compartment_id: OCID of the compartment
            vcn_id: Optional OCID of VCN to filter subnets
            limit: Maximum number of subnets to return (default 50)
            format: Output format ('json' or 'markdown')

        Returns:
            List of subnets with name, CIDR, type (public/private), and state
        """
        return await _list_subnets_logic(compartment_id, vcn_id, limit, format)

    @mcp.tool()
    async def oci_network_list_security_lists(
        compartment_id: str,
        vcn_id: str | None = None,
        limit: int = 50,
        format: str = "json",
    ) -> str:
        """List OCI security lists in a compartment or VCN.

        Args:
            compartment_id: OCID of the compartment
            vcn_id: Optional OCID of VCN to filter security lists
            limit: Maximum number to return (default 50)
            format: Output format ('json' or 'markdown')

        Returns:
            List of security lists with ingress/egress rule counts
        """
        return await _list_security_lists_logic(compartment_id, vcn_id, limit, format)
