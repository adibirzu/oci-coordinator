from typing import Optional, List, Dict, Any
import json
from src.mcp.server.auth import get_compute_client, get_monitoring_client

async def _troubleshoot_instance_logic(instance_id: str) -> str:
    """Internal logic for troubleshooting an instance."""
    compute = get_compute_client()
    monitoring = get_monitoring_client()
    
    try:
        # 1. Check instance state
        instance = compute.get_instance(instance_id=instance_id).data
        state = instance.lifecycle_state
        name = instance.display_name
        
        # 2. (Simulated) Check alarms and metrics
        # In a real implementation, we would query OCI Monitoring
        
        # 3. Generate RCA Report
        report = [
            f"# Root Cause Analysis: {name}",
            f"**Current State**: {state}",
            "",
            "## Findings",
            f"- Instance is in {state} state." if state != "RUNNING" else "- Instance is RUNNING but user reports issues.",
            "- No active alarms detected (simulated).",
            "",
            "## Recommendations",
            "1. Check console logs for boot errors." if state != "RUNNING" else "1. Check OS-level metrics (CPU/Memory).",
            "2. Verify security list rules for connectivity."
        ]
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error troubleshooting instance: {e}"

def register_troubleshoot_skills(mcp):
    """Register troubleshooting skills."""
    
    @mcp.tool()
    async def troubleshoot_instance(instance_id: str) -> str:
        """Perform a multi-step troubleshooting analysis on a compute instance. 
        
        Args:
            instance_id: OCID of the compute instance
        """
        return await _troubleshoot_instance_logic(instance_id)
