from typing import Optional, List, Dict, Any
import json
import oci
from src.mcp.server.auth import get_usage_client
from datetime import datetime, timedelta

async def _get_cost_summary_logic(
    compartment_id: str,
    days: int = 30,
    format: str = "markdown"
) -> str:
    """Internal logic for cost summary."""
    client = get_usage_client()
    
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        request = oci.usage_api.models.RequestSummarizedUsagesDetails(
            tenant_id=compartment_id,
            time_usage_started=start_time,
            time_usage_ended=end_time,
            granularity="MONTHLY",
            query_type="USAGE"
        )
        
        response = client.request_summarized_usages(request)
        usages = response.data.items
        
        if format == "json":
            return json.dumps([{"amount": u.computed_amount, "currency": u.currency} for u in usages], indent=2)
            
        total = sum(u.computed_amount for u in usages)
        currency = usages[0].currency if usages else "USD"
        
        return f"Total cost for last {days} days: **{total:.2f} {currency}**"
        
    except Exception as e:
        return f"Error getting cost summary: {e}"

def register_cost_tools(mcp):
    @mcp.tool()
    async def get_cost_summary(compartment_id: str, days: int = 30, format: str = "markdown") -> str:
        """Get summarized cost for a compartment."""
        return await _get_cost_summary_logic(compartment_id, days, format)