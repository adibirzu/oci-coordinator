import json
from datetime import datetime, timedelta, timezone
from typing import Any

from opentelemetry import trace

from src.mcp.server.auth import get_compute_client, get_monitoring_client, get_oci_config

# Get tracer for observability tools
_tracer = trace.get_tracer("mcp-oci-observability")


async def _get_metrics_logic(
    compartment_id: str,
    namespace: str,
    query: str,
    format: str = "markdown"
) -> str:
    """Internal logic for getting metrics."""
    with _tracer.start_as_current_span("mcp.observability.get_metrics") as span:
        try:
            config = get_oci_config()
            client = get_monitoring_client()
            
            span.set_attribute("compartment_id", compartment_id)
            span.set_attribute("namespace", namespace)
            span.set_attribute("query", query[:100])
            
            # Calculate time range (last 1 hour by default)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)
            
            # Build the summarize metrics request
            from oci.monitoring.models import SummarizeMetricsDataDetails
            
            summarize_details = SummarizeMetricsDataDetails(
                namespace=namespace,
                query=query,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                resolution="5m",  # 5-minute resolution
            )
            
            response = client.summarize_metrics_data(
                compartment_id=compartment_id,
                summarize_metrics_data_details=summarize_details,
            )
            
            metrics = response.data
            span.set_attribute("metric_count", len(metrics) if metrics else 0)
            
            if not metrics:
                return json.dumps({
                    "type": "metrics",
                    "namespace": namespace,
                    "query": query,
                    "time_range": {"start": start_time.isoformat(), "end": end_time.isoformat()},
                    "message": "No metrics found for the specified query",
                    "data": []
                })
            
            # Format metrics data
            metrics_data = []
            for m in metrics:
                metric_entry = {
                    "name": m.name,
                    "namespace": m.namespace,
                    "resource_group": getattr(m, "resource_group", None),
                    "dimensions": dict(m.dimensions) if hasattr(m, "dimensions") and m.dimensions else {},
                    "datapoints": []
                }
                
                if hasattr(m, "aggregated_datapoints") and m.aggregated_datapoints:
                    for dp in m.aggregated_datapoints[-10:]:  # Last 10 datapoints
                        metric_entry["datapoints"].append({
                            "timestamp": str(dp.timestamp) if dp.timestamp else None,
                            "value": dp.value,
                            "count": getattr(dp, "count", None),
                        })
                
                metrics_data.append(metric_entry)
            
            return json.dumps({
                "type": "metrics",
                "namespace": namespace,
                "query": query,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "metric_count": len(metrics_data),
                "data": metrics_data,
            })
            
        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error getting metrics: {e}"})


async def _list_alarms_logic(
    compartment_id: str | None = None,
    lifecycle_state: str | None = None,
    limit: int = 50,
    profile: str | None = None,
) -> str:
    """List alarms from OCI Monitoring service."""
    with _tracer.start_as_current_span("mcp.observability.list_alarms") as span:
        try:
            config = get_oci_config(profile=profile)
            client = get_monitoring_client(profile=profile)

            # Use tenancy as root compartment if not specified
            compartment = compartment_id or config.get("tenancy")
            if not compartment:
                return json.dumps({"error": "No compartment_id provided and tenancy not found"})

            span.set_attribute("compartment_id", compartment)

            # Build request parameters
            kwargs: dict[str, Any] = {
                "compartment_id": compartment,
                "compartment_id_in_subtree": True,
                "limit": limit,
            }
            if lifecycle_state:
                kwargs["lifecycle_state"] = lifecycle_state
                span.set_attribute("lifecycle_state", lifecycle_state)

            # List alarms
            response = client.list_alarms(**kwargs)
            alarms = response.data

            span.set_attribute("alarm_count", len(alarms))

            # Build typed response
            alarm_list = []
            for alarm in alarms:
                severity = getattr(alarm, "severity", "INFO")
                alarm_list.append({
                    "name": alarm.display_name,
                    "state": alarm.lifecycle_state,
                    "severity": severity,
                    "namespace": alarm.namespace,
                    "query": alarm.query[:100] if hasattr(alarm, "query") and alarm.query else None,
                    "is_enabled": alarm.is_enabled,
                })

            return json.dumps({
                "type": "alarms",
                "count": len(alarm_list),
                "alarms": alarm_list,
            })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error listing alarms: {e}"})


async def _get_instance_metrics_logic(
    instance_id: str | None = None,
    instance_name: str | None = None,
    compartment_id: str | None = None,
    hours_back: int = 1,
    profile: str | None = None,
) -> str:
    """Get compute instance metrics (CPU, memory, network, disk).

    Can look up instance by name or OCID. Returns comprehensive metrics
    for capacity analysis and troubleshooting.
    """
    with _tracer.start_as_current_span("mcp.observability.get_instance_metrics") as span:
        try:
            config = get_oci_config(profile=profile)
            monitoring_client = get_monitoring_client(profile=profile)

            # Use tenancy as fallback compartment
            compartment = compartment_id or config.get("tenancy")

            # If instance_name provided, look up the instance
            actual_instance_id = instance_id
            instance_display_name = None

            if instance_name and not instance_id:
                if not compartment:
                    return json.dumps({
                        "error": "compartment_id is required when using instance_name lookup"
                    })

                # Search for instance by name
                compute_client = get_compute_client()
                try:
                    response = compute_client.list_instances(
                        compartment_id=compartment,
                        lifecycle_state="RUNNING",
                    )
                    instances = response.data or []

                    # Also check stopped instances
                    response2 = compute_client.list_instances(
                        compartment_id=compartment,
                        lifecycle_state="STOPPED",
                    )
                    instances.extend(response2.data or [])

                    # Find matching instance
                    search_name = instance_name.lower()
                    matches = [i for i in instances if search_name in (i.display_name or "").lower()]

                    if not matches:
                        return json.dumps({
                            "type": "instance_metrics",
                            "error": f"No instance found matching '{instance_name}'",
                            "suggestion": "Use 'list instances' to see available instances"
                        })

                    if len(matches) > 1:
                        return json.dumps({
                            "type": "instance_metrics",
                            "error": f"Multiple instances match '{instance_name}'",
                            "matches": [{"name": i.display_name, "id": i.id, "state": i.lifecycle_state} for i in matches],
                            "suggestion": "Please provide the full instance OCID or a more specific name"
                        })

                    actual_instance_id = matches[0].id
                    instance_display_name = matches[0].display_name
                    compartment = matches[0].compartment_id

                except Exception as e:
                    return json.dumps({
                        "error": f"Failed to look up instance by name: {e}"
                    })

            if not actual_instance_id:
                return json.dumps({
                    "error": "Either instance_id or instance_name is required"
                })

            span.set_attribute("instance_id", actual_instance_id)
            span.set_attribute("hours_back", hours_back)

            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)

            from oci.monitoring.models import SummarizeMetricsDataDetails

            # Define metrics to collect
            metric_queries = {
                "cpu_utilization": f'CpuUtilization[1m]{{resourceId = "{actual_instance_id}"}}.mean()',
                "memory_utilization": f'MemoryUtilization[1m]{{resourceId = "{actual_instance_id}"}}.mean()',
                "disk_read_bytes": f'DiskBytesRead[1m]{{resourceId = "{actual_instance_id}"}}.rate()',
                "disk_write_bytes": f'DiskBytesWritten[1m]{{resourceId = "{actual_instance_id}"}}.rate()',
                "network_in_bytes": f'NetworksBytesIn[1m]{{resourceId = "{actual_instance_id}"}}.rate()',
                "network_out_bytes": f'NetworksBytesOut[1m]{{resourceId = "{actual_instance_id}"}}.rate()',
            }

            results = {
                "type": "instance_metrics",
                "instance_id": actual_instance_id,
                "instance_name": instance_display_name,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours_back,
                },
                "metrics": {},
                "summary": {},
            }

            # Query each metric
            for metric_name, query in metric_queries.items():
                try:
                    details = SummarizeMetricsDataDetails(
                        namespace="oci_computeagent",
                        query=query,
                        start_time=start_time.isoformat(),
                        end_time=end_time.isoformat(),
                        resolution="5m",
                    )

                    response = monitoring_client.summarize_metrics_data(
                        compartment_id=compartment,
                        summarize_metrics_data_details=details,
                    )

                    if response.data:
                        datapoints = []
                        values = []
                        for m in response.data:
                            if hasattr(m, "aggregated_datapoints") and m.aggregated_datapoints:
                                for dp in m.aggregated_datapoints[-12:]:  # Last 12 points (1 hour at 5m resolution)
                                    if dp.value is not None:
                                        values.append(dp.value)
                                        datapoints.append({
                                            "timestamp": str(dp.timestamp),
                                            "value": round(dp.value, 2),
                                        })

                        results["metrics"][metric_name] = datapoints
                        if values:
                            results["summary"][metric_name] = {
                                "avg": round(sum(values) / len(values), 2),
                                "min": round(min(values), 2),
                                "max": round(max(values), 2),
                                "latest": round(values[-1], 2) if values else None,
                            }
                except Exception as e:
                    results["metrics"][metric_name] = {"error": str(e)}

            # Add health assessment
            cpu_avg = results["summary"].get("cpu_utilization", {}).get("avg", 0)
            mem_avg = results["summary"].get("memory_utilization", {}).get("avg", 0)

            health_status = "healthy"
            recommendations = []

            if cpu_avg > 90:
                health_status = "critical"
                recommendations.append("CPU critically high - consider scaling up or load balancing")
            elif cpu_avg > 80:
                health_status = "warning"
                recommendations.append("CPU utilization high - monitor for sustained load")
            elif cpu_avg < 10:
                recommendations.append("CPU utilization low - consider rightsizing to reduce costs")

            if mem_avg > 90:
                health_status = "critical"
                recommendations.append("Memory critically high - add memory or optimize applications")
            elif mem_avg > 80:
                if health_status != "critical":
                    health_status = "warning"
                recommendations.append("Memory utilization high - monitor for OOM issues")

            results["health"] = {
                "status": health_status,
                "cpu_avg": cpu_avg,
                "memory_avg": mem_avg,
                "recommendations": recommendations,
            }

            return json.dumps(results)

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({"error": f"Error getting instance metrics: {e}"})


async def _list_oci_profiles_logic() -> str:
    """List available OCI profiles from config file."""
    import configparser
    import os

    config_file = os.path.expanduser(os.getenv("OCI_CONFIG_FILE", "~/.oci/config"))

    try:
        if not os.path.exists(config_file):
            return json.dumps({
                "error": f"OCI config file not found: {config_file}",
                "suggestion": "Create ~/.oci/config with your OCI credentials"
            })

        parser = configparser.ConfigParser()
        parser.read(config_file)

        profiles = []
        for section in parser.sections():
            profile_info = {
                "name": section,
                "region": parser.get(section, "region", fallback="N/A"),
                "has_tenancy": parser.has_option(section, "tenancy"),
            }
            profiles.append(profile_info)

        return json.dumps({
            "type": "oci_profiles",
            "config_file": config_file,
            "count": len(profiles),
            "profiles": profiles,
        })

    except Exception as e:
        return json.dumps({"error": f"Error reading OCI config: {e}"})


def register_observability_tools(mcp):
    @mcp.tool()
    async def oci_observability_get_instance_metrics(
        instance_id: str | None = None,
        instance_name: str | None = None,
        compartment_id: str | None = None,
        hours_back: int = 1,
        profile: str | None = None,
    ) -> str:
        """Get compute instance metrics (CPU, memory, network, disk).

        Can look up instance by name or OCID. Returns comprehensive metrics
        for capacity analysis, health assessment, and troubleshooting.

        Args:
            instance_id: OCID of the instance (optional if instance_name provided)
            instance_name: Display name of the instance (will search and match)
            compartment_id: OCID of the compartment (defaults to tenancy root)
            hours_back: Number of hours of metrics to retrieve (default 1)
            profile: OCI profile name

        Returns:
            JSON with CPU, memory, disk, network metrics plus health assessment
        """
        return await _get_instance_metrics_logic(
            instance_id=instance_id,
            instance_name=instance_name,
            compartment_id=compartment_id,
            hours_back=hours_back,
            profile=profile,
        )

    @mcp.tool()
    async def oci_list_profiles() -> str:
        """List available OCI profiles from the config file.

        Returns the profiles configured in ~/.oci/config with their regions.
        Useful for multi-tenancy support and profile selection.

        Returns:
            JSON with list of profile names and regions
        """
        return await _list_oci_profiles_logic()

    @mcp.tool()
    async def oci_observability_get_metrics(compartment_id: str, namespace: str, query: str, format: str = "markdown") -> str:
        """Get monitoring metrics from OCI Monitoring service.

        Args:
            compartment_id: OCID of the compartment
            namespace: Metric namespace (e.g., 'oci_computeagent')
            query: MQL query string
            format: Output format ('json' or 'markdown')

        Returns:
            Metric data points
        """
        return await _get_metrics_logic(compartment_id, namespace, query, format)

    @mcp.tool()
    async def oci_observability_list_alarms(
        compartment_id: str | None = None,
        lifecycle_state: str | None = None,
        limit: int = 50,
        profile: str | None = None,
    ) -> str:
        """List alarms from OCI Monitoring service.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root)
            lifecycle_state: Filter by state (ACTIVE, INACTIVE, DELETED)
            limit: Maximum number of alarms to return (default 50)
            profile: OCI profile name

        Returns:
            JSON with alarm list including name, state, severity, namespace
        """
        return await _list_alarms_logic(compartment_id, lifecycle_state, limit, profile)
