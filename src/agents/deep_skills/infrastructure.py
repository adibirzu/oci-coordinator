"""
Infrastructure DeepSkills.

Advanced skills for compute, network, and resource management:
- InstanceHealthSkill: Instance health monitoring and diagnostics
- NetworkAnalysisSkill: Network topology and security analysis
- ComputeMetricsSkill: Performance metrics and capacity planning
"""

from src.agents.core.deep_skills import (
    DeepSkill,
    SkillContext,
    SkillResult,
    SkillStatus,
    register_skill,
)
import structlog

logger = structlog.get_logger()


# =============================================================================
# Instance Health Skill
# =============================================================================

@register_skill(
    skill_id="instance_health_check",
    name="Instance Health Check",
    description="Comprehensive health analysis for compute instances including metrics, status, and recommendations",
    compatible_agents=["infrastructure", "coordinator"],
    required_mcp_tools=[
        "oci_compute_list_instances",
        "oci_compute_get_instance",
        "oci_observability_get_instance_metrics",
    ],
    requires_code_execution=True,
    tags=["compute", "health", "monitoring", "instances"],
)
class InstanceHealthSkill(DeepSkill):
    """
    Analyzes compute instance health by combining:
    1. Instance state and configuration from Compute API
    2. CPU/Memory/Disk metrics from Monitoring
    3. Health scoring and recommendations via code execution
    """

    HEALTH_ANALYSIS_CODE = '''
# Instance Health Analysis
# Input: instances, metrics
# Output: health_report with scores and recommendations

def calculate_health_score(instance, instance_metrics):
    """Calculate health score (0-100) for an instance."""
    score = 100
    issues = []
    recommendations = []

    # Check instance state
    state = instance.get("lifecycle_state", "UNKNOWN")
    if state != "RUNNING":
        score -= 30
        issues.append(f"Instance not running (state: {state})")
        if state == "STOPPED":
            recommendations.append("Consider starting the instance if needed")

    # Check metrics if available
    if instance_metrics:
        # CPU analysis
        cpu_avg = instance_metrics.get("cpu_avg", 0)
        cpu_max = instance_metrics.get("cpu_max", 0)

        if cpu_avg > 90:
            score -= 20
            issues.append(f"Critical CPU usage: {cpu_avg:.1f}% average")
            recommendations.append("Consider scaling up or optimizing workload")
        elif cpu_avg > 75:
            score -= 10
            issues.append(f"High CPU usage: {cpu_avg:.1f}% average")
            recommendations.append("Monitor CPU trends, plan for scaling if sustained")
        elif cpu_avg < 10 and state == "RUNNING":
            issues.append(f"Low CPU utilization: {cpu_avg:.1f}% - may be over-provisioned")
            recommendations.append("Consider rightsizing to reduce costs")

        # Memory analysis
        mem_avg = instance_metrics.get("memory_avg", 0)
        if mem_avg > 90:
            score -= 15
            issues.append(f"Critical memory usage: {mem_avg:.1f}%")
            recommendations.append("Add memory or optimize memory-intensive processes")
        elif mem_avg > 80:
            score -= 5
            issues.append(f"High memory usage: {mem_avg:.1f}%")

    # Determine health status
    if score >= 90:
        status = "HEALTHY"
    elif score >= 70:
        status = "WARNING"
    elif score >= 50:
        status = "DEGRADED"
    else:
        status = "CRITICAL"

    return {
        "score": score,
        "status": status,
        "issues": issues,
        "recommendations": recommendations
    }

def analyze_fleet_health(instances, metrics_by_instance):
    """Analyze health across all instances."""
    results = []
    total_score = 0
    status_counts = {"HEALTHY": 0, "WARNING": 0, "DEGRADED": 0, "CRITICAL": 0}
    all_recommendations = []

    for instance in instances:
        instance_id = instance.get("id", "unknown")
        instance_name = instance.get("display_name", "unnamed")
        instance_metrics = metrics_by_instance.get(instance_id, {})

        health = calculate_health_score(instance, instance_metrics)
        total_score += health["score"]
        status_counts[health["status"]] += 1

        results.append({
            "instance_id": instance_id,
            "instance_name": instance_name,
            "shape": instance.get("shape", "unknown"),
            "state": instance.get("lifecycle_state", "UNKNOWN"),
            "health_score": health["score"],
            "health_status": health["status"],
            "issues": health["issues"],
            "recommendations": health["recommendations"]
        })

        # Collect unique recommendations
        for rec in health["recommendations"]:
            if rec not in all_recommendations:
                all_recommendations.append(rec)

    # Sort by health score (worst first)
    results.sort(key=lambda x: x["health_score"])

    avg_score = total_score / len(instances) if instances else 0

    return {
        "summary": {
            "total_instances": len(instances),
            "average_health_score": round(avg_score, 1),
            "status_distribution": status_counts,
            "critical_count": status_counts["CRITICAL"],
            "warning_count": status_counts["WARNING"]
        },
        "instances": results,
        "fleet_recommendations": all_recommendations[:10]  # Top 10
    }

# Execute analysis
result = analyze_fleet_health(instances, metrics)
'''

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute instance health analysis."""
        compartment_id = context.parameters.get("compartment_id")
        instance_id = context.parameters.get("instance_id")  # Optional: specific instance
        include_metrics = context.parameters.get("include_metrics", True)

        try:
            # Step 1: Get instances
            if instance_id:
                # Single instance
                instance_result = await self.call_mcp_tool(
                    context,
                    "oci_compute_get_instance",
                    {
                        "params": {
                            "instance_id": instance_id,
                            "include_metrics": include_metrics,
                            "response_format": "json"
                        }
                    }
                )
                instances = [instance_result] if instance_result else []
            else:
                # List instances in compartment
                list_result = await self.call_mcp_tool(
                    context,
                    "oci_compute_list_instances",
                    {
                        "params": {
                            "compartment_id": compartment_id,
                            "lifecycle_state": None,  # All states
                            "limit": 50,
                            "response_format": "json"
                        }
                    }
                )
                instances = list_result.get("instances", []) if isinstance(list_result, dict) else []

            if not instances:
                return SkillResult(
                    success=True,
                    status=SkillStatus.COMPLETED,
                    data={"message": "No instances found", "instances": []},
                    summary="No compute instances found in the specified scope"
                )

            # Step 2: Get metrics for each instance
            metrics_by_instance = {}
            if include_metrics:
                for instance in instances[:20]:  # Limit to avoid timeout
                    inst_id = instance.get("id")
                    if inst_id and instance.get("lifecycle_state") == "RUNNING":
                        try:
                            metrics_result = await self.call_mcp_tool(
                                context,
                                "oci_observability_get_instance_metrics",
                                {
                                    "params": {
                                        "instance_id": inst_id,
                                        "window": "1h",
                                        "response_format": "json"
                                    }
                                }
                            )
                            if metrics_result and not metrics_result.get("error"):
                                metrics_by_instance[inst_id] = {
                                    "cpu_avg": metrics_result.get("cpu_avg", 0),
                                    "cpu_max": metrics_result.get("cpu_max", 0),
                                    "memory_avg": metrics_result.get("memory_avg", 0),
                                    "memory_max": metrics_result.get("memory_max", 0)
                                }
                        except Exception as e:
                            logger.debug("metrics_fetch_failed", instance_id=inst_id, error=str(e))

            # Step 3: Analyze health
            analysis = await self.execute_code(
                context,
                self.HEALTH_ANALYSIS_CODE,
                variables={
                    "instances": instances,
                    "metrics": metrics_by_instance
                }
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    status=SkillStatus.FAILED,
                    error=f"Health analysis failed: {analysis.error}"
                )

            result = analysis.result
            summary = result.get("summary", {})

            return SkillResult(
                success=True,
                status=SkillStatus.COMPLETED,
                data=result,
                summary=f"Analyzed {summary.get('total_instances', 0)} instances. "
                        f"Average health: {summary.get('average_health_score', 0)}%. "
                        f"Critical: {summary.get('critical_count', 0)}, "
                        f"Warning: {summary.get('warning_count', 0)}"
            )

        except Exception as e:
            logger.error("instance_health_skill_error", error=str(e))
            return SkillResult(
                success=False,
                status=SkillStatus.FAILED,
                error=str(e)
            )

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test instance health skill."""
        # Test code execution with mock data
        test_instances = [
            {"id": "inst-1", "display_name": "test-1", "lifecycle_state": "RUNNING", "shape": "VM.Standard.E4.Flex"},
            {"id": "inst-2", "display_name": "test-2", "lifecycle_state": "STOPPED", "shape": "VM.Standard.E4.Flex"}
        ]
        test_metrics = {
            "inst-1": {"cpu_avg": 45, "cpu_max": 80, "memory_avg": 60, "memory_max": 75}
        }

        analysis = await self.execute_code(
            context,
            self.HEALTH_ANALYSIS_CODE,
            variables={"instances": test_instances, "metrics": test_metrics}
        )

        if not analysis.success:
            return SkillResult(
                success=False,
                status=SkillStatus.FAILED,
                error=f"Code execution test failed: {analysis.error}"
            )

        return SkillResult(
            success=True,
            status=SkillStatus.COMPLETED,
            data={"test": "passed", "analysis_sample": analysis.result}
        )


# =============================================================================
# Network Analysis Skill
# =============================================================================

@register_skill(
    skill_id="network_security_analysis",
    name="Network Security Analysis",
    description="Analyze VCN topology, security lists, and identify network security risks",
    compatible_agents=["infrastructure", "security", "coordinator"],
    required_mcp_tools=[
        "oci_network_list_vcns",
        "oci_network_get_vcn",
        "oci_network_list_subnets",
        "oci_network_list_security_lists",
        "oci_network_analyze_security",
    ],
    requires_code_execution=True,
    tags=["network", "security", "vcn", "firewall"],
)
class NetworkAnalysisSkill(DeepSkill):
    """
    Analyzes network configuration and security by:
    1. Mapping VCN topology (subnets, gateways)
    2. Analyzing security list rules
    3. Identifying risky configurations
    """

    NETWORK_ANALYSIS_CODE = '''
# Network Security Analysis
# Input: vcns, subnets, security_lists, security_analysis
# Output: comprehensive network security report

def categorize_risk(rule):
    """Categorize risk level of a security rule."""
    source = rule.get("source", "")
    dest = rule.get("destination", "")
    protocol = str(rule.get("protocol", "")).lower()
    port_min = rule.get("tcp_options", {}).get("destination_port_range", {}).get("min")
    port_max = rule.get("tcp_options", {}).get("destination_port_range", {}).get("max")

    # Check for overly permissive rules
    if source == "0.0.0.0/0" or dest == "0.0.0.0/0":
        # Check for dangerous ports
        dangerous_ports = {22, 23, 3389, 1433, 3306, 5432, 27017, 6379}
        if port_min and port_max:
            for port in range(port_min, min(port_max + 1, port_min + 100)):
                if port in dangerous_ports:
                    return "CRITICAL", f"Public access to sensitive port {port}"
        if protocol == "all" or (port_min == 1 and port_max == 65535):
            return "CRITICAL", "All ports open to internet"
        return "HIGH", "Public access configured"

    # Internal ranges are lower risk
    if source.startswith("10.") or source.startswith("172.") or source.startswith("192.168."):
        return "LOW", "Internal network access"

    return "MEDIUM", "Standard rule"

def analyze_security_lists(security_lists):
    """Analyze all security lists for risks."""
    findings = []

    for sl in security_lists:
        sl_name = sl.get("display_name", "unnamed")
        sl_id = sl.get("id", "unknown")

        # Analyze ingress rules
        for rule in sl.get("ingress_security_rules", []):
            risk_level, reason = categorize_risk(rule)
            if risk_level in ["CRITICAL", "HIGH"]:
                findings.append({
                    "security_list": sl_name,
                    "security_list_id": sl_id,
                    "rule_type": "INGRESS",
                    "risk_level": risk_level,
                    "reason": reason,
                    "source": rule.get("source", "N/A"),
                    "protocol": rule.get("protocol", "N/A")
                })

        # Analyze egress rules
        for rule in sl.get("egress_security_rules", []):
            risk_level, reason = categorize_risk(rule)
            if risk_level in ["CRITICAL", "HIGH"]:
                findings.append({
                    "security_list": sl_name,
                    "security_list_id": sl_id,
                    "rule_type": "EGRESS",
                    "risk_level": risk_level,
                    "reason": reason,
                    "destination": rule.get("destination", "N/A"),
                    "protocol": rule.get("protocol", "N/A")
                })

    return findings

def build_topology_summary(vcns, subnets):
    """Build a summary of network topology."""
    topology = []

    for vcn in vcns:
        vcn_id = vcn.get("id", "")
        vcn_subnets = [s for s in subnets if s.get("vcn_id") == vcn_id]

        public_subnets = len([s for s in vcn_subnets if not s.get("prohibit_public_ip_on_vnic", True)])
        private_subnets = len(vcn_subnets) - public_subnets

        topology.append({
            "vcn_name": vcn.get("display_name", "unnamed"),
            "vcn_id": vcn_id,
            "cidr_blocks": vcn.get("cidr_blocks", []),
            "state": vcn.get("lifecycle_state", "UNKNOWN"),
            "total_subnets": len(vcn_subnets),
            "public_subnets": public_subnets,
            "private_subnets": private_subnets
        })

    return topology

def generate_recommendations(findings, topology):
    """Generate security recommendations."""
    recommendations = []

    # Check for critical findings
    critical_count = len([f for f in findings if f["risk_level"] == "CRITICAL"])
    if critical_count > 0:
        recommendations.append({
            "priority": "CRITICAL",
            "recommendation": f"Address {critical_count} critical security rule(s) with public internet access to sensitive ports",
            "action": "Review and restrict ingress rules to specific IP ranges"
        })

    # Check for public subnets
    total_public = sum(t["public_subnets"] for t in topology)
    if total_public > 0:
        recommendations.append({
            "priority": "HIGH",
            "recommendation": f"Review {total_public} public subnet(s) for necessity",
            "action": "Ensure only required resources are in public subnets"
        })

    # General recommendations
    if not recommendations:
        recommendations.append({
            "priority": "INFO",
            "recommendation": "No critical security issues detected",
            "action": "Continue monitoring and maintain current security posture"
        })

    return recommendations

# Build analysis
security_findings = analyze_security_lists(security_lists)
topology_summary = build_topology_summary(vcns, subnets)
recommendations = generate_recommendations(security_findings, topology_summary)

# Calculate security score
critical_count = len([f for f in security_findings if f["risk_level"] == "CRITICAL"])
high_count = len([f for f in security_findings if f["risk_level"] == "HIGH"])
security_score = max(0, 100 - (critical_count * 25) - (high_count * 10))

result = {
    "summary": {
        "total_vcns": len(vcns),
        "total_subnets": len(subnets),
        "total_security_lists": len(security_lists),
        "security_score": security_score,
        "critical_findings": critical_count,
        "high_findings": high_count
    },
    "topology": topology_summary,
    "security_findings": security_findings[:20],  # Top 20
    "recommendations": recommendations
}
'''

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute network security analysis."""
        compartment_id = context.parameters.get("compartment_id")
        vcn_id = context.parameters.get("vcn_id")  # Optional: specific VCN

        try:
            # Step 1: Get VCNs
            if vcn_id:
                vcn_result = await self.call_mcp_tool(
                    context,
                    "oci_network_get_vcn",
                    {"params": {"vcn_id": vcn_id, "response_format": "json"}}
                )
                vcns = [vcn_result] if vcn_result else []
            else:
                vcn_result = await self.call_mcp_tool(
                    context,
                    "oci_network_list_vcns",
                    {"params": {"compartment_id": compartment_id, "response_format": "json"}}
                )
                vcns = vcn_result.get("vcns", []) if isinstance(vcn_result, dict) else []

            if not vcns:
                return SkillResult(
                    success=True,
                    status=SkillStatus.COMPLETED,
                    data={"message": "No VCNs found"},
                    summary="No Virtual Cloud Networks found in the specified scope"
                )

            # Step 2: Get subnets
            subnet_result = await self.call_mcp_tool(
                context,
                "oci_network_list_subnets",
                {
                    "params": {
                        "compartment_id": compartment_id,
                        "vcn_id": vcn_id,
                        "response_format": "json"
                    }
                }
            )
            subnets = subnet_result.get("subnets", []) if isinstance(subnet_result, dict) else []

            # Step 3: Get security lists
            sl_result = await self.call_mcp_tool(
                context,
                "oci_network_list_security_lists",
                {
                    "params": {
                        "compartment_id": compartment_id,
                        "vcn_id": vcn_id,
                        "response_format": "json"
                    }
                }
            )
            security_lists = sl_result.get("security_lists", []) if isinstance(sl_result, dict) else []

            # Step 4: Get security analysis from MCP tool
            security_analysis = {}
            if vcn_id:
                try:
                    analysis_result = await self.call_mcp_tool(
                        context,
                        "oci_network_analyze_security",
                        {"params": {"vcn_id": vcn_id, "response_format": "json"}}
                    )
                    security_analysis = analysis_result if isinstance(analysis_result, dict) else {}
                except Exception:
                    pass  # Analysis is optional

            # Step 5: Run comprehensive analysis
            analysis = await self.execute_code(
                context,
                self.NETWORK_ANALYSIS_CODE,
                variables={
                    "vcns": vcns,
                    "subnets": subnets,
                    "security_lists": security_lists,
                    "security_analysis": security_analysis
                }
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    status=SkillStatus.FAILED,
                    error=f"Network analysis failed: {analysis.error}"
                )

            result = analysis.result
            summary = result.get("summary", {})

            return SkillResult(
                success=True,
                status=SkillStatus.COMPLETED,
                data=result,
                summary=f"Analyzed {summary.get('total_vcns', 0)} VCN(s), "
                        f"{summary.get('total_subnets', 0)} subnet(s). "
                        f"Security score: {summary.get('security_score', 0)}/100. "
                        f"Critical findings: {summary.get('critical_findings', 0)}"
            )

        except Exception as e:
            logger.error("network_analysis_skill_error", error=str(e))
            return SkillResult(
                success=False,
                status=SkillStatus.FAILED,
                error=str(e)
            )

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test network analysis skill."""
        test_vcns = [{"id": "vcn-1", "display_name": "test-vcn", "cidr_blocks": ["10.0.0.0/16"], "lifecycle_state": "AVAILABLE"}]
        test_subnets = [{"id": "subnet-1", "vcn_id": "vcn-1", "prohibit_public_ip_on_vnic": False}]
        test_sls = [{"id": "sl-1", "display_name": "test-sl", "ingress_security_rules": [], "egress_security_rules": []}]

        analysis = await self.execute_code(
            context,
            self.NETWORK_ANALYSIS_CODE,
            variables={
                "vcns": test_vcns,
                "subnets": test_subnets,
                "security_lists": test_sls,
                "security_analysis": {}
            }
        )

        return SkillResult(
            success=analysis.success,
            status=SkillStatus.COMPLETED if analysis.success else SkillStatus.FAILED,
            data={"test": "passed"} if analysis.success else None,
            error=analysis.error if not analysis.success else None
        )


# =============================================================================
# Compute Metrics Skill
# =============================================================================

@register_skill(
    skill_id="compute_metrics_analysis",
    name="Compute Metrics Analysis",
    description="Analyze compute resource utilization, identify trends, and provide capacity recommendations",
    compatible_agents=["infrastructure", "finops", "coordinator"],
    required_mcp_tools=[
        "oci_observability_get_instance_metrics",
        "oci_compute_list_instances",
        "oci_observability_list_alarms",
    ],
    requires_code_execution=True,
    tags=["compute", "metrics", "capacity", "optimization"],
)
class ComputeMetricsSkill(DeepSkill):
    """
    Analyzes compute metrics for capacity planning and optimization:
    1. Collects CPU/Memory/Disk metrics across instances
    2. Identifies utilization patterns
    3. Provides rightsizing and capacity recommendations
    """

    METRICS_ANALYSIS_CODE = '''
# Compute Metrics Analysis
# Input: instances, metrics_data, alarms
# Output: utilization analysis with recommendations

def analyze_utilization(instance, metrics):
    """Analyze utilization for a single instance."""
    cpu_avg = metrics.get("cpu_avg", 0)
    cpu_max = metrics.get("cpu_max", 0)
    mem_avg = metrics.get("memory_avg", 0)
    mem_max = metrics.get("memory_max", 0)

    # Categorize utilization
    if cpu_avg < 10 and mem_avg < 20:
        category = "UNDERUTILIZED"
        recommendation = "Consider downsizing or consolidating workload"
        potential_savings = "HIGH"
    elif cpu_avg > 80 or mem_avg > 85:
        category = "OVERUTILIZED"
        recommendation = "Consider scaling up or out"
        potential_savings = "N/A"
    elif cpu_avg > 50 or mem_avg > 60:
        category = "OPTIMAL"
        recommendation = "Resources well-utilized"
        potential_savings = "LOW"
    else:
        category = "MODERATE"
        recommendation = "Monitor for optimization opportunities"
        potential_savings = "MEDIUM"

    return {
        "category": category,
        "recommendation": recommendation,
        "potential_savings": potential_savings,
        "cpu_utilization": {
            "average": round(cpu_avg, 1),
            "max": round(cpu_max, 1),
            "headroom": round(100 - cpu_max, 1)
        },
        "memory_utilization": {
            "average": round(mem_avg, 1),
            "max": round(mem_max, 1),
            "headroom": round(100 - mem_max, 1)
        }
    }

def aggregate_fleet_metrics(instances, metrics_data):
    """Aggregate metrics across the fleet."""
    analysis_results = []
    category_counts = {"UNDERUTILIZED": 0, "MODERATE": 0, "OPTIMAL": 0, "OVERUTILIZED": 0}

    total_cpu_avg = 0
    total_mem_avg = 0
    analyzed_count = 0

    for instance in instances:
        instance_id = instance.get("id", "unknown")
        metrics = metrics_data.get(instance_id, {})

        if not metrics:
            continue

        analyzed_count += 1
        analysis = analyze_utilization(instance, metrics)
        category_counts[analysis["category"]] += 1

        total_cpu_avg += analysis["cpu_utilization"]["average"]
        total_mem_avg += analysis["memory_utilization"]["average"]

        analysis_results.append({
            "instance_id": instance_id,
            "instance_name": instance.get("display_name", "unnamed"),
            "shape": instance.get("shape", "unknown"),
            **analysis
        })

    # Sort by category priority (underutilized first for cost optimization)
    category_order = {"OVERUTILIZED": 0, "UNDERUTILIZED": 1, "MODERATE": 2, "OPTIMAL": 3}
    analysis_results.sort(key=lambda x: category_order.get(x["category"], 4))

    avg_cpu = total_cpu_avg / analyzed_count if analyzed_count > 0 else 0
    avg_mem = total_mem_avg / analyzed_count if analyzed_count > 0 else 0

    return {
        "fleet_summary": {
            "total_instances": len(instances),
            "analyzed_instances": analyzed_count,
            "average_cpu_utilization": round(avg_cpu, 1),
            "average_memory_utilization": round(avg_mem, 1),
            "utilization_distribution": category_counts
        },
        "instances": analysis_results
    }

def generate_capacity_recommendations(fleet_analysis, alarms):
    """Generate capacity and optimization recommendations."""
    recommendations = []
    summary = fleet_analysis.get("fleet_summary", {})
    distribution = summary.get("utilization_distribution", {})

    # Check for underutilization
    underutilized = distribution.get("UNDERUTILIZED", 0)
    if underutilized > 0:
        recommendations.append({
            "type": "COST_OPTIMIZATION",
            "priority": "HIGH",
            "title": f"{underutilized} underutilized instance(s) detected",
            "description": "Consider rightsizing or consolidating workloads",
            "potential_impact": "10-40% cost reduction per instance"
        })

    # Check for overutilization
    overutilized = distribution.get("OVERUTILIZED", 0)
    if overutilized > 0:
        recommendations.append({
            "type": "CAPACITY",
            "priority": "HIGH",
            "title": f"{overutilized} overutilized instance(s) detected",
            "description": "Consider scaling up or distributing load",
            "potential_impact": "Improved performance and reliability"
        })

    # Check active alarms
    active_alarms = [a for a in alarms if a.get("severity") in ["CRITICAL", "WARNING"]]
    if active_alarms:
        recommendations.append({
            "type": "MONITORING",
            "priority": "MEDIUM",
            "title": f"{len(active_alarms)} active alarm(s) require attention",
            "description": "Review and address monitoring alerts",
            "potential_impact": "Proactive issue prevention"
        })

    # Check average utilization
    avg_cpu = summary.get("average_cpu_utilization", 0)
    if avg_cpu < 20:
        recommendations.append({
            "type": "FLEET_OPTIMIZATION",
            "priority": "MEDIUM",
            "title": "Low overall fleet utilization",
            "description": f"Average CPU at {avg_cpu}% - consider fleet consolidation",
            "potential_impact": "Significant cost reduction opportunity"
        })

    return recommendations

# Execute analysis
fleet_analysis = aggregate_fleet_metrics(instances, metrics_data)
recommendations = generate_capacity_recommendations(fleet_analysis, alarms)

result = {
    **fleet_analysis,
    "recommendations": recommendations,
    "active_alarms": len([a for a in alarms if a.get("severity") in ["CRITICAL", "WARNING"]])
}
'''

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute compute metrics analysis."""
        compartment_id = context.parameters.get("compartment_id")
        time_window = context.parameters.get("time_window", "1h")
        include_alarms = context.parameters.get("include_alarms", True)

        try:
            # Step 1: Get instances
            instance_result = await self.call_mcp_tool(
                context,
                "oci_compute_list_instances",
                {
                    "params": {
                        "compartment_id": compartment_id,
                        "lifecycle_state": "RUNNING",
                        "limit": 50,
                        "response_format": "json"
                    }
                }
            )
            instances = instance_result.get("instances", []) if isinstance(instance_result, dict) else []

            if not instances:
                return SkillResult(
                    success=True,
                    status=SkillStatus.COMPLETED,
                    data={"message": "No running instances found"},
                    summary="No running compute instances to analyze"
                )

            # Step 2: Collect metrics for each instance
            metrics_data = {}
            for instance in instances[:20]:  # Limit to avoid timeout
                inst_id = instance.get("id")
                if inst_id:
                    try:
                        metrics_result = await self.call_mcp_tool(
                            context,
                            "oci_observability_get_instance_metrics",
                            {
                                "params": {
                                    "instance_id": inst_id,
                                    "window": time_window,
                                    "response_format": "json"
                                }
                            }
                        )
                        if metrics_result and not metrics_result.get("error"):
                            metrics_data[inst_id] = {
                                "cpu_avg": metrics_result.get("cpu_avg", 0),
                                "cpu_max": metrics_result.get("cpu_max", 0),
                                "memory_avg": metrics_result.get("memory_avg", 0),
                                "memory_max": metrics_result.get("memory_max", 0)
                            }
                    except Exception as e:
                        logger.debug("metrics_fetch_failed", instance_id=inst_id, error=str(e))

            # Step 3: Get alarms
            alarms = []
            if include_alarms:
                try:
                    alarm_result = await self.call_mcp_tool(
                        context,
                        "oci_observability_list_alarms",
                        {
                            "params": {
                                "compartment_id": compartment_id,
                                "response_format": "json"
                            }
                        }
                    )
                    alarms = alarm_result.get("alarms", []) if isinstance(alarm_result, dict) else []
                except Exception:
                    pass  # Alarms are optional

            # Step 4: Analyze metrics
            analysis = await self.execute_code(
                context,
                self.METRICS_ANALYSIS_CODE,
                variables={
                    "instances": instances,
                    "metrics_data": metrics_data,
                    "alarms": alarms
                }
            )

            if not analysis.success:
                return SkillResult(
                    success=False,
                    status=SkillStatus.FAILED,
                    error=f"Metrics analysis failed: {analysis.error}"
                )

            result = analysis.result
            summary = result.get("fleet_summary", {})

            return SkillResult(
                success=True,
                status=SkillStatus.COMPLETED,
                data=result,
                summary=f"Analyzed {summary.get('analyzed_instances', 0)}/{summary.get('total_instances', 0)} instances. "
                        f"Avg CPU: {summary.get('average_cpu_utilization', 0)}%, "
                        f"Avg Memory: {summary.get('average_memory_utilization', 0)}%. "
                        f"Recommendations: {len(result.get('recommendations', []))}"
            )

        except Exception as e:
            logger.error("compute_metrics_skill_error", error=str(e))
            return SkillResult(
                success=False,
                status=SkillStatus.FAILED,
                error=str(e)
            )

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test compute metrics skill."""
        test_instances = [
            {"id": "inst-1", "display_name": "test-1", "shape": "VM.Standard.E4.Flex"},
            {"id": "inst-2", "display_name": "test-2", "shape": "VM.Standard.E4.Flex"}
        ]
        test_metrics = {
            "inst-1": {"cpu_avg": 15, "cpu_max": 40, "memory_avg": 30, "memory_max": 50},
            "inst-2": {"cpu_avg": 85, "cpu_max": 95, "memory_avg": 70, "memory_max": 85}
        }
        test_alarms = []

        analysis = await self.execute_code(
            context,
            self.METRICS_ANALYSIS_CODE,
            variables={
                "instances": test_instances,
                "metrics_data": test_metrics,
                "alarms": test_alarms
            }
        )

        return SkillResult(
            success=analysis.success,
            status=SkillStatus.COMPLETED if analysis.success else SkillStatus.FAILED,
            data={"test": "passed", "sample": analysis.result} if analysis.success else None,
            error=analysis.error if not analysis.success else None
        )
