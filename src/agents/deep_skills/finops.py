"""
FinOps DeepSkills.

Advanced skills for cost analysis and financial operations that combine:
- MCP tool calls (FinOps MCP, OCI Cost APIs)
- Code execution for anomaly detection and trend analysis
- Self-testing capabilities

Skills:
| Skill             | Purpose                              | Primary Tool              |
|-------------------|--------------------------------------|---------------------------|
| CostAnomalySkill  | Detect cost spikes and anomalies     | oci_cost_detect_anomalies |
| CostByServiceSkill| Service cost breakdown and analysis  | oci_cost_by_service       |
| BudgetAlertSkill  | Budget monitoring and alerts         | oci_cost_budget_status    |
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import structlog

from src.agents.core.deep_skills import (
    DeepSkill,
    DeepSkillConfig,
    SkillContext,
    SkillResult,
    register_skill,
)

logger = structlog.get_logger()


# =============================================================================
# Cost Anomaly Detection Skill
# =============================================================================

@register_skill(
    skill_id="finops_cost_anomaly",
    name="Cost Anomaly Detection",
    description="Detect unusual cost patterns, spikes, and anomalies across services",
    compatible_agents=["finops", "coordinator"],
    required_mcp_tools=["oci_cost_detect_anomalies", "oci_cost_monthly_trend"],
    requires_code_execution=True,
    tags=["finops", "cost", "anomaly", "monitoring"],
)
class CostAnomalySkill(DeepSkill):
    """
    Detect cost anomalies using statistical analysis.

    Steps:
    1. Get recent cost data
    2. Apply anomaly detection algorithm
    3. Correlate with service changes
    4. Generate alerts and recommendations
    """

    ANOMALY_ANALYSIS_CODE = """
import json
from statistics import mean, stdev

def analyze_anomalies(anomaly_data, trend_data):
    anomalies = anomaly_data.get('anomalies', [])
    trend = trend_data.get('monthly_costs', [])

    if not anomalies:
        return {
            'has_anomalies': False,
            'summary': 'No cost anomalies detected',
            'anomalies': [],
            'recommendations': []
        }

    # Categorize anomalies by severity
    critical = [a for a in anomalies if a.get('severity') == 'CRITICAL']
    high = [a for a in anomalies if a.get('severity') == 'HIGH']
    medium = [a for a in anomalies if a.get('severity') == 'MEDIUM']

    # Calculate trend baseline for context
    if trend and len(trend) >= 3:
        recent_costs = [t.get('cost', 0) for t in trend[-3:]]
        baseline = mean(recent_costs)
        volatility = stdev(recent_costs) / baseline if baseline > 0 else 0
    else:
        baseline = 0
        volatility = 0

    # Group anomalies by service
    service_anomalies = {}
    for a in anomalies:
        service = a.get('service', 'Unknown')
        if service not in service_anomalies:
            service_anomalies[service] = {
                'count': 0,
                'total_impact': 0
            }
        service_anomalies[service]['count'] += 1
        service_anomalies[service]['total_impact'] += a.get('cost_impact', 0)

    # Sort services by impact
    top_services = sorted(
        service_anomalies.items(),
        key=lambda x: x[1]['total_impact'],
        reverse=True
    )[:5]

    # Generate recommendations
    recommendations = []
    if critical:
        recommendations.append(f'CRITICAL: {len(critical)} critical anomalies require immediate review')
    if top_services:
        top_service = top_services[0]
        recommendations.append(f'Service "{top_service[0]}" has highest anomaly impact - investigate resource usage')
    if volatility > 0.3:
        recommendations.append('High cost volatility detected - consider budget alerts')

    total_impact = sum(a.get('cost_impact', 0) for a in anomalies)

    return {
        'has_anomalies': True,
        'summary': f'{len(anomalies)} anomalies detected, ${total_impact:.2f} total impact',
        'critical_count': len(critical),
        'high_count': len(high),
        'medium_count': len(medium),
        'total_cost_impact': total_impact,
        'top_services': [{'service': s, **d} for s, d in top_services],
        'baseline_monthly_cost': baseline,
        'cost_volatility': volatility,
        'recommendations': recommendations
    }

result = analyze_anomalies(anomaly_data, trend_data)
return result
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute cost anomaly detection."""
        try:
            tenancy_ocid = context.parameters.get("tenancy_ocid")
            days_back = context.parameters.get("days_back", 30)
            threshold = context.parameters.get("threshold", 2.0)

            if not tenancy_ocid:
                return SkillResult(
                    success=False,
                    error="Required parameter: tenancy_ocid"
                )

            # Calculate time range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)

            # Step 1: Get anomaly detection results
            anomaly_result = await self.call_mcp_tool(
                context,
                "oci_cost_detect_anomalies",
                {
                    "params": {
                        "tenancy_ocid": tenancy_ocid,
                        "time_start": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                        "time_end": end_date.strftime("%Y-%m-%dT23:59:59Z"),
                        "threshold": threshold,
                        "response_format": "json"
                    }
                }
            )

            # Step 2: Get monthly trend for context
            trend_result = await self.call_mcp_tool(
                context,
                "oci_cost_monthly_trend",
                {
                    "params": {
                        "tenancy_ocid": tenancy_ocid,
                        "months_back": 6,
                        "response_format": "json"
                    }
                }
            )

            # Step 3: Analyze with code execution
            analysis = await self.execute_code(
                context,
                self.ANOMALY_ANALYSIS_CODE,
                variables={
                    "anomaly_data": anomaly_result,
                    "trend_data": trend_result
                }
            )

            if not analysis.success:
                return SkillResult(
                    success=True,
                    data={
                        "raw_anomalies": anomaly_result,
                        "raw_trend": trend_result,
                        "analysis_error": analysis.error
                    }
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "tenancy_ocid": tenancy_ocid,
                    "days_analyzed": days_back,
                    "threshold": threshold
                }
            )

        except Exception as e:
            logger.error("cost_anomaly_detection_failed", error=str(e))
            return SkillResult(success=False, error=str(e))

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test skill functionality."""
        try:
            result = await self.call_mcp_tool(
                context,
                "oci_cost_ping",
                {}
            )
            success = result.get("status") == "ok" or "pong" in str(result).lower()
            return SkillResult(
                success=success,
                data={"test": "cost_api_connectivity", "result": result} if success else None,
                error=None if success else "Cost API connectivity test failed"
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e), error_type="SelfTestError")


# =============================================================================
# Cost By Service Analysis Skill
# =============================================================================

@register_skill(
    skill_id="finops_cost_by_service",
    name="Cost By Service Analysis",
    description="Analyze costs broken down by service with trending and optimization opportunities",
    compatible_agents=["finops", "coordinator"],
    required_mcp_tools=["oci_cost_by_service"],
    requires_code_execution=True,
    tags=["finops", "cost", "service", "analysis"],
)
class CostByServiceSkill(DeepSkill):
    """
    Analyze costs by OCI service.

    Steps:
    1. Get service cost breakdown
    2. Calculate trends and percentages
    3. Identify optimization opportunities
    4. Generate recommendations
    """

    SERVICE_ANALYSIS_CODE = """
import json

def analyze_service_costs(data):
    services = data.get('services', data.get('items', []))
    total_cost = data.get('total_cost', 0)

    if not services:
        return {
            'has_data': False,
            'summary': 'No service cost data available',
            'services': [],
            'recommendations': []
        }

    # Calculate total if not provided
    if total_cost == 0:
        total_cost = sum(s.get('cost', s.get('amount', 0)) for s in services)

    # Normalize and calculate percentages
    analyzed_services = []
    for svc in services:
        cost = svc.get('cost', svc.get('amount', 0))
        pct = (cost / total_cost * 100) if total_cost > 0 else 0
        analyzed_services.append({
            'service': svc.get('service', svc.get('service_name', 'Unknown')),
            'cost': cost,
            'percentage': round(pct, 2),
            'compartment': svc.get('compartment_name', 'N/A'),
            'trend': svc.get('trend', 'stable')
        })

    # Sort by cost
    analyzed_services.sort(key=lambda x: x['cost'], reverse=True)

    # Calculate concentration (top 3 services share)
    top_3_cost = sum(s['cost'] for s in analyzed_services[:3])
    concentration = (top_3_cost / total_cost * 100) if total_cost > 0 else 0

    # Identify potential optimization targets
    optimization_targets = []
    for svc in analyzed_services:
        if svc['percentage'] > 30:
            optimization_targets.append({
                'service': svc['service'],
                'reason': 'High concentration (>30% of spend)',
                'potential_savings': 'Review for right-sizing'
            })
        if svc['trend'] == 'increasing' and svc['percentage'] > 10:
            optimization_targets.append({
                'service': svc['service'],
                'reason': 'Rapidly increasing cost',
                'potential_savings': 'Investigate usage growth'
            })

    # Generate recommendations
    recommendations = []
    if concentration > 80:
        recommendations.append('High cost concentration in top 3 services - diversification risk')
    if len(optimization_targets) > 0:
        recommendations.append(f'{len(optimization_targets)} services identified for optimization review')

    # Check for common optimization opportunities
    compute_services = [s for s in analyzed_services if 'Compute' in s['service']]
    if compute_services and compute_services[0]['percentage'] > 20:
        recommendations.append('Compute costs significant - consider reserved capacity or right-sizing')

    return {
        'has_data': True,
        'summary': f'{len(analyzed_services)} services, ${total_cost:.2f} total',
        'total_cost': total_cost,
        'service_count': len(analyzed_services),
        'top_services': analyzed_services[:10],
        'top_3_concentration': round(concentration, 2),
        'optimization_targets': optimization_targets[:5],
        'recommendations': recommendations
    }

result = analyze_service_costs(data)
return result
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute cost by service analysis."""
        try:
            tenancy_ocid = context.parameters.get("tenancy_ocid")
            time_start = context.parameters.get("time_start")
            time_end = context.parameters.get("time_end")
            top_n = context.parameters.get("top_n", 10)

            if not tenancy_ocid:
                return SkillResult(
                    success=False,
                    error="Required parameter: tenancy_ocid"
                )

            # Default to last 30 days if no time range specified
            if not time_end:
                time_end = datetime.utcnow().strftime("%Y-%m-%dT23:59:59Z")
            if not time_start:
                start_dt = datetime.utcnow() - timedelta(days=30)
                time_start = start_dt.strftime("%Y-%m-%dT00:00:00Z")

            # Get service cost breakdown
            service_result = await self.call_mcp_tool(
                context,
                "oci_cost_by_service",
                {
                    "params": {
                        "tenancy_ocid": tenancy_ocid,
                        "time_start": time_start,
                        "time_end": time_end,
                        "top_n": top_n,
                        "response_format": "json"
                    }
                }
            )

            # Analyze with code execution
            analysis = await self.execute_code(
                context,
                self.SERVICE_ANALYSIS_CODE,
                variables={"data": service_result}
            )

            if not analysis.success:
                return SkillResult(
                    success=True,
                    data={"raw_data": service_result, "analysis_error": analysis.error}
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "tenancy_ocid": tenancy_ocid,
                    "time_range": {"start": time_start, "end": time_end}
                }
            )

        except Exception as e:
            logger.error("cost_by_service_analysis_failed", error=str(e))
            return SkillResult(success=False, error=str(e))


# =============================================================================
# Budget Alert Skill
# =============================================================================

@register_skill(
    skill_id="finops_budget_alert",
    name="Budget Alert Analysis",
    description="Monitor budget thresholds and generate alerts for budget concerns",
    compatible_agents=["finops", "coordinator"],
    required_mcp_tools=["oci_cost_budget_status"],
    requires_code_execution=True,
    tags=["finops", "budget", "alerts", "monitoring"],
)
class BudgetAlertSkill(DeepSkill):
    """
    Monitor budgets and generate alerts.

    Steps:
    1. Get budget status for compartment
    2. Calculate burn rate and projections
    3. Identify at-risk budgets
    4. Generate alerts and recommendations
    """

    BUDGET_ANALYSIS_CODE = """
import json
from datetime import datetime

def analyze_budgets(data):
    budgets = data.get('budgets', data.get('items', []))

    if not budgets:
        return {
            'has_budgets': False,
            'summary': 'No budgets configured',
            'budgets': [],
            'alerts': [],
            'recommendations': ['Configure budgets to track spending']
        }

    analyzed_budgets = []
    alerts = []

    for budget in budgets:
        name = budget.get('display_name', budget.get('name', 'Unnamed'))
        amount = budget.get('amount', 0)
        actual_spend = budget.get('actual_spend', budget.get('spent', 0))
        forecasted = budget.get('forecasted_spend', actual_spend * 1.1)

        # Calculate utilization
        utilization = (actual_spend / amount * 100) if amount > 0 else 0
        forecast_utilization = (forecasted / amount * 100) if amount > 0 else 0

        # Determine status
        if utilization >= 100:
            status = 'EXCEEDED'
        elif utilization >= 90:
            status = 'CRITICAL'
        elif utilization >= 75:
            status = 'WARNING'
        elif forecast_utilization >= 100:
            status = 'AT_RISK'
        else:
            status = 'HEALTHY'

        budget_analysis = {
            'name': name,
            'amount': amount,
            'actual_spend': actual_spend,
            'forecasted_spend': forecasted,
            'utilization_pct': round(utilization, 2),
            'forecast_utilization_pct': round(forecast_utilization, 2),
            'status': status,
            'remaining': amount - actual_spend
        }
        analyzed_budgets.append(budget_analysis)

        # Generate alerts for concerning budgets
        if status in ['EXCEEDED', 'CRITICAL']:
            alerts.append({
                'severity': 'HIGH' if status == 'EXCEEDED' else 'MEDIUM',
                'budget': name,
                'message': f'Budget {name} is at {utilization:.1f}% utilization',
                'action': 'Review spending immediately' if status == 'EXCEEDED' else 'Monitor closely'
            })
        elif status == 'AT_RISK':
            alerts.append({
                'severity': 'LOW',
                'budget': name,
                'message': f'Budget {name} forecasted to exceed limit',
                'action': 'Review upcoming expenses'
            })

    # Sort by utilization (most concerning first)
    analyzed_budgets.sort(key=lambda x: x['utilization_pct'], reverse=True)

    # Summary statistics
    exceeded_count = len([b for b in analyzed_budgets if b['status'] == 'EXCEEDED'])
    at_risk_count = len([b for b in analyzed_budgets if b['status'] in ['CRITICAL', 'AT_RISK']])
    healthy_count = len([b for b in analyzed_budgets if b['status'] == 'HEALTHY'])

    # Generate recommendations
    recommendations = []
    if exceeded_count > 0:
        recommendations.append(f'{exceeded_count} budgets exceeded - immediate action required')
    if at_risk_count > 0:
        recommendations.append(f'{at_risk_count} budgets at risk - review and adjust spending')
    if all(b['utilization_pct'] < 50 for b in analyzed_budgets):
        recommendations.append('All budgets under 50% - consider tightening budget limits')

    return {
        'has_budgets': True,
        'summary': f'{len(budgets)} budgets, {exceeded_count} exceeded, {at_risk_count} at risk',
        'total_budgets': len(budgets),
        'exceeded_count': exceeded_count,
        'at_risk_count': at_risk_count,
        'healthy_count': healthy_count,
        'budgets': analyzed_budgets,
        'alerts': alerts,
        'recommendations': recommendations
    }

result = analyze_budgets(data)
return result
"""

    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute budget alert analysis."""
        try:
            compartment_id = context.parameters.get("compartment_id")
            include_children = context.parameters.get("include_children", False)

            if not compartment_id:
                return SkillResult(
                    success=False,
                    error="Required parameter: compartment_id"
                )

            # Get budget status
            budget_result = await self.call_mcp_tool(
                context,
                "oci_cost_budget_status",
                {
                    "compartment_id": compartment_id,
                    "include_children": include_children,
                    "response_format": "json"
                }
            )

            # Analyze with code execution
            analysis = await self.execute_code(
                context,
                self.BUDGET_ANALYSIS_CODE,
                variables={"data": budget_result}
            )

            if not analysis.success:
                return SkillResult(
                    success=True,
                    data={"raw_data": budget_result, "analysis_error": analysis.error}
                )

            return SkillResult(
                success=True,
                data=analysis.result,
                metadata={
                    "compartment_id": compartment_id,
                    "include_children": include_children
                }
            )

        except Exception as e:
            logger.error("budget_alert_analysis_failed", error=str(e))
            return SkillResult(success=False, error=str(e))

    async def self_test(self, context: SkillContext) -> SkillResult:
        """Test skill functionality."""
        try:
            result = await self.call_mcp_tool(
                context,
                "finops_list_providers",
                {}
            )
            success = "providers" in result or isinstance(result, list)
            return SkillResult(
                success=success,
                data={"test": "finops_providers", "result": result} if success else None,
                error=None if success else "FinOps providers test failed"
            )
        except Exception as e:
            return SkillResult(success=False, error=str(e), error_type="SelfTestError")
