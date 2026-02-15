"""FOCUS v1.3 cost normalization integration for the FinOpsAgent.

Wraps the ``multicloudfocusreports`` sibling project to provide:
- FOCUS v1.3 schema access (50+ baseline columns + OCI extensions)
- OCI cost normalization (CSV → FOCUS-compliant NDJSON)
- Optimization field computation (savings, idle cost, unit cost)
- Multicloud cost comparison via normalized schema

The sibling project path is resolved from the ``MULTICLOUDFOCUSREPORTS_PATH``
environment variable (defaults to ``~/dev/multicloudfocusreports``).

Usage::

    from src.agents.finops.focus_integration import FocusIntegration

    focus = FocusIntegration()
    if focus.available:
        columns = focus.get_focus_columns()
        normalized = focus.normalize_cost_data(raw_data)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import structlog
from opentelemetry import trace

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("oci-focus-integration")

_DEFAULT_FOCUS_PATH = Path.home() / "dev" / "multicloudfocusreports"

# FOCUS v1.3 core cost columns — subset most relevant for FinOps agent queries.
# Full schema has 50+ columns; these are the ones the agent uses most often.
FOCUS_COST_COLUMNS = [
    "Billed Cost",
    "Effective Cost",
    "List Cost",
    "Contracted Cost",
    "Service Name",
    "Service Category",
    "Provider Name",
    "Region Name",
    "Resource Id",
    "Resource Name",
    "Resource Type",
    "Billing Account Name",
    "Sub Account Name",
    "Billing Period Start",
    "Billing Period End",
    "Billing Currency",
    "Charge Category",
    "Charge Description",
    "Consumed Quantity",
    "Consumed Unit",
    "Pricing Quantity",
    "Pricing Unit",
    "List Unit Price",
    "Pricing Category",
    "Commitment Discount Type",
    "Tags",
]

# OCI-specific FOCUS extension fields
OCI_EXTENSION_COLUMNS = [
    "oci_Attributed Cost",
    "oci_Cost Overage",
    "oci_Compartment Id",
    "oci_Compartment Name",
    "oci_Overage Flag",
    "oci_Unit Price Overage",
    "oci_Billed Quantity Overage",
    "oci_Attributed Usage",
    "oci_Reference Number",
    "oci_Back Reference Number",
]

# Computed optimization extension fields
OPTIMIZATION_COLUMNS = [
    "x_Opt Total Savings",
    "x_Opt Commitment Discount Savings",
    "x_Opt Negotiated Discount Savings",
    "x_Opt Effective Savings Rate",
    "x_Opt Is Idle",
    "x_Opt Idle Cost",
    "x_Opt Unit Cost",
    "x_Opt Pricing Category Group",
    "x_Opt Has Tags",
]


def _safe_float(value: Any) -> float | None:
    """Parse a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return None


def compute_optimization_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Compute FOCUS optimization extension fields for a cost row.

    These fields provide derived metrics useful for cost analysis:
    savings, idle detection, unit cost, and pricing category grouping.

    Args:
        row: Cost data row with FOCUS column names.

    Returns:
        Dict of computed ``x_Opt_*`` fields.
    """
    list_cost = _safe_float(row.get("List Cost"))
    effective_cost = _safe_float(row.get("Effective Cost"))
    contracted_cost = _safe_float(row.get("Contracted Cost"))
    consumed_qty = _safe_float(row.get("Consumed Quantity"))
    pricing_category = str(row.get("Pricing Category", "")).lower()
    commitment_type = row.get("Commitment Discount Type")
    tags = row.get("Tags")

    result: dict[str, Any] = {}

    # Total savings
    if list_cost is not None and effective_cost is not None:
        result["x_Opt Total Savings"] = list_cost - effective_cost
    else:
        result["x_Opt Total Savings"] = None

    # Commitment discount savings
    if commitment_type and list_cost is not None and effective_cost is not None:
        result["x_Opt Commitment Discount Savings"] = list_cost - effective_cost
    else:
        result["x_Opt Commitment Discount Savings"] = None

    # Negotiated discount savings
    if list_cost is not None and contracted_cost is not None:
        result["x_Opt Negotiated Discount Savings"] = list_cost - contracted_cost
    else:
        result["x_Opt Negotiated Discount Savings"] = None

    # Effective savings rate
    if list_cost and result.get("x_Opt Total Savings") is not None:
        result["x_Opt Effective Savings Rate"] = result["x_Opt Total Savings"] / list_cost
    else:
        result["x_Opt Effective Savings Rate"] = None

    # Idle detection
    is_idle = (
        consumed_qty is not None
        and consumed_qty < 0.01
        and effective_cost is not None
        and effective_cost > 0
    )
    result["x_Opt Is Idle"] = str(is_idle)
    result["x_Opt Idle Cost"] = effective_cost if is_idle else 0.0

    # Unit cost
    if effective_cost is not None and consumed_qty and consumed_qty > 0:
        result["x_Opt Unit Cost"] = effective_cost / consumed_qty
    else:
        result["x_Opt Unit Cost"] = None

    # Pricing category grouping
    if "commit" in pricing_category or "reserved" in pricing_category:
        result["x_Opt Pricing Category Group"] = "Committed"
    elif "spot" in pricing_category:
        result["x_Opt Pricing Category Group"] = "Spot"
    elif "on-demand" in pricing_category or "on demand" in pricing_category:
        result["x_Opt Pricing Category Group"] = "On-Demand"
    else:
        result["x_Opt Pricing Category Group"] = "Other"

    # Tag presence
    result["x_Opt Has Tags"] = str(bool(tags))

    return result


class FocusIntegration:
    """FOCUS v1.3 integration layer for the FinOps agent.

    Provides cost normalization, schema access, and optimization field
    computation using the ``multicloudfocusreports`` sibling project.
    """

    def __init__(self, focus_path: str | Path | None = None) -> None:
        self._repo_root = Path(
            focus_path
            or os.getenv("MULTICLOUDFOCUSREPORTS_PATH")
            or _DEFAULT_FOCUS_PATH
        )
        self._available: bool | None = None
        self._schema_module: Any = None
        self._normalizer_module: Any = None

    # ── Availability ─────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Check if the sibling project is accessible."""
        if self._available is None:
            self._available = (
                self._repo_root.is_dir()
                and (self._repo_root / "src" / "focus_pipeline").is_dir()
            )
            if self._available:
                logger.info("FOCUS integration available", path=str(self._repo_root))
            else:
                logger.warning(
                    "FOCUS integration unavailable — sibling project not found",
                    expected_path=str(self._repo_root),
                )
        return self._available

    def _ensure_path(self) -> bool:
        """Ensure the sibling project is on sys.path."""
        if not self.available:
            return False
        repo_str = str(self._repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        return True

    # ── Schema Access ────────────────────────────────────────────────

    def get_focus_columns(
        self,
        version: str = "1.3",
        include_oci_extensions: bool = True,
        include_optimization: bool = True,
    ) -> list[str]:
        """Get FOCUS column names for a given version.

        Attempts to use the sibling project's ``FocusSchema`` for the full
        column set. Falls back to the built-in ``FOCUS_COST_COLUMNS`` if
        the project is unavailable.

        Args:
            version: FOCUS schema version (default "1.3").
            include_oci_extensions: Include OCI-specific extension fields.
            include_optimization: Include computed optimization fields.

        Returns:
            List of column name strings.
        """
        columns: list[str] = []

        if self._ensure_path():
            try:
                from src.focus_pipeline.core.schema import FocusSchema
                schema = FocusSchema(self._repo_root)
                if include_oci_extensions:
                    columns = schema.all_columns_with_extensions(version, csps=["oci"])
                else:
                    columns = schema.columns_for(version)
            except Exception as e:
                logger.debug("FocusSchema import failed, using built-in columns", error=str(e))

        if not columns:
            columns = list(FOCUS_COST_COLUMNS)
            if include_oci_extensions:
                columns.extend(OCI_EXTENSION_COLUMNS)

        if include_optimization:
            columns.extend(OPTIMIZATION_COLUMNS)

        return columns

    # ── Normalization ────────────────────────────────────────────────

    def normalize_cost_row(
        self,
        raw_row: dict[str, Any],
        include_optimization: bool = True,
    ) -> dict[str, Any]:
        """Normalize a single OCI cost data row to FOCUS v1.3 schema.

        This performs header canonicalization (mapping OCI Usage API field
        names to FOCUS column names) and optionally computes optimization
        fields.

        Args:
            raw_row: Raw cost data dict from OCI Usage API.
            include_optimization: Compute ``x_Opt_*`` fields.

        Returns:
            Normalized dict with FOCUS column names.
        """
        # OCI Usage API → FOCUS column name mapping
        field_map = {
            "cost": "Billed Cost",
            "billedcost": "Billed Cost",
            "effectivecost": "Effective Cost",
            "listcost": "List Cost",
            "contractedcost": "Contracted Cost",
            "service": "Service Name",
            "servicename": "Service Name",
            "servicecategory": "Service Category",
            "region": "Region Name",
            "regionname": "Region Name",
            "resourceid": "Resource Id",
            "resourcename": "Resource Name",
            "resourcetype": "Resource Type",
            "tenancyname": "Billing Account Name",
            "compartmentname": "Sub Account Name",
            "compartmentid": "oci_Compartment Id",
            "timeusagestarted": "Billing Period Start",
            "timeusageended": "Billing Period End",
            "currency": "Billing Currency",
            "quantity": "Consumed Quantity",
            "unit": "Consumed Unit",
            "unitprice": "List Unit Price",
            "skuid": "Sku Id",
            "tags": "Tags",
            "attributedcost": "oci_Attributed Cost",
            "costoverage": "oci_Cost Overage",
            "overageflag": "oci_Overage Flag",
        }

        normalized: dict[str, Any] = {}
        for key, value in raw_row.items():
            # Canonicalize: lowercase, strip non-alnum
            canon = "".join(c for c in key.lower() if c.isalnum())
            focus_name = field_map.get(canon)
            if focus_name:
                normalized[focus_name] = value
            else:
                # Keep unmapped fields as-is
                normalized[key] = value

        # Set provider
        normalized.setdefault("Provider Name", "Oracle")

        if include_optimization:
            normalized.update(compute_optimization_fields(normalized))

        return normalized

    def normalize_cost_data(
        self,
        raw_rows: list[dict[str, Any]],
        include_optimization: bool = True,
    ) -> list[dict[str, Any]]:
        """Normalize a list of OCI cost data rows to FOCUS v1.3 schema.

        Args:
            raw_rows: List of raw cost data dicts.
            include_optimization: Compute ``x_Opt_*`` fields.

        Returns:
            List of normalized dicts.
        """
        with tracer.start_as_current_span("focus.normalize_cost_data") as span:
            span.set_attribute("focus.row_count", len(raw_rows))
            span.set_attribute("focus.include_optimization", include_optimization)
            result = [
                self.normalize_cost_row(row, include_optimization=include_optimization)
                for row in raw_rows
            ]
            span.set_attribute("focus.normalized_count", len(result))
            return result

    # ── Analysis Helpers ─────────────────────────────────────────────

    def compute_savings_summary(
        self,
        normalized_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute savings summary from FOCUS-normalized cost data.

        Args:
            normalized_rows: FOCUS-normalized cost rows (with optimization fields).

        Returns:
            Summary dict with total costs, savings, idle resources, and
            pricing category breakdown.
        """
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("focus.savings_analysis_rows", len(normalized_rows))

        total_billed = 0.0
        total_effective = 0.0
        total_list = 0.0
        total_savings = 0.0
        total_idle_cost = 0.0
        idle_count = 0
        by_category: dict[str, float] = {}

        for row in normalized_rows:
            billed = _safe_float(row.get("Billed Cost")) or 0.0
            effective = _safe_float(row.get("Effective Cost")) or 0.0
            list_cost = _safe_float(row.get("List Cost")) or 0.0
            savings = _safe_float(row.get("x_Opt Total Savings")) or 0.0
            idle_cost = _safe_float(row.get("x_Opt Idle Cost")) or 0.0
            is_idle = row.get("x_Opt Is Idle") == "True"
            category = row.get("x_Opt Pricing Category Group", "Other")

            total_billed += billed
            total_effective += effective
            total_list += list_cost
            total_savings += savings
            total_idle_cost += idle_cost
            if is_idle:
                idle_count += 1
            by_category[category] = by_category.get(category, 0.0) + effective

        savings_rate = (total_savings / total_list) if total_list > 0 else 0.0

        return {
            "total_billed_cost": round(total_billed, 2),
            "total_effective_cost": round(total_effective, 2),
            "total_list_cost": round(total_list, 2),
            "total_savings": round(total_savings, 2),
            "effective_savings_rate": round(savings_rate, 4),
            "total_idle_cost": round(total_idle_cost, 2),
            "idle_resource_count": idle_count,
            "cost_by_pricing_category": {k: round(v, 2) for k, v in by_category.items()},
            "row_count": len(normalized_rows),
        }

    def compare_providers(
        self,
        provider_data: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Compare costs across cloud providers using FOCUS normalization.

        Args:
            provider_data: Dict mapping provider name to their FOCUS-normalized rows.

        Returns:
            Comparison summary with per-provider totals.
        """
        comparison: dict[str, Any] = {}
        for provider, rows in provider_data.items():
            comparison[provider] = self.compute_savings_summary(rows)

        return {
            "providers": comparison,
            "provider_count": len(provider_data),
        }
