"""
Parameter Corrector for Self-Healing Agents.

Automatically fixes tool parameters based on:
1. Error messages from failed calls
2. Tool schema requirements
3. LLM-guided corrections
4. Common OCI patterns
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = structlog.get_logger(__name__)


@dataclass
class CorrectionResult:
    """Result of parameter correction attempt."""

    original_params: dict[str, Any]
    corrected_params: dict[str, Any]
    changes_made: list[str]
    confidence: float = 0.0
    reasoning: str = ""
    corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "original_params": self.original_params,
            "corrected_params": self.corrected_params,
            "changes_made": self.changes_made,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "corrected": self.corrected,
        }


# OCI-specific parameter corrections
OCI_PARAM_CORRECTIONS = {
    # OCID format corrections
    "compartment_id": {
        "pattern": r"^ocid1\.compartment\.",
        "fix": "Ensure value starts with 'ocid1.compartment.'",
        "example": "ocid1.compartment.oc1..aaaaaaa...",
    },
    "database_id": {
        "pattern": r"^ocid1\.(autonomousdatabase|database)\.",
        "fix": "Ensure value is a valid database OCID",
        "example": "ocid1.autonomousdatabase.oc1..aaaaaaa...",
    },
    "instance_id": {
        "pattern": r"^ocid1\.instance\.",
        "fix": "Ensure value is a valid instance OCID",
        "example": "ocid1.instance.oc1..aaaaaaa...",
    },
    "tenancy_id": {
        "pattern": r"^ocid1\.tenancy\.",
        "fix": "Ensure value is a valid tenancy OCID",
        "example": "ocid1.tenancy.oc1..aaaaaaa...",
    },
    # Time format corrections
    "time_start": {
        "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        "fix": "Use ISO 8601 format: YYYY-MM-DDTHH:MM:SS",
        "transform": "iso8601",
    },
    "time_end": {
        "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        "fix": "Use ISO 8601 format: YYYY-MM-DDTHH:MM:SS",
        "transform": "iso8601",
    },
    # Numeric corrections
    "days": {
        "pattern": r"^\d+$",
        "fix": "Must be a positive integer",
        "transform": "int",
        "default": 30,
    },
    "limit": {
        "pattern": r"^\d+$",
        "fix": "Must be a positive integer",
        "transform": "int",
        "default": 100,
    },
}

# Tool-specific parameter requirements
TOOL_PARAM_REQUIREMENTS = {
    "oci_cost_get_summary": {
        "optional": ["compartment_id", "days", "service_filter"],
        "defaults": {"days": 30},
    },
    "oci_database_list_autonomous": {
        "optional": ["compartment_id", "limit"],
        "defaults": {"limit": 100},
    },
    "oci_opsi_search_databases": {
        "optional": ["search_query", "compartment_id"],
        "defaults": {},
    },
    "oci_compute_list_instances": {
        "optional": ["compartment_id", "limit"],
        "defaults": {"limit": 100},
    },
    "oci_logging_search_logs": {
        "required": ["search_query"],
        "optional": ["time_start", "time_end", "limit"],
        "defaults": {"limit": 100},
    },
    "oci_logan_execute_query": {
        "required": ["query"],
        "optional": ["time_start", "time_end", "namespace"],
        "defaults": {},
    },
}


class ParameterCorrector:
    """
    Corrects tool parameters using rules and LLM guidance.

    Features:
    - OCI-specific OCID validation and correction
    - Time format normalization
    - Type coercion
    - Missing parameter inference
    - LLM-guided complex corrections
    """

    def __init__(self, llm: "BaseChatModel | None" = None):
        """
        Initialize parameter corrector.

        Args:
            llm: LangChain LLM for complex corrections
        """
        self.llm = llm
        self._correction_history: list[CorrectionResult] = []
        self._logger = logger.bind(component="ParameterCorrector")

    async def correct(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        error_message: str | None = None,
        tool_schema: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> CorrectionResult:
        """
        Attempt to correct tool parameters.

        Args:
            tool_name: Name of the tool
            parameters: Current parameters
            error_message: Error from previous attempt (if any)
            tool_schema: Tool's JSON schema (if available)
            use_llm: Whether to use LLM for complex corrections

        Returns:
            CorrectionResult with corrected parameters
        """
        original = dict(parameters)
        corrected = dict(parameters)
        changes: list[str] = []

        # 1. Apply OCI-specific corrections
        corrected, oci_changes = self._apply_oci_corrections(corrected)
        changes.extend(oci_changes)

        # 2. Apply tool-specific defaults
        corrected, default_changes = self._apply_tool_defaults(tool_name, corrected)
        changes.extend(default_changes)

        # 3. Validate against schema
        if tool_schema:
            corrected, schema_changes = self._validate_against_schema(
                corrected, tool_schema
            )
            changes.extend(schema_changes)

        # 4. Parse error message for hints
        if error_message:
            corrected, error_changes = self._correct_from_error(
                corrected, error_message
            )
            changes.extend(error_changes)

        # 5. Use LLM for complex corrections
        if use_llm and self.llm and error_message:
            corrected, llm_changes = await self._correct_with_llm(
                tool_name, corrected, error_message, original
            )
            changes.extend(llm_changes)

        result = CorrectionResult(
            original_params=original,
            corrected_params=corrected,
            changes_made=changes,
            confidence=0.9 if changes else 1.0,
            reasoning=f"Applied {len(changes)} corrections" if changes else "No corrections needed",
            corrected=bool(changes),
        )

        self._correction_history.append(result)
        return result

    def _apply_oci_corrections(
        self, params: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Apply OCI-specific parameter corrections."""
        changes = []

        for param_name, value in list(params.items()):
            if value is None:
                continue

            correction_rule = OCI_PARAM_CORRECTIONS.get(param_name)
            if not correction_rule:
                continue

            # Check if value matches expected pattern
            pattern = correction_rule.get("pattern")
            if pattern and not re.match(pattern, str(value)):
                # Apply transform if available
                transform = correction_rule.get("transform")
                if transform == "int":
                    try:
                        params[param_name] = int(value)
                        changes.append(f"Converted {param_name} to int: {value} -> {params[param_name]}")
                    except (ValueError, TypeError):
                        default = correction_rule.get("default")
                        if default is not None:
                            params[param_name] = default
                            changes.append(f"Set {param_name} to default: {default}")

                elif transform == "iso8601":
                    # Try to parse and normalize time
                    normalized = self._normalize_time(value)
                    if normalized and normalized != str(value):
                        params[param_name] = normalized
                        changes.append(f"Normalized {param_name} time format")

        return params, changes

    def _normalize_time(self, value: Any) -> str | None:
        """Normalize time value to ISO 8601 format."""
        from datetime import datetime

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, str):
            # Already in ISO format
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
                return value

            # Try common formats
            formats = [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%m/%d/%Y",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue

        return None

    def _apply_tool_defaults(
        self, tool_name: str, params: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Apply tool-specific default values."""
        changes = []

        requirements = TOOL_PARAM_REQUIREMENTS.get(tool_name)
        if not requirements:
            return params, changes

        defaults = requirements.get("defaults", {})
        for param_name, default_value in defaults.items():
            if param_name not in params or params[param_name] is None:
                params[param_name] = default_value
                changes.append(f"Applied default for {param_name}: {default_value}")

        return params, changes

    def _validate_against_schema(
        self, params: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Validate and correct parameters against JSON schema."""
        changes = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for prop_name, prop_schema in properties.items():
            value = params.get(prop_name)
            expected_type = prop_schema.get("type")

            # Check required fields
            if prop_name in required and value is None:
                default = prop_schema.get("default")
                if default is not None:
                    params[prop_name] = default
                    changes.append(f"Set required {prop_name} to default: {default}")

            # Type coercion
            if value is not None and expected_type:
                if expected_type == "integer" and not isinstance(value, int):
                    try:
                        params[prop_name] = int(value)
                        changes.append(f"Coerced {prop_name} to integer")
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    try:
                        params[prop_name] = float(value)
                        changes.append(f"Coerced {prop_name} to number")
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "boolean" and not isinstance(value, bool):
                    params[prop_name] = str(value).lower() in ("true", "1", "yes")
                    changes.append(f"Coerced {prop_name} to boolean")
                elif expected_type == "string" and not isinstance(value, str):
                    params[prop_name] = str(value)
                    changes.append(f"Coerced {prop_name} to string")

        return params, changes

    def _correct_from_error(
        self, params: dict[str, Any], error_message: str
    ) -> tuple[dict[str, Any], list[str]]:
        """Extract correction hints from error message."""
        changes = []
        error_lower = error_message.lower()

        # Pattern: "parameter X must be Y"
        must_be_match = re.search(
            r"parameter\s+['\"]?(\w+)['\"]?\s+must\s+be\s+['\"]?(\w+)['\"]?",
            error_lower,
        )
        if must_be_match:
            param_name = must_be_match.group(1)
            expected = must_be_match.group(2)
            if param_name in params:
                # Try to coerce to expected type
                if expected in ("integer", "int", "number"):
                    try:
                        params[param_name] = int(params[param_name])
                        changes.append(f"Corrected {param_name} type from error hint")
                    except (ValueError, TypeError):
                        pass

        # Pattern: "invalid value for X: Y"
        invalid_match = re.search(
            r"invalid\s+value\s+for\s+['\"]?(\w+)['\"]?",
            error_lower,
        )
        if invalid_match:
            param_name = invalid_match.group(1)
            # Try to get default
            if param_name in OCI_PARAM_CORRECTIONS:
                default = OCI_PARAM_CORRECTIONS[param_name].get("default")
                if default is not None:
                    params[param_name] = default
                    changes.append(f"Reset {param_name} to default after error")

        # Pattern: missing required parameter
        missing_match = re.search(
            r"missing\s+required\s+parameter[:\s]+['\"]?(\w+)['\"]?",
            error_lower,
        )
        if missing_match:
            param_name = missing_match.group(1)
            if param_name not in params or params[param_name] is None:
                if param_name in OCI_PARAM_CORRECTIONS:
                    default = OCI_PARAM_CORRECTIONS[param_name].get("default")
                    if default is not None:
                        params[param_name] = default
                        changes.append(f"Added missing required {param_name}")

        return params, changes

    async def _correct_with_llm(
        self,
        tool_name: str,
        params: dict[str, Any],
        error_message: str,
        original_params: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Use LLM to correct parameters based on error."""
        changes = []

        prompt = f"""You are helping fix parameters for an OCI tool call that failed.

TOOL: {tool_name}

CURRENT PARAMETERS:
{json.dumps(params, indent=2, default=str)}

ERROR MESSAGE:
{error_message}

Analyze the error and suggest parameter corrections. Respond in JSON format:
{{
    "corrections": {{
        "param_name": "corrected_value"
    }},
    "reasoning": "Why these corrections should fix the error"
}}

Only include parameters that need to be changed. Consider:
- OCI OCID formats (ocid1.resource.oc1.region.unique_id)
- Time formats (ISO 8601)
- Required vs optional parameters
- Type requirements (string, integer, boolean)"""

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                result = json.loads(json_match.group())
                corrections = result.get("corrections", {})

                for param_name, new_value in corrections.items():
                    if param_name in params and params[param_name] != new_value:
                        params[param_name] = new_value
                        changes.append(f"LLM corrected {param_name}: {new_value}")

        except Exception as e:
            self._logger.warning("LLM correction failed", error=str(e))

        return params, changes

    def get_correction_statistics(self) -> dict[str, Any]:
        """Get statistics on corrections made."""
        total = len(self._correction_history)
        corrected = sum(1 for r in self._correction_history if r.corrected)

        # Count corrections by type
        correction_types: dict[str, int] = {}
        for result in self._correction_history:
            for change in result.changes_made:
                # Extract correction type from change message
                if "default" in change.lower():
                    correction_types["default"] = correction_types.get("default", 0) + 1
                elif "coerced" in change.lower() or "converted" in change.lower():
                    correction_types["type_coercion"] = correction_types.get("type_coercion", 0) + 1
                elif "llm" in change.lower():
                    correction_types["llm"] = correction_types.get("llm", 0) + 1
                else:
                    correction_types["other"] = correction_types.get("other", 0) + 1

        return {
            "total_attempts": total,
            "corrected_count": corrected,
            "correction_rate": corrected / total if total > 0 else 0,
            "by_type": correction_types,
        }
