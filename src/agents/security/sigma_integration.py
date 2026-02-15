"""Sigma→OCL integration for the SecurityThreatAgent.

Wraps the ``oci-log-analytics-detections`` sibling project to provide:
- Sigma rule loading and querying
- Sigma→OCL (OCI Log Analytics Query Language) conversion
- MITRE ATT&CK technique-to-detection mapping
- Pre-built threat-hunting OCL queries

The sibling project path is resolved from the ``OCI_LOG_ANALYTICS_DETECTIONS_PATH``
environment variable (defaults to ``~/dev/oci-log-analytics-detections``).

Usage::

    from src.agents.security.sigma_integration import SigmaIntegration

    sigma = SigmaIntegration()
    if sigma.available:
        ocl = sigma.convert_rule_to_ocl("OCI IAM Policy Changed")
        rules = sigma.search_rules(mitre_technique="T1098")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import structlog
from opentelemetry import trace

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("oci-sigma-integration")

# Default location of sibling project
_DEFAULT_DETECTIONS_PATH = Path.home() / "dev" / "oci-log-analytics-detections"


class SigmaRule:
    """Lightweight representation of a parsed Sigma rule."""

    __slots__ = (
        "title", "sigma_id", "description", "level", "tags",
        "mitre_tactics", "mitre_techniques", "logsource",
        "detection", "falsepositives", "path",
    )

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        self.title: str = data.get("title", "")
        self.sigma_id: str = data.get("id", "")
        self.description: str = data.get("description", "")
        self.level: str = data.get("level", "informational")
        self.tags: list[str] = data.get("tags", [])
        self.logsource: dict[str, str] = data.get("logsource", {})
        self.detection: dict[str, Any] = data.get("detection", {})
        self.falsepositives: list[str] = data.get("falsepositives", [])
        self.path = path

        # Extract MITRE info from tags
        self.mitre_tactics: list[str] = []
        self.mitre_techniques: list[str] = []
        for tag in self.tags:
            if tag.startswith("attack.t"):
                self.mitre_techniques.append(tag.replace("attack.", "").upper())
            elif tag.startswith("attack."):
                self.mitre_tactics.append(tag.replace("attack.", ""))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "title": self.title,
            "sigma_id": self.sigma_id,
            "description": self.description,
            "level": self.level,
            "tags": self.tags,
            "mitre_tactics": self.mitre_tactics,
            "mitre_techniques": self.mitre_techniques,
            "logsource": self.logsource,
            "falsepositives": self.falsepositives,
        }


class SigmaIntegration:
    """Sigma→OCL integration layer for the security agent.

    Loads Sigma rules from the sibling ``oci-log-analytics-detections`` project
    and provides conversion and search capabilities.
    """

    def __init__(self, detections_path: str | Path | None = None) -> None:
        self._repo_root = Path(
            detections_path
            or os.getenv("OCI_LOG_ANALYTICS_DETECTIONS_PATH")
            or _DEFAULT_DETECTIONS_PATH
        )
        self._rules: list[SigmaRule] = []
        self._converted_queries: dict[str, dict[str, Any]] = {}
        self._available: bool | None = None

    # ── Availability ─────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Check if the sibling project is accessible."""
        if self._available is None:
            self._available = (
                self._repo_root.is_dir()
                and (self._repo_root / "rules").is_dir()
            )
            if self._available:
                logger.info(
                    "Sigma integration available",
                    path=str(self._repo_root),
                )
            else:
                logger.warning(
                    "Sigma integration unavailable — sibling project not found",
                    expected_path=str(self._repo_root),
                )
        return self._available

    # ── Rule Loading ─────────────────────────────────────────────────

    def load_rules(self) -> int:
        """Load all Sigma rules from the rules/ directory.

        Returns:
            Number of rules loaded.
        """
        if not self.available:
            return 0

        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed — cannot load Sigma rules")
            return 0

        with tracer.start_as_current_span("sigma.load_rules") as span:
            self._rules.clear()
            rules_dir = self._repo_root / "rules"

            for yaml_path in sorted(rules_dir.rglob("*.yaml")):
                try:
                    with open(yaml_path) as f:
                        data = yaml.safe_load(f)
                    if data and isinstance(data, dict) and "title" in data:
                        self._rules.append(SigmaRule(data, path=yaml_path))
                except Exception as e:
                    logger.debug("Failed to parse Sigma rule", path=str(yaml_path), error=str(e))

            span.set_attribute("sigma.rules_loaded", len(self._rules))
            logger.info("Sigma rules loaded", count=len(self._rules))
            return len(self._rules)

    def _load_converted_queries(self) -> None:
        """Load pre-converted OCL queries from queries/ directory."""
        queries_dir = self._repo_root / "queries"
        if not queries_dir.is_dir():
            return

        for json_path in queries_dir.glob("*.json"):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "query" in data:
                    key = data.get("title", json_path.stem)
                    self._converted_queries[key] = data
            except Exception:
                continue

    @property
    def rules(self) -> list[SigmaRule]:
        """Get loaded rules (auto-loads if needed)."""
        if not self._rules and self.available:
            self.load_rules()
        return self._rules

    # ── Conversion ───────────────────────────────────────────────────

    def convert_rule_to_ocl(self, rule_title: str) -> dict[str, Any] | None:
        """Convert a Sigma rule to OCL by title.

        Uses the sibling project's ``convert_sigma.py`` converter if available,
        otherwise falls back to pre-converted queries.

        Args:
            rule_title: Title or partial title of the Sigma rule.

        Returns:
            Conversion result dict with ``query``, ``mitre_attack``, ``level``,
            or None if not found.
        """
        with tracer.start_as_current_span("sigma.convert_to_ocl") as span:
            span.set_attribute("sigma.rule_title", rule_title)

            # Try pre-converted queries first
            if not self._converted_queries:
                self._load_converted_queries()

            # Exact match
            if rule_title in self._converted_queries:
                span.set_attribute("sigma.match_type", "exact")
                return self._converted_queries[rule_title]

            # Partial match
            title_lower = rule_title.lower()
            for title, query_data in self._converted_queries.items():
                if title_lower in title.lower():
                    span.set_attribute("sigma.match_type", "partial")
                    return query_data

            # Try live conversion via the converter script
            result = self._convert_live(rule_title)
            span.set_attribute("sigma.match_type", "live" if result else "not_found")
            return result

    def _convert_live(self, rule_title: str) -> dict[str, Any] | None:
        """Attempt live conversion using the sibling project's converter."""
        converter_path = self._repo_root / "scripts" / "convert_sigma.py"
        if not converter_path.exists():
            return None

        # Find matching rule
        rule = self._find_rule(rule_title)
        if not rule or not rule.path:
            return None

        try:
            # Add sibling project to path temporarily
            repo_str = str(self._repo_root)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

            # Import the converter module
            import importlib.util
            spec = importlib.util.spec_from_file_location("convert_sigma", str(converter_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "convert_rule"):
                    import yaml
                    with open(rule.path) as f:
                        rule_data = yaml.safe_load(f)

                    mapping_path = self._repo_root / "config" / "sigma_oci_mapping.yaml"
                    with open(mapping_path) as f:
                        mapping = yaml.safe_load(f)

                    result = module.convert_rule(rule_data, mapping)
                    return result
        except Exception as e:
            logger.debug("Live Sigma conversion failed", rule=rule_title, error=str(e))

        return None

    def convert_all_rules(self) -> list[dict[str, Any]]:
        """Convert all loaded rules to OCL.

        Returns:
            List of conversion result dicts.
        """
        results = []
        for rule in self.rules:
            result = self.convert_rule_to_ocl(rule.title)
            if result:
                results.append(result)
        return results

    # ── Search & Query ───────────────────────────────────────────────

    def _find_rule(self, title: str) -> SigmaRule | None:
        """Find a rule by exact or partial title match."""
        title_lower = title.lower()
        for rule in self.rules:
            if rule.title.lower() == title_lower:
                return rule
        for rule in self.rules:
            if title_lower in rule.title.lower():
                return rule
        return None

    def search_rules(
        self,
        *,
        mitre_technique: str | None = None,
        mitre_tactic: str | None = None,
        level: str | None = None,
        platform: str | None = None,
        keyword: str | None = None,
    ) -> list[SigmaRule]:
        """Search Sigma rules by various criteria.

        Args:
            mitre_technique: MITRE ATT&CK technique ID (e.g., "T1098").
            mitre_tactic: MITRE tactic name (e.g., "persistence").
            level: Severity level (critical, high, medium, low, informational).
            platform: Target platform (oci, linux, windows).
            keyword: Keyword search in title and description.

        Returns:
            Matching Sigma rules.
        """
        results = self.rules
        if mitre_technique:
            tech = mitre_technique.upper()
            results = [r for r in results if tech in r.mitre_techniques]
        if mitre_tactic:
            tactic = mitre_tactic.lower()
            results = [r for r in results if tactic in r.mitre_tactics]
        if level:
            results = [r for r in results if r.level == level.lower()]
        if platform:
            results = [r for r in results if r.logsource.get("product", "").lower() == platform.lower()]
        if keyword:
            kw = keyword.lower()
            results = [
                r for r in results
                if kw in r.title.lower() or kw in r.description.lower()
            ]
        return results

    def get_mitre_coverage(self) -> dict[str, list[str]]:
        """Get MITRE ATT&CK coverage summary.

        Returns:
            Dict mapping tactic → list of technique IDs covered.
        """
        coverage: dict[str, list[str]] = {}
        for rule in self.rules:
            for tactic in rule.mitre_tactics:
                if tactic not in coverage:
                    coverage[tactic] = []
                for tech in rule.mitre_techniques:
                    if tech not in coverage[tactic]:
                        coverage[tactic].append(tech)
        return coverage

    def get_threat_hunting_queries(
        self,
        platform: str = "oci",
        min_level: str = "medium",
    ) -> list[dict[str, Any]]:
        """Get pre-built threat hunting OCL queries for a platform.

        Args:
            platform: Target platform (oci, linux, windows).
            min_level: Minimum severity level.

        Returns:
            List of query dicts with ``title``, ``query``, ``level``, ``mitre_attack``.
        """
        severity_order = {
            "informational": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
        }
        min_severity = severity_order.get(min_level.lower(), 2)

        if not self._converted_queries:
            self._load_converted_queries()

        results = []
        for title, query_data in self._converted_queries.items():
            logsource = query_data.get("logsource", {})
            rule_platform = logsource.get("product", "").lower()
            rule_level = query_data.get("level", "informational").lower()

            if rule_platform == platform.lower() and severity_order.get(rule_level, 0) >= min_severity:
                results.append({
                    "title": title,
                    "query": query_data.get("query", ""),
                    "level": rule_level,
                    "mitre_attack": query_data.get("mitre_attack", {}),
                    "description": query_data.get("description", ""),
                })

        return sorted(results, key=lambda x: severity_order.get(x["level"], 0), reverse=True)

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics about loaded rules.

        Returns:
            Dict with rule counts by platform, severity, and MITRE coverage.
        """
        rules = self.rules
        by_platform: dict[str, int] = {}
        by_level: dict[str, int] = {}
        techniques: set[str] = set()

        for rule in rules:
            platform = rule.logsource.get("product", "unknown")
            by_platform[platform] = by_platform.get(platform, 0) + 1
            by_level[rule.level] = by_level.get(rule.level, 0) + 1
            techniques.update(rule.mitre_techniques)

        return {
            "total_rules": len(rules),
            "by_platform": by_platform,
            "by_level": by_level,
            "mitre_techniques_covered": len(techniques),
            "mitre_coverage": self.get_mitre_coverage(),
        }
