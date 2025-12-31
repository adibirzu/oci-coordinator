"""
Evaluation metrics collection and reporting.

Provides aggregation and analysis of evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from src.evaluation.dataset import Difficulty, EvaluationDataset, TestCategory
from src.evaluation.judge import JudgmentResult, JudgmentScore

logger = structlog.get_logger(__name__)


@dataclass
class CategoryMetrics:
    """Metrics for a specific category."""

    category: str
    total: int = 0
    passed: int = 0
    partial: int = 0
    failed: int = 0
    skipped: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate (pass + partial)."""
        if self.total == 0:
            return 0.0
        return (self.passed + self.partial) / self.total

    @property
    def strict_pass_rate(self) -> float:
        """Calculate strict pass rate (only full passes)."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "category": self.category,
            "total": self.total,
            "passed": self.passed,
            "partial": self.partial,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": self.pass_rate,
            "strict_pass_rate": self.strict_pass_rate,
            "avg_score": self.avg_score,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class CriteriaMetrics:
    """Metrics for individual criteria across all cases."""

    intent_correct_rate: float = 0.0
    routing_correct_rate: float = 0.0
    domains_correct_rate: float = 0.0
    tools_correct_rate: float = 0.0
    response_relevant_rate: float = 0.0
    response_complete_rate: float = 0.0
    response_accurate_rate: float = 0.0
    no_hallucinations_rate: float = 0.0
    no_harmful_actions_rate: float = 0.0
    privacy_preserved_rate: float = 0.0
    latency_acceptable_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize to dictionary."""
        return {
            "intent_correct_rate": self.intent_correct_rate,
            "routing_correct_rate": self.routing_correct_rate,
            "domains_correct_rate": self.domains_correct_rate,
            "tools_correct_rate": self.tools_correct_rate,
            "response_relevant_rate": self.response_relevant_rate,
            "response_complete_rate": self.response_complete_rate,
            "response_accurate_rate": self.response_accurate_rate,
            "no_hallucinations_rate": self.no_hallucinations_rate,
            "no_harmful_actions_rate": self.no_harmful_actions_rate,
            "privacy_preserved_rate": self.privacy_preserved_rate,
            "latency_acceptable_rate": self.latency_acceptable_rate,
        }


@dataclass
class MetricsReport:
    """
    Complete metrics report for an evaluation run.

    Provides comprehensive statistics about evaluation results.
    """

    # Run metadata
    run_id: str
    dataset_name: str
    dataset_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Overall metrics
    total_cases: int = 0
    passed: int = 0
    partial: int = 0
    failed: int = 0
    skipped: int = 0

    # Score distributions
    avg_overall_score: float = 0.0
    avg_correctness_score: float = 0.0
    avg_quality_score: float = 0.0
    avg_safety_score: float = 0.0
    avg_efficiency_score: float = 0.0

    # Latency metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Breakdown by category
    by_category: dict[str, CategoryMetrics] = field(default_factory=dict)

    # Breakdown by difficulty
    by_difficulty: dict[str, CategoryMetrics] = field(default_factory=dict)

    # Breakdown by domain
    by_domain: dict[str, CategoryMetrics] = field(default_factory=dict)

    # Criteria metrics
    criteria_metrics: CriteriaMetrics = field(default_factory=CriteriaMetrics)

    # Target thresholds
    workflow_routing_target: float = 0.70  # 70%+ workflow routing
    task_success_target: float = 0.85  # 85%+ task success rate

    # Workflow routing ratio (actual)
    workflow_routing_ratio: float = 0.0

    @property
    def overall_pass_rate(self) -> float:
        """Calculate overall pass rate (pass + partial)."""
        if self.total_cases == 0:
            return 0.0
        return (self.passed + self.partial) / self.total_cases

    @property
    def strict_pass_rate(self) -> float:
        """Calculate strict pass rate."""
        if self.total_cases == 0:
            return 0.0
        return self.passed / self.total_cases

    @property
    def meets_workflow_target(self) -> bool:
        """Check if workflow routing target is met."""
        return self.workflow_routing_ratio >= self.workflow_routing_target

    @property
    def meets_success_target(self) -> bool:
        """Check if task success target is met."""
        return self.overall_pass_rate >= self.task_success_target

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "timestamp": self.timestamp.isoformat(),
            "total_cases": self.total_cases,
            "passed": self.passed,
            "partial": self.partial,
            "failed": self.failed,
            "skipped": self.skipped,
            "overall_pass_rate": self.overall_pass_rate,
            "strict_pass_rate": self.strict_pass_rate,
            "avg_overall_score": self.avg_overall_score,
            "avg_correctness_score": self.avg_correctness_score,
            "avg_quality_score": self.avg_quality_score,
            "avg_safety_score": self.avg_safety_score,
            "avg_efficiency_score": self.avg_efficiency_score,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "workflow_routing_ratio": self.workflow_routing_ratio,
            "meets_workflow_target": self.meets_workflow_target,
            "meets_success_target": self.meets_success_target,
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "by_difficulty": {k: v.to_dict() for k, v in self.by_difficulty.items()},
            "by_domain": {k: v.to_dict() for k, v in self.by_domain.items()},
            "criteria_metrics": self.criteria_metrics.to_dict(),
        }

    def format_markdown(self) -> str:
        """Format report as markdown."""
        lines = [
            f"# Evaluation Report: {self.run_id}",
            "",
            f"**Dataset**: {self.dataset_name} v{self.dataset_version}",
            f"**Timestamp**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Summary",
            "",
            f"| Metric | Value | Target |",
            f"|--------|-------|--------|",
            f"| Total Cases | {self.total_cases} | - |",
            f"| Pass Rate | {self.overall_pass_rate:.1%} | {self.task_success_target:.0%} |",
            f"| Workflow Routing | {self.workflow_routing_ratio:.1%} | {self.workflow_routing_target:.0%} |",
            f"| Avg Score | {self.avg_overall_score:.2f} | - |",
            "",
            "### Score Breakdown",
            "",
            f"| Score Type | Value |",
            f"|------------|-------|",
            f"| Passed | {self.passed} |",
            f"| Partial | {self.partial} |",
            f"| Failed | {self.failed} |",
            f"| Skipped | {self.skipped} |",
            "",
            "### Score Components",
            "",
            f"| Component | Score |",
            f"|-----------|-------|",
            f"| Correctness | {self.avg_correctness_score:.2f} |",
            f"| Quality | {self.avg_quality_score:.2f} |",
            f"| Safety | {self.avg_safety_score:.2f} |",
            f"| Efficiency | {self.avg_efficiency_score:.2f} |",
            "",
            "### Latency",
            "",
            f"| Percentile | Value |",
            f"|------------|-------|",
            f"| Average | {self.avg_latency_ms:.0f}ms |",
            f"| P50 | {self.p50_latency_ms:.0f}ms |",
            f"| P95 | {self.p95_latency_ms:.0f}ms |",
            f"| P99 | {self.p99_latency_ms:.0f}ms |",
            "",
        ]

        # Add category breakdown
        if self.by_category:
            lines.extend([
                "## By Category",
                "",
                "| Category | Total | Pass Rate | Avg Score |",
                "|----------|-------|-----------|-----------|",
            ])
            for cat, metrics in sorted(self.by_category.items()):
                lines.append(
                    f"| {cat} | {metrics.total} | {metrics.pass_rate:.1%} | {metrics.avg_score:.2f} |"
                )
            lines.append("")

        # Add domain breakdown
        if self.by_domain:
            lines.extend([
                "## By Domain",
                "",
                "| Domain | Total | Pass Rate | Avg Score |",
                "|--------|-------|-----------|-----------|",
            ])
            for domain, metrics in sorted(self.by_domain.items()):
                lines.append(
                    f"| {domain} | {metrics.total} | {metrics.pass_rate:.1%} | {metrics.avg_score:.2f} |"
                )
            lines.append("")

        # Add criteria metrics
        lines.extend([
            "## Criteria Pass Rates",
            "",
            "| Criterion | Pass Rate |",
            "|-----------|-----------|",
            f"| Intent Correct | {self.criteria_metrics.intent_correct_rate:.1%} |",
            f"| Routing Correct | {self.criteria_metrics.routing_correct_rate:.1%} |",
            f"| Domains Correct | {self.criteria_metrics.domains_correct_rate:.1%} |",
            f"| Response Relevant | {self.criteria_metrics.response_relevant_rate:.1%} |",
            f"| Response Accurate | {self.criteria_metrics.response_accurate_rate:.1%} |",
            f"| No Hallucinations | {self.criteria_metrics.no_hallucinations_rate:.1%} |",
            f"| Privacy Preserved | {self.criteria_metrics.privacy_preserved_rate:.1%} |",
            "",
        ])

        # Add targets status
        lines.extend([
            "## Targets",
            "",
            f"- Workflow Routing: {'✅' if self.meets_workflow_target else '❌'} "
            f"({self.workflow_routing_ratio:.1%} vs {self.workflow_routing_target:.0%} target)",
            f"- Task Success: {'✅' if self.meets_success_target else '❌'} "
            f"({self.overall_pass_rate:.1%} vs {self.task_success_target:.0%} target)",
            "",
        ])

        return "\n".join(lines)


class EvaluationMetrics:
    """
    Aggregates evaluation results into metrics.

    Collects judgment results and computes statistics.
    """

    def __init__(self, dataset: EvaluationDataset, run_id: str | None = None):
        """
        Initialize metrics aggregator.

        Args:
            dataset: The evaluation dataset
            run_id: Optional run identifier
        """
        self._dataset = dataset
        self._run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._results: list[JudgmentResult] = []
        self._latencies: list[float] = []
        self._workflow_count = 0
        self._total_routed = 0
        self._logger = logger.bind(component="EvaluationMetrics")

    def add_result(
        self,
        result: JudgmentResult,
        latency_ms: float | None = None,
        routing_type: str | None = None,
    ) -> None:
        """
        Add a judgment result.

        Args:
            result: The judgment result
            latency_ms: Optional latency in milliseconds
            routing_type: The routing type used (workflow, agent, etc.)
        """
        self._results.append(result)

        if latency_ms is not None:
            self._latencies.append(latency_ms)

        if routing_type:
            self._total_routed += 1
            if routing_type == "workflow":
                self._workflow_count += 1

    def compute_report(self) -> MetricsReport:
        """Compute the metrics report from collected results."""
        self._logger.info("Computing metrics report", result_count=len(self._results))

        report = MetricsReport(
            run_id=self._run_id,
            dataset_name=self._dataset.name,
            dataset_version=self._dataset.version,
            total_cases=len(self._results),
        )

        if not self._results:
            return report

        # Count score types
        for result in self._results:
            if result.score == JudgmentScore.PASS:
                report.passed += 1
            elif result.score == JudgmentScore.PARTIAL:
                report.partial += 1
            elif result.score == JudgmentScore.FAIL:
                report.failed += 1
            else:
                report.skipped += 1

        # Calculate averages
        scores = [r.criteria.overall_score() for r in self._results]
        report.avg_overall_score = sum(scores) / len(scores)

        correctness = [r.criteria.correctness_score() for r in self._results]
        report.avg_correctness_score = sum(correctness) / len(correctness)

        quality = [r.criteria.quality_score() for r in self._results]
        report.avg_quality_score = sum(quality) / len(quality)

        safety = [r.criteria.safety_score() for r in self._results]
        report.avg_safety_score = sum(safety) / len(safety)

        efficiency = [r.criteria.efficiency_score() for r in self._results]
        report.avg_efficiency_score = sum(efficiency) / len(efficiency)

        # Latency percentiles
        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            n = len(sorted_latencies)
            report.avg_latency_ms = sum(sorted_latencies) / n
            report.p50_latency_ms = sorted_latencies[int(n * 0.50)]
            report.p95_latency_ms = sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[0]
            report.p99_latency_ms = sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[0]

        # Workflow routing ratio
        if self._total_routed > 0:
            report.workflow_routing_ratio = self._workflow_count / self._total_routed

        # Compute criteria metrics
        report.criteria_metrics = self._compute_criteria_metrics()

        # Compute breakdowns
        report.by_category = self._compute_category_breakdown()
        report.by_difficulty = self._compute_difficulty_breakdown()
        report.by_domain = self._compute_domain_breakdown()

        return report

    def _compute_criteria_metrics(self) -> CriteriaMetrics:
        """Compute pass rates for each criterion."""
        n = len(self._results)
        if n == 0:
            return CriteriaMetrics()

        return CriteriaMetrics(
            intent_correct_rate=sum(1 for r in self._results if r.criteria.intent_correct) / n,
            routing_correct_rate=sum(1 for r in self._results if r.criteria.routing_correct) / n,
            domains_correct_rate=sum(1 for r in self._results if r.criteria.domains_correct) / n,
            tools_correct_rate=sum(1 for r in self._results if r.criteria.tools_correct) / n,
            response_relevant_rate=sum(1 for r in self._results if r.criteria.response_relevant) / n,
            response_complete_rate=sum(1 for r in self._results if r.criteria.response_complete) / n,
            response_accurate_rate=sum(1 for r in self._results if r.criteria.response_accurate) / n,
            no_hallucinations_rate=sum(1 for r in self._results if r.criteria.no_hallucinations) / n,
            no_harmful_actions_rate=sum(1 for r in self._results if r.criteria.no_harmful_actions) / n,
            privacy_preserved_rate=sum(1 for r in self._results if r.criteria.privacy_preserved) / n,
            latency_acceptable_rate=sum(1 for r in self._results if r.criteria.latency_acceptable) / n,
        )

    def _compute_category_breakdown(self) -> dict[str, CategoryMetrics]:
        """Compute metrics broken down by category."""
        breakdown: dict[str, CategoryMetrics] = {}

        for result in self._results:
            case = self._dataset.get_by_id(result.case_id)
            if not case:
                continue

            cat = case.category.value
            if cat not in breakdown:
                breakdown[cat] = CategoryMetrics(category=cat)

            metrics = breakdown[cat]
            metrics.total += 1

            if result.score == JudgmentScore.PASS:
                metrics.passed += 1
            elif result.score == JudgmentScore.PARTIAL:
                metrics.partial += 1
            elif result.score == JudgmentScore.FAIL:
                metrics.failed += 1
            else:
                metrics.skipped += 1

        # Calculate averages
        for cat, metrics in breakdown.items():
            cat_results = [
                r for r in self._results
                if self._dataset.get_by_id(r.case_id) and
                   self._dataset.get_by_id(r.case_id).category.value == cat
            ]
            if cat_results:
                metrics.avg_score = sum(r.criteria.overall_score() for r in cat_results) / len(cat_results)

        return breakdown

    def _compute_difficulty_breakdown(self) -> dict[str, CategoryMetrics]:
        """Compute metrics broken down by difficulty."""
        breakdown: dict[str, CategoryMetrics] = {}

        for result in self._results:
            case = self._dataset.get_by_id(result.case_id)
            if not case:
                continue

            diff = case.difficulty.value
            if diff not in breakdown:
                breakdown[diff] = CategoryMetrics(category=diff)

            metrics = breakdown[diff]
            metrics.total += 1

            if result.score == JudgmentScore.PASS:
                metrics.passed += 1
            elif result.score == JudgmentScore.PARTIAL:
                metrics.partial += 1
            elif result.score == JudgmentScore.FAIL:
                metrics.failed += 1
            else:
                metrics.skipped += 1

        return breakdown

    def _compute_domain_breakdown(self) -> dict[str, CategoryMetrics]:
        """Compute metrics broken down by domain."""
        breakdown: dict[str, CategoryMetrics] = {}

        for result in self._results:
            case = self._dataset.get_by_id(result.case_id)
            if not case:
                continue

            for domain in case.expected.domains:
                if domain not in breakdown:
                    breakdown[domain] = CategoryMetrics(category=domain)

                metrics = breakdown[domain]
                metrics.total += 1

                if result.score == JudgmentScore.PASS:
                    metrics.passed += 1
                elif result.score == JudgmentScore.PARTIAL:
                    metrics.partial += 1
                elif result.score == JudgmentScore.FAIL:
                    metrics.failed += 1
                else:
                    metrics.skipped += 1

        return breakdown
