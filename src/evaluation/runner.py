"""
Evaluation Runner.

Orchestrates the evaluation process by running queries through
the coordinator and evaluating results.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import structlog

from src.evaluation.dataset import (
    Difficulty,
    EvaluationCase,
    EvaluationDataset,
    TestCategory,
    load_dataset,
)
from src.evaluation.judge import JudgmentResult, LLMJudge
from src.evaluation.metrics import EvaluationMetrics, MetricsReport

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationResult:
    """
    Result from evaluating a single case.

    Contains both the coordinator's response and the judgment.
    """

    case: EvaluationCase
    actual_result: dict[str, Any]
    judgment: JudgmentResult
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "case_id": self.case.id,
            "query": self.case.query,
            "actual_result": self.actual_result,
            "judgment": self.judgment.to_dict(),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""

    # Dataset selection
    dataset_path: str | Path | None = None
    categories: list[TestCategory] | None = None
    difficulties: list[Difficulty] | None = None
    tags: list[str] | None = None
    max_cases: int | None = None

    # Execution settings
    concurrent: bool = False
    max_concurrent: int = 5
    timeout_seconds: int = 60

    # Judge settings
    judge_provider: str = "anthropic"
    skip_llm_judge: bool = False  # Skip LLM evaluation (for testing)

    # Output settings
    output_dir: str | Path = "evaluation_results"
    save_results: bool = True
    verbose: bool = False


class EvaluationRunner:
    """
    Runs evaluation against the OCI Coordinator.

    Orchestrates the full evaluation pipeline:
    1. Load evaluation dataset
    2. Run queries through coordinator
    3. Evaluate responses with LLM judge
    4. Aggregate metrics and generate report
    """

    def __init__(
        self,
        coordinator: Any = None,
        config: EvaluationConfig | None = None,
    ):
        """
        Initialize the evaluation runner.

        Args:
            coordinator: The coordinator instance to evaluate
            config: Evaluation configuration
        """
        self._coordinator = coordinator
        self._config = config or EvaluationConfig()
        self._judge = LLMJudge(
            provider=self._config.judge_provider,
            skip_llm=self._config.skip_llm_judge or (coordinator is None),
        )
        self._logger = logger.bind(component="EvaluationRunner")

    async def run(
        self,
        dataset: EvaluationDataset | None = None,
    ) -> MetricsReport:
        """
        Run the full evaluation.

        Args:
            dataset: Optional dataset (loads from config if not provided)

        Returns:
            MetricsReport with evaluation results
        """
        # Load dataset
        if dataset is None:
            if self._config.dataset_path:
                dataset = load_dataset(self._config.dataset_path)
            else:
                # Default to gold standard
                default_path = Path(__file__).parent / "datasets" / "gold_standard.yaml"
                dataset = load_dataset(default_path)

        self._logger.info(
            "Starting evaluation",
            dataset=dataset.name,
            total_cases=len(dataset.cases),
        )

        # Filter cases
        cases = self._filter_cases(dataset.cases)
        self._logger.info("Cases after filtering", count=len(cases))

        # Initialize metrics collector
        metrics = EvaluationMetrics(dataset)

        # Run evaluation
        results: list[EvaluationResult] = []

        if self._config.concurrent:
            results = await self._run_concurrent(cases)
        else:
            results = await self._run_sequential(cases)

        # Collect metrics
        for result in results:
            routing_type = result.actual_result.get("routing", {}).get("routing_type")
            metrics.add_result(
                result.judgment,
                latency_ms=result.latency_ms,
                routing_type=routing_type,
            )

        # Generate report
        report = metrics.compute_report()

        # Save results
        if self._config.save_results:
            await self._save_results(dataset, results, report)

        self._logger.info(
            "Evaluation complete",
            total=report.total_cases,
            passed=report.passed,
            pass_rate=f"{report.overall_pass_rate:.1%}",
        )

        return report

    def _filter_cases(self, cases: list[EvaluationCase]) -> list[EvaluationCase]:
        """Filter cases based on configuration."""
        filtered = cases

        # Filter by category
        if self._config.categories:
            filtered = [c for c in filtered if c.category in self._config.categories]

        # Filter by difficulty
        if self._config.difficulties:
            filtered = [c for c in filtered if c.difficulty in self._config.difficulties]

        # Filter by tags
        if self._config.tags:
            filtered = [
                c for c in filtered
                if any(t in c.tags for t in self._config.tags)
            ]

        # Limit cases
        if self._config.max_cases:
            filtered = filtered[: self._config.max_cases]

        return filtered

    async def _run_sequential(
        self,
        cases: list[EvaluationCase],
    ) -> list[EvaluationResult]:
        """Run cases sequentially."""
        results = []

        for i, case in enumerate(cases):
            self._logger.info(
                "Evaluating case",
                case_id=case.id,
                progress=f"{i + 1}/{len(cases)}",
            )

            result = await self._evaluate_case(case)
            results.append(result)

            if self._config.verbose:
                self._logger.info(
                    "Case result",
                    case_id=case.id,
                    score=result.judgment.score.value,
                    latency=f"{result.latency_ms:.0f}ms",
                )

        return results

    async def _run_concurrent(
        self,
        cases: list[EvaluationCase],
    ) -> list[EvaluationResult]:
        """Run cases concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        async def run_with_semaphore(case: EvaluationCase) -> EvaluationResult:
            async with semaphore:
                return await self._evaluate_case(case)

        tasks = [run_with_semaphore(case) for case in cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._logger.error(
                    "Case failed with exception",
                    case_id=cases[i].id,
                    error=str(result),
                )
                # Create a failed result
                final_results.append(
                    EvaluationResult(
                        case=cases[i],
                        actual_result={},
                        judgment=JudgmentResult(
                            case_id=cases[i].id,
                            score="fail",
                            criteria=None,
                            reasoning=f"Exception: {result}",
                        ),
                        latency_ms=0,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluate a single case."""
        start_time = time.time()
        actual_result: dict[str, Any] = {}
        error: str | None = None

        try:
            # Run query through coordinator
            if self._coordinator:
                result = await asyncio.wait_for(
                    self._invoke_coordinator(case),
                    timeout=self._config.timeout_seconds,
                )
                actual_result = result
            else:
                # Mock result for testing without coordinator
                actual_result = self._create_mock_result(case)

        except asyncio.TimeoutError:
            error = f"Timeout after {self._config.timeout_seconds}s"
            self._logger.warning("Case timed out", case_id=case.id)
        except Exception as e:
            error = str(e)
            self._logger.error("Case failed", case_id=case.id, error=error)

        latency_ms = (time.time() - start_time) * 1000
        actual_result["latency_ms"] = latency_ms

        # Judge the result
        judgment = await self._judge.judge(case, actual_result)

        return EvaluationResult(
            case=case,
            actual_result=actual_result,
            judgment=judgment,
            latency_ms=latency_ms,
            error=error,
        )

    async def _invoke_coordinator(self, case: EvaluationCase) -> dict[str, Any]:
        """Invoke the coordinator with a query."""
        # Build context from case
        context = case.context.copy() if case.context else {}

        # Invoke coordinator
        result = await self._coordinator.invoke(case.query, context=context)

        # Extract state for evaluation
        return {
            "query": case.query,
            "intent": result.get("intent", {}).to_dict() if result.get("intent") else {},
            "routing": result.get("routing", {}).to_dict() if result.get("routing") else {},
            "tool_calls": [tc.to_dict() for tc in result.get("tool_calls", [])],
            "final_response": result.get("final_response", ""),
            "current_agent": result.get("current_agent"),
            "workflow_name": result.get("workflow_name"),
            "iteration": result.get("iteration", 0),
        }

    def _create_mock_result(self, case: EvaluationCase) -> dict[str, Any]:
        """Create a mock result for testing without coordinator."""
        # This allows testing the evaluation pipeline without a running coordinator
        return {
            "query": case.query,
            "intent": {
                "intent": "mock_intent",
                "category": case.expected.intent_category or "query",
                "confidence": 0.75,
                "domains": case.expected.domains or [],
            },
            "routing": {
                "routing_type": case.expected.routing_type or "direct",
                "target": case.expected.routing_target,
                "confidence": 0.75,
                "reasoning": "Mock routing for evaluation testing",
            },
            "tool_calls": [
                {"id": f"mock_{i}", "name": tool, "arguments": {}}
                for i, tool in enumerate(case.expected.expected_tools or [])
            ],
            "final_response": f"Mock response for: {case.query}",
            "current_agent": case.expected.routing_target,
            "workflow_name": None,
            "iteration": 1,
        }

    async def _save_results(
        self,
        dataset: EvaluationDataset,
        results: list[EvaluationResult],
        report: MetricsReport,
    ) -> None:
        """Save evaluation results to files."""
        import json

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "run_id": report.run_id,
                    "dataset": dataset.name,
                    "timestamp": timestamp,
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
                default=str,
            )
        self._logger.info("Saved results", path=str(results_file))

        # Save metrics report
        report_file = output_dir / f"report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        self._logger.info("Saved report", path=str(report_file))

        # Save markdown report
        markdown_file = output_dir / f"report_{timestamp}.md"
        with open(markdown_file, "w") as f:
            f.write(report.format_markdown())
        self._logger.info("Saved markdown report", path=str(markdown_file))


async def run_evaluation(
    coordinator: Any = None,
    dataset_path: str | Path | None = None,
    **kwargs,
) -> MetricsReport:
    """
    Convenience function to run evaluation.

    Args:
        coordinator: The coordinator instance to evaluate
        dataset_path: Path to evaluation dataset
        **kwargs: Additional configuration options

    Returns:
        MetricsReport with evaluation results
    """
    config = EvaluationConfig(
        dataset_path=dataset_path,
        **kwargs,
    )
    runner = EvaluationRunner(coordinator=coordinator, config=config)
    return await runner.run()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OCI Coordinator evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to evaluation dataset YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Maximum number of cases to run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mock coordinator (for testing)",
    )

    args = parser.parse_args()

    config = EvaluationConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        verbose=args.verbose,
    )

    coordinator = None if args.mock else None  # Would load real coordinator here

    runner = EvaluationRunner(coordinator=coordinator, config=config)

    async def main():
        report = await runner.run()
        print("\n" + report.format_markdown())

    asyncio.run(main())
