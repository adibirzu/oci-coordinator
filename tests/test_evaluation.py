"""
Tests for the evaluation framework.
"""

import pytest
from pathlib import Path

from src.evaluation.dataset import (
    EvaluationCase,
    EvaluationDataset,
    ExpectedResult,
    TestCategory,
    Difficulty,
    load_dataset,
)
from src.evaluation.judge import JudgmentCriteria, JudgmentResult, JudgmentScore
from src.evaluation.metrics import EvaluationMetrics, MetricsReport
from src.evaluation.runner import EvaluationRunner, EvaluationConfig


class TestDataset:
    """Tests for evaluation dataset."""

    def test_load_gold_standard(self):
        """Test loading the gold standard dataset."""
        path = Path(__file__).parent.parent / "src/evaluation/datasets/gold_standard.yaml"
        dataset = load_dataset(path)

        assert dataset.name == "gold_standard"
        assert len(dataset.cases) >= 50  # We have 60 cases

    def test_filter_by_category(self):
        """Test filtering cases by category."""
        path = Path(__file__).parent.parent / "src/evaluation/datasets/gold_standard.yaml"
        dataset = load_dataset(path)

        db_cases = dataset.filter_by_category(TestCategory.DOMAIN_DATABASE)
        assert len(db_cases) == 10

        security_cases = dataset.filter_by_category(TestCategory.DOMAIN_SECURITY)
        assert len(security_cases) == 10

    def test_filter_by_difficulty(self):
        """Test filtering cases by difficulty."""
        path = Path(__file__).parent.parent / "src/evaluation/datasets/gold_standard.yaml"
        dataset = load_dataset(path)

        easy_cases = dataset.filter_by_difficulty(Difficulty.EASY)
        assert len(easy_cases) > 0

    def test_get_by_id(self):
        """Test getting a case by ID."""
        path = Path(__file__).parent.parent / "src/evaluation/datasets/gold_standard.yaml"
        dataset = load_dataset(path)

        case = dataset.get_by_id("db-001")
        assert case is not None
        assert "database" in case.expected.domains

    def test_statistics(self):
        """Test dataset statistics."""
        path = Path(__file__).parent.parent / "src/evaluation/datasets/gold_standard.yaml"
        dataset = load_dataset(path)

        stats = dataset.get_statistics()
        assert stats["total_cases"] >= 50
        assert "by_category" in stats
        assert "by_difficulty" in stats
        assert "by_domain" in stats


class TestJudgmentCriteria:
    """Tests for judgment criteria."""

    def test_correctness_score(self):
        """Test correctness score calculation."""
        criteria = JudgmentCriteria(
            intent_correct=True,
            routing_correct=True,
            domains_correct=True,
            tools_correct=False,
        )
        assert criteria.correctness_score() == 0.75

    def test_overall_score_all_pass(self):
        """Test overall score with all passing."""
        criteria = JudgmentCriteria(
            intent_correct=True,
            routing_correct=True,
            domains_correct=True,
            tools_correct=True,
            response_relevant=True,
            response_complete=True,
            response_accurate=True,
            no_hallucinations=True,
            no_harmful_actions=True,
            privacy_preserved=True,
            latency_acceptable=True,
            minimal_tool_calls=True,
        )
        assert criteria.overall_score() == 1.0

    def test_overall_score_weighted(self):
        """Test that overall score is properly weighted."""
        # Correctness: 40%, Quality: 25%, Safety: 25%, Efficiency: 10%
        criteria = JudgmentCriteria(
            intent_correct=True,
            routing_correct=True,
            domains_correct=True,
            tools_correct=True,
            response_relevant=False,
            response_complete=False,
            response_accurate=False,
            no_hallucinations=True,
            no_harmful_actions=True,
            privacy_preserved=True,
            latency_acceptable=True,
            minimal_tool_calls=True,
        )
        # Correctness: 1.0 * 0.40 = 0.40
        # Quality: 0.0 * 0.25 = 0.00
        # Safety: 1.0 * 0.25 = 0.25
        # Efficiency: 1.0 * 0.10 = 0.10
        # Total: 0.75
        assert criteria.overall_score() == pytest.approx(0.75)


class TestMetrics:
    """Tests for metrics collection."""

    def test_metrics_report_pass_rate(self):
        """Test pass rate calculation."""
        path = Path(__file__).parent.parent / "src/evaluation/datasets/gold_standard.yaml"
        dataset = load_dataset(path)
        metrics = EvaluationMetrics(dataset)

        # Add some results
        for i in range(10):
            score = JudgmentScore.PASS if i < 7 else JudgmentScore.FAIL
            result = JudgmentResult(
                case_id=f"test-{i}",
                score=score,
                criteria=JudgmentCriteria(),
                reasoning="Test",
            )
            metrics.add_result(result)

        report = metrics.compute_report()
        assert report.passed == 7
        assert report.failed == 3
        assert report.overall_pass_rate == 0.7

    def test_workflow_routing_ratio(self):
        """Test workflow routing ratio tracking."""
        path = Path(__file__).parent.parent / "src/evaluation/datasets/gold_standard.yaml"
        dataset = load_dataset(path)
        metrics = EvaluationMetrics(dataset)

        # Add results with routing types
        for i in range(10):
            result = JudgmentResult(
                case_id=f"test-{i}",
                score=JudgmentScore.PASS,
                criteria=JudgmentCriteria(),
                reasoning="Test",
            )
            routing_type = "workflow" if i < 7 else "agent"
            metrics.add_result(result, routing_type=routing_type)

        report = metrics.compute_report()
        assert report.workflow_routing_ratio == 0.7
        assert report.meets_workflow_target  # Target is 70%


class TestRunner:
    """Tests for evaluation runner."""

    @pytest.mark.asyncio
    async def test_run_with_mock(self):
        """Test running evaluation with mock coordinator."""
        config = EvaluationConfig(
            max_cases=5,
            save_results=False,
        )
        runner = EvaluationRunner(coordinator=None, config=config)

        report = await runner.run()

        assert report.total_cases == 5
        # Mock results should mostly pass (they match expected)
        assert report.overall_pass_rate >= 0.5

    @pytest.mark.asyncio
    async def test_filter_by_category(self):
        """Test filtering by category during run."""
        config = EvaluationConfig(
            categories=[TestCategory.DOMAIN_DATABASE],
            save_results=False,
        )
        runner = EvaluationRunner(coordinator=None, config=config)

        report = await runner.run()

        # Should only have database cases
        assert report.total_cases == 10
