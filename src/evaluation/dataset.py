"""
Evaluation dataset schema and loader.

Defines the structure for evaluation cases and provides
utilities for loading and managing evaluation datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)


class TestCategory(str, Enum):
    """Categories of test cases."""

    # Intent classification tests
    INTENT_CLASSIFICATION = "intent_classification"

    # Routing tests
    ROUTING_WORKFLOW = "routing_workflow"
    ROUTING_AGENT = "routing_agent"
    ROUTING_DIRECT = "routing_direct"
    ROUTING_ESCALATION = "routing_escalation"

    # Domain-specific tests
    DOMAIN_DATABASE = "domain_database"
    DOMAIN_SECURITY = "domain_security"
    DOMAIN_FINOPS = "domain_finops"
    DOMAIN_LOGS = "domain_logs"
    DOMAIN_INFRASTRUCTURE = "domain_infrastructure"

    # Multi-domain tests
    MULTI_DOMAIN = "multi_domain"

    # Edge cases
    EDGE_AMBIGUOUS = "edge_ambiguous"
    EDGE_MALFORMED = "edge_malformed"
    EDGE_ADVERSARIAL = "edge_adversarial"

    # End-to-end tests
    E2E_WORKFLOW = "e2e_workflow"
    E2E_AGENT = "e2e_agent"


class Difficulty(str, Enum):
    """Test case difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ExpectedResult:
    """Expected result for an evaluation case."""

    # Intent expectations
    intent_category: str | None = None  # Expected IntentCategory
    intent_name: str | None = None  # Expected specific intent

    # Routing expectations
    routing_type: str | None = None  # workflow, agent, direct, escalate
    routing_target: str | None = None  # Workflow name or agent role

    # Domain expectations
    domains: list[str] = field(default_factory=list)  # Expected domains

    # Response expectations
    response_contains: list[str] = field(default_factory=list)  # Keywords to check
    response_not_contains: list[str] = field(default_factory=list)  # Keywords to avoid

    # Tool expectations
    expected_tools: list[str] = field(default_factory=list)  # Tools that should be called

    # Quality thresholds
    min_confidence: float = 0.0  # Minimum confidence score
    max_latency_ms: int = 30000  # Maximum acceptable latency

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "intent_category": self.intent_category,
            "intent_name": self.intent_name,
            "routing_type": self.routing_type,
            "routing_target": self.routing_target,
            "domains": self.domains,
            "response_contains": self.response_contains,
            "response_not_contains": self.response_not_contains,
            "expected_tools": self.expected_tools,
            "min_confidence": self.min_confidence,
            "max_latency_ms": self.max_latency_ms,
        }


@dataclass
class EvaluationCase:
    """
    A single evaluation test case.

    Attributes:
        id: Unique identifier for the case
        query: The user query to test
        expected: Expected results
        category: Test category
        difficulty: Difficulty level
        tags: Additional tags for filtering
        description: Human-readable description
        context: Optional context (session state, etc.)
    """

    id: str
    query: str
    expected: ExpectedResult
    category: TestCategory
    difficulty: Difficulty = Difficulty.MEDIUM
    tags: list[str] = field(default_factory=list)
    description: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "expected": self.expected.to_dict(),
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "tags": self.tags,
            "description": self.description,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationCase:
        """Deserialize from dictionary."""
        expected_data = data.get("expected", {})
        expected = ExpectedResult(
            intent_category=expected_data.get("intent_category"),
            intent_name=expected_data.get("intent_name"),
            routing_type=expected_data.get("routing_type"),
            routing_target=expected_data.get("routing_target"),
            domains=expected_data.get("domains", []),
            response_contains=expected_data.get("response_contains", []),
            response_not_contains=expected_data.get("response_not_contains", []),
            expected_tools=expected_data.get("expected_tools", []),
            min_confidence=expected_data.get("min_confidence", 0.0),
            max_latency_ms=expected_data.get("max_latency_ms", 30000),
        )

        return cls(
            id=data["id"],
            query=data["query"],
            expected=expected,
            category=TestCategory(data.get("category", "intent_classification")),
            difficulty=Difficulty(data.get("difficulty", "medium")),
            tags=data.get("tags", []),
            description=data.get("description", ""),
            context=data.get("context", {}),
        )


@dataclass
class EvaluationDataset:
    """
    Collection of evaluation cases.

    Provides filtering and iteration over test cases.
    """

    name: str
    version: str
    cases: list[EvaluationCase]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of cases."""
        return len(self.cases)

    def __iter__(self):
        """Iterate over cases."""
        return iter(self.cases)

    def filter_by_category(self, category: TestCategory) -> list[EvaluationCase]:
        """Filter cases by category."""
        return [c for c in self.cases if c.category == category]

    def filter_by_difficulty(self, difficulty: Difficulty) -> list[EvaluationCase]:
        """Filter cases by difficulty."""
        return [c for c in self.cases if c.difficulty == difficulty]

    def filter_by_tags(self, tags: list[str]) -> list[EvaluationCase]:
        """Filter cases that have any of the specified tags."""
        return [c for c in self.cases if any(t in c.tags for t in tags)]

    def filter_by_domain(self, domain: str) -> list[EvaluationCase]:
        """Filter cases involving a specific domain."""
        return [c for c in self.cases if domain in c.expected.domains]

    def get_by_id(self, case_id: str) -> EvaluationCase | None:
        """Get a case by its ID."""
        for case in self.cases:
            if case.id == case_id:
                return case
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        stats: dict[str, Any] = {
            "total_cases": len(self.cases),
            "by_category": {},
            "by_difficulty": {},
            "by_domain": {},
        }

        for case in self.cases:
            # Count by category
            cat = case.category.value
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # Count by difficulty
            diff = case.difficulty.value
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

            # Count by domain
            for domain in case.expected.domains:
                stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1

        return stats

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata,
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationDataset:
        """Deserialize from dictionary."""
        cases = [EvaluationCase.from_dict(c) for c in data.get("cases", [])]
        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            cases=cases,
        )


def load_dataset(path: str | Path) -> EvaluationDataset:
    """
    Load an evaluation dataset from a YAML file.

    Args:
        path: Path to the YAML file

    Returns:
        Loaded EvaluationDataset
    """
    path = Path(path)
    logger.info("Loading evaluation dataset", path=str(path))

    with open(path) as f:
        data = yaml.safe_load(f)

    dataset = EvaluationDataset.from_dict(data)
    logger.info(
        "Dataset loaded",
        name=dataset.name,
        cases=len(dataset.cases),
    )

    return dataset


def save_dataset(dataset: EvaluationDataset, path: str | Path) -> None:
    """
    Save an evaluation dataset to a YAML file.

    Args:
        dataset: Dataset to save
        path: Path to save to
    """
    path = Path(path)
    logger.info("Saving evaluation dataset", path=str(path))

    with open(path, "w") as f:
        yaml.dump(dataset.to_dict(), f, default_flow_style=False, sort_keys=False)

    logger.info("Dataset saved", path=str(path))
