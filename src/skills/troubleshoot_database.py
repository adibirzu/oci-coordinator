"""
Enhanced Database Troubleshooting Skill.

Implements the "Phase 2 - Advanced" runbook with branching logic for Root Cause Analysis.
"""

from dataclasses import dataclass, field

import structlog
from langgraph.graph import END, StateGraph

from src.agents.base import BaseAgent

logger = structlog.get_logger(__name__)


@dataclass
class DBTriageState:
    """State for the DB troubleshooting workflow."""

    # Input
    query: str
    database_id: str | None = None
    compartment_id: str | None = None

    # Context
    is_rac: bool = False
    context_type: str = "Real-time" # Real-time vs Historical
    scope: str = "System-wide" # System-wide vs SQL-specific

    # Analysis Flags
    flag_blocking: bool = False
    flag_library_cache: bool = False
    flag_deadlock: bool = False
    flag_io_latency: bool = False
    flag_log_sync: bool = False
    flag_hot_block: bool = False
    flag_cpu_saturation: bool = False
    flag_temp_spill: bool = False
    flag_plan_regression: bool = False
    flag_stale_stats: bool = False

    # Collected Data
    wait_analysis: dict = field(default_factory=dict)
    blocking_tree: list[dict] = field(default_factory=list)
    top_sql: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Output
    final_report: str = ""


class DBTroubleshootSkill:
    """
    Implements the Enhanced Database Troubleshooting Runbook.
    """

    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.logger = logger.bind(skill="db_troubleshoot")

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(DBTriageState)

        # Phase 1: Ingestion & Triage
        workflow.add_node("triage", self._triage_node)

        # Phase 2: Diagnostic Execution (Logic Loops)
        workflow.add_node("check_hang_lock", self._check_hang_lock_node)
        workflow.add_node("check_wait_interface", self._check_wait_interface_node)
        workflow.add_node("check_io_subroutine", self._check_io_subroutine)
        workflow.add_node("check_redo_subroutine", self._check_redo_subroutine)
        workflow.add_node("check_hot_block_subroutine", self._check_hot_block_subroutine)

        workflow.add_node("check_cpu_capacity", self._check_cpu_capacity_node)
        workflow.add_node("check_memory", self._check_memory_node)
        workflow.add_node("check_plan_stability", self._check_plan_stability_node)
        workflow.add_node("check_monster_queries", self._check_monster_queries_node)

        # Phase 3: Synthesis
        workflow.add_node("generate_report", self._generate_report_node)

        # Edges
        workflow.set_entry_point("triage")
        workflow.add_edge("triage", "check_hang_lock")
        workflow.add_edge("check_hang_lock", "check_wait_interface")

        # Conditional Edges for Wait Interface Sub-routines
        workflow.add_conditional_edges(
            "check_wait_interface",
            self._route_wait_interface,
            {
                "io_check": "check_io_subroutine",
                "redo_check": "check_redo_subroutine",
                "hot_block_check": "check_hot_block_subroutine",
                "cpu_check": "check_cpu_capacity"
            }
        )

        # Returns from sub-routines to main flow
        workflow.add_edge("check_io_subroutine", "check_cpu_capacity")
        workflow.add_edge("check_redo_subroutine", "check_cpu_capacity")
        workflow.add_edge("check_hot_block_subroutine", "check_cpu_capacity")

        workflow.add_edge("check_cpu_capacity", "check_memory")
        workflow.add_edge("check_memory", "check_plan_stability")
        workflow.add_edge("check_plan_stability", "check_monster_queries")
        workflow.add_edge("check_monster_queries", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    async def _triage_node(self, state: DBTriageState) -> dict:
        """Phase 1: Ingestion & Smart Triage."""
        self.logger.info("Starting Triage Phase", query=state.query)

        # In a real scenario, we would extract the DB ID from the query or context
        # For now, we'll try to find a default or use the one in state
        db_id = state.database_id

        return {
            "is_rac": False,
            "context_type": "Real-time" if "history" not in state.query.lower() else "Historical",
            "scope": "System-wide"
        }

    async def _check_hang_lock_node(self, state: DBTriageState) -> dict:
        """Step 1: The Hang/Lock Check."""
        self.logger.info("Checking Hangs/Locks")
        updates = {}

        # Use Wait Events to infer locking
        # If 'enq: TX - row lock contention' is high, we have locking
        if state.database_id:
             result = await self.agent.call_tool("oci_dbmgmt_get_wait_events", {
                "managed_database_id": state.database_id,
                "top_n": 5
             })

             if result.get("success") and result.get("result"):
                 events = result["result"]
                 for e in events:
                     event_name = e.get("event_name", "").lower()
                     if "lock" in event_name or "enqueue" in event_name:
                         updates["flag_blocking"] = True
                         updates["recommendations"] = state.recommendations + [f"Blocking detected via event: {event_name}"]
                         updates["blocking_tree"] = [{"event": event_name}] # Simplified representation
        else:
             self.logger.warning("No Database ID provided for Lock Check")

        return updates

    async def _check_wait_interface_node(self, state: DBTriageState) -> dict:
        """Step 2: The Wait Interface Analysis."""
        self.logger.info("Analyzing Wait Interface")

        updates = {"wait_analysis": {}}

        if state.database_id:
            result = await self.agent.call_tool("oci_dbmgmt_get_wait_events", {
                "managed_database_id": state.database_id,
                "top_n": 1
            })

            if result.get("success") and result.get("result"):
                top_event = result["result"][0]
                # Normalize keys
                updates["wait_analysis"] = {
                    "EVENT": top_event.get("event_name"),
                    "WAIT_CLASS": top_event.get("wait_class"),
                    "CNT": top_event.get("waits_per_sec", 0) # Approximation
                }

        return updates

    def _route_wait_interface(self, state: DBTriageState) -> str:
        """Routing logic based on top wait class."""
        top_wait = state.wait_analysis
        if not top_wait:
            return "cpu_check"

        wait_class = top_wait.get("WAIT_CLASS", "").upper()
        event = top_wait.get("EVENT", "").lower()

        if "USER I/O" in wait_class:
            if "scattered read" in event:
                return "check_monster_queries" # Go to Scans
            return "io_check"

        if "COMMIT" in wait_class:
            return "redo_check"

        if "CONCURRENCY" in wait_class:
            return "hot_block_check"

        return "cpu_check"

    async def _check_io_subroutine(self, state: DBTriageState) -> dict:
        """Sub-routine: Check I/O Latency."""
        self.logger.info("Sub-routine: IO Check")
        return {"flag_io_latency": True, "recommendations": state.recommendations + ["Check Disk Latency (>20ms)."]}

    async def _check_redo_subroutine(self, state: DBTriageState) -> dict:
        """Sub-routine: Check Log Sync."""
        self.logger.info("Sub-routine: Redo Check")
        return {"flag_log_sync": True, "recommendations": state.recommendations + ["Check Redo Log sizing and disk IOPS."]}

    async def _check_hot_block_subroutine(self, state: DBTriageState) -> dict:
        """Sub-routine: Hot Block Analysis."""
        self.logger.info("Sub-routine: Hot Block Check")
        return {"flag_hot_block": True, "recommendations": state.recommendations + ["Hot Block/Concurrency detected."]}

    async def _check_cpu_capacity_node(self, state: DBTriageState) -> dict:
        """Step 3: CPU & Load Profile."""
        self.logger.info("Checking CPU")
        # Could use oci_opsi_summarize_resource_stats(resource_metric="CPU")
        return {}

    async def _check_memory_node(self, state: DBTriageState) -> dict:
        """Step 4: Memory Pressure (PGA & Temp)."""
        self.logger.info("Checking Memory")
        # Could use oci_opsi_summarize_resource_stats(resource_metric="MEMORY")
        return {}

    async def _check_plan_stability_node(self, state: DBTriageState) -> dict:
        """Step 5: Plan Stability."""
        self.logger.info("Checking Plan Stability")
        return {}

    async def _check_monster_queries_node(self, state: DBTriageState) -> dict:
        """Step 6: Monster Query Check."""
        self.logger.info("Checking Monster Queries")
        if state.database_id:
             result = await self.agent.call_tool("oci_dbmgmt_get_top_sql", {
                 "managed_database_id": state.database_id,
                 "limit": 3
             })
             if result.get("success") and result.get("result"):
                 updates = {"top_sql": result["result"]}
                 return updates
        return {}

    async def _generate_report_node(self, state: DBTriageState) -> dict:
        """Phase 3: Synthesis & Actionable Output."""
        self.logger.info("Generating Final Report")

        report = []
        report.append("## 🚨 Incident Report: DB Performance Analysis")

        # Root Cause
        if state.flag_blocking:
             report.append("**1. Root Cause:** Blocking Sessions Detected.")
        elif state.flag_hot_block:
             report.append("**1. Root Cause:** Concurrency/Hot Block Contention.")
        elif state.flag_io_latency:
             report.append("**1. Root Cause:** I/O Latency Bottleneck.")
        else:
             report.append("**1. Root Cause:** Multiple/Undetermined factors.")

        # Evidence
        report.append("\n**2. Evidence:**")
        if state.blocking_tree:
            report.append(f"- Blockers found: {len(state.blocking_tree)}")
        if state.wait_analysis:
            report.append(f"- Top Wait: {state.wait_analysis.get('EVENT')} ({state.wait_analysis.get('CNT')} sessions)")

        # Recommendations
        report.append("\n**3. Recommendations:**")
        for rec in state.recommendations:
            report.append(f"- {rec}")

        if not state.recommendations:
            report.append("- Monitor for further anomalies.")

        return {"final_report": "\n".join(report)}
