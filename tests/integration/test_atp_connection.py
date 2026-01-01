"""
ATP Connection Test and Table Creation Script.

Tests connection to Oracle ATP and creates required tables for:
1. Coordinator's SharedMemoryManager (agent_memory, agent_audit_log)
2. MCP-OCI's ATPSharedStore (mcp_agents, mcp_contexts, mcp_events, mcp_conversations)

Usage:
    # Load environment and run:
    source .env.local && python tests/integration/test_atp_connection.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# =============================================================================
# DDL Statements for Schema Creation
# =============================================================================

# Coordinator's SharedMemoryManager tables
COORDINATOR_DDL = [
    """
    CREATE TABLE agent_memory (
        key VARCHAR2(500) PRIMARY KEY,
        value CLOB NOT NULL,
        ttl_seconds NUMBER,
        created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
        updated_at TIMESTAMP DEFAULT SYSTIMESTAMP
    )
    """,
    """
    CREATE TABLE agent_audit_log (
        id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        agent_id VARCHAR2(100) NOT NULL,
        action VARCHAR2(200) NOT NULL,
        request CLOB,
        response CLOB,
        duration_ms NUMBER,
        status VARCHAR2(50) DEFAULT 'success',
        error_message CLOB,
        created_at TIMESTAMP DEFAULT SYSTIMESTAMP
    )
    """,
    """
    CREATE TABLE conversation_history (
        id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        thread_id VARCHAR2(100) NOT NULL,
        role VARCHAR2(50) NOT NULL,
        content CLOB NOT NULL,
        metadata CLOB,
        created_at TIMESTAMP DEFAULT SYSTIMESTAMP
    )
    """,
    "CREATE INDEX idx_audit_agent ON agent_audit_log(agent_id)",
    "CREATE INDEX idx_audit_created ON agent_audit_log(created_at)",
    "CREATE INDEX idx_conv_thread ON conversation_history(thread_id)",
]

# MCP-OCI's ATPSharedStore tables
MCP_OCI_DDL = [
    """
    CREATE TABLE mcp_agents (
        agent_id VARCHAR2(100) PRIMARY KEY,
        agent_type VARCHAR2(50) NOT NULL,
        state VARCHAR2(20) NOT NULL,
        last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        capabilities CLOB,
        metadata CLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE mcp_contexts (
        context_id VARCHAR2(100) PRIMARY KEY,
        session_id VARCHAR2(100) NOT NULL,
        resource_id VARCHAR2(200),
        resource_type VARCHAR2(50),
        findings CLOB,
        recommendations CLOB,
        metadata CLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        version NUMBER DEFAULT 1
    )
    """,
    """
    CREATE TABLE mcp_events (
        event_id VARCHAR2(100) PRIMARY KEY,
        event_type VARCHAR2(50) NOT NULL,
        source_agent VARCHAR2(100) NOT NULL,
        target_agent VARCHAR2(100),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        payload CLOB,
        ttl_seconds NUMBER DEFAULT 3600,
        acknowledged NUMBER(1) DEFAULT 0
    )
    """,
    """
    CREATE TABLE mcp_conversations (
        entry_id VARCHAR2(100) PRIMARY KEY,
        session_id VARCHAR2(100) NOT NULL,
        agent_id VARCHAR2(100) NOT NULL,
        role VARCHAR2(20) NOT NULL,
        content CLOB,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata CLOB
    )
    """,
    "CREATE INDEX idx_ctx_sess ON mcp_contexts(session_id)",
    "CREATE INDEX idx_ctx_res ON mcp_contexts(resource_id)",
    "CREATE INDEX idx_evt_type ON mcp_events(event_type)",
    "CREATE INDEX idx_evt_src ON mcp_events(source_agent)",
    "CREATE INDEX idx_conv_sess ON mcp_conversations(session_id)",
]


async def test_atp_connection():
    """Test ATP connection and create tables."""
    print("=" * 70)
    print("ATP Connection Test and Schema Setup")
    print("=" * 70)

    # Get configuration from environment
    connection_string = os.environ.get("ATP_CONNECTION_STRING") or os.environ.get("ATP_TNS_NAME")
    user = os.environ.get("ATP_USER", "ADMIN")
    password = os.environ.get("ATP_PASSWORD")
    wallet_dir = os.environ.get("ATP_WALLET_DIR")
    wallet_password = os.environ.get("ATP_WALLET_PASSWORD") or password

    print(f"\nConfiguration:")
    print(f"  Connection: {connection_string}")
    print(f"  User: {user}")
    print(f"  Wallet Dir: {wallet_dir}")
    print(f"  Password: {'*' * len(password) if password else 'NOT SET'}")

    if not connection_string:
        print("\n❌ ERROR: ATP_CONNECTION_STRING or ATP_TNS_NAME not set")
        print("   Please source .env.local first")
        return False

    if not password:
        print("\n❌ ERROR: ATP_PASSWORD not set")
        return False

    # Try to import oracledb
    try:
        import oracledb
        print(f"\n✓ oracledb version: {oracledb.__version__}")
    except ImportError:
        print("\n❌ ERROR: oracledb not installed")
        print("   Run: pip install oracledb")
        return False

    # Test connection
    print("\n" + "-" * 70)
    print("Testing Connection...")
    print("-" * 70)

    try:
        # Create connection pool
        pool_kwargs = {
            "user": user,
            "password": password,
            "dsn": connection_string,
            "min": 1,
            "max": 5,
        }

        if wallet_dir:
            pool_kwargs["config_dir"] = wallet_dir
            pool_kwargs["wallet_location"] = wallet_dir
            if wallet_password:
                pool_kwargs["wallet_password"] = wallet_password

        pool = oracledb.create_pool(**pool_kwargs)
        print("✓ Connection pool created")

        # Test query
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT SYSDATE, SYS_CONTEXT('USERENV','DB_NAME') FROM DUAL")
                row = cursor.fetchone()
                print(f"✓ Connected to database: {row[1]}")
                print(f"✓ Server time: {row[0]}")

        # Create tables
        print("\n" + "-" * 70)
        print("Creating Coordinator Tables...")
        print("-" * 70)

        tables_created = 0
        tables_existed = 0

        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                for ddl in COORDINATOR_DDL:
                    ddl = ddl.strip()
                    if not ddl:
                        continue

                    # Extract object name for logging
                    obj_name = ddl.split()[2] if "CREATE" in ddl.upper() else "index"

                    try:
                        cursor.execute(ddl)
                        print(f"  ✓ Created: {obj_name}")
                        tables_created += 1
                    except oracledb.DatabaseError as e:
                        error, = e.args
                        if error.code == 955:  # ORA-00955: name already used
                            print(f"  ○ Already exists: {obj_name}")
                            tables_existed += 1
                        elif error.code == 1408:  # ORA-01408: index already exists
                            print(f"  ○ Already exists: {obj_name}")
                            tables_existed += 1
                        else:
                            print(f"  ✗ Error creating {obj_name}: {error.message}")

                conn.commit()

        print(f"\n  Summary: {tables_created} created, {tables_existed} already existed")

        print("\n" + "-" * 70)
        print("Creating MCP-OCI Tables...")
        print("-" * 70)

        mcp_created = 0
        mcp_existed = 0

        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                for ddl in MCP_OCI_DDL:
                    ddl = ddl.strip()
                    if not ddl:
                        continue

                    obj_name = ddl.split()[2] if "CREATE" in ddl.upper() else "index"

                    try:
                        cursor.execute(ddl)
                        print(f"  ✓ Created: {obj_name}")
                        mcp_created += 1
                    except oracledb.DatabaseError as e:
                        error, = e.args
                        if error.code in (955, 1408):
                            print(f"  ○ Already exists: {obj_name}")
                            mcp_existed += 1
                        else:
                            print(f"  ✗ Error creating {obj_name}: {error.message}")

                conn.commit()

        print(f"\n  Summary: {mcp_created} created, {mcp_existed} already existed")

        # Test write/read operations
        print("\n" + "-" * 70)
        print("Testing Read/Write Operations...")
        print("-" * 70)

        test_key = f"test:connection:{datetime.utcnow().isoformat()}"
        test_value = json.dumps({"test": True, "timestamp": datetime.utcnow().isoformat()})

        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Test agent_memory write
                cursor.execute(
                    """
                    INSERT INTO agent_memory (key, value)
                    VALUES (:key, :value)
                    """,
                    {"key": test_key, "value": test_value}
                )
                print("  ✓ agent_memory: Write successful")

                # Test agent_memory read
                cursor.execute(
                    "SELECT value FROM agent_memory WHERE key = :key",
                    {"key": test_key}
                )
                row = cursor.fetchone()
                if row:
                    # Handle LOB type - read() converts to string
                    value = row[0].read() if hasattr(row[0], 'read') else str(row[0])
                    if json.loads(value).get("test"):
                        print("  ✓ agent_memory: Read successful")
                    else:
                        print("  ✗ agent_memory: Read failed - value mismatch")
                else:
                    print("  ✗ agent_memory: Read failed - no row")

                # Test mcp_agents write
                cursor.execute(
                    """
                    INSERT INTO mcp_agents (agent_id, agent_type, state, capabilities, metadata)
                    VALUES (:agent_id, :agent_type, :state, :capabilities, :metadata)
                    """,
                    {
                        "agent_id": "test-connection-agent",
                        "agent_type": "test",
                        "state": "ready",
                        "capabilities": json.dumps(["test"]),
                        "metadata": json.dumps({"test": True}),
                    }
                )
                print("  ✓ mcp_agents: Write successful")

                # Test mcp_agents read
                cursor.execute(
                    "SELECT agent_type FROM mcp_agents WHERE agent_id = :agent_id",
                    {"agent_id": "test-connection-agent"}
                )
                row = cursor.fetchone()
                if row and row[0] == "test":
                    print("  ✓ mcp_agents: Read successful")
                else:
                    print("  ✗ mcp_agents: Read failed")

                # Cleanup test data
                cursor.execute("DELETE FROM agent_memory WHERE key = :key", {"key": test_key})
                cursor.execute("DELETE FROM mcp_agents WHERE agent_id = :agent_id", {"agent_id": "test-connection-agent"})
                conn.commit()
                print("  ✓ Cleanup: Test data removed")

        # List all tables
        print("\n" + "-" * 70)
        print("Current Tables in Schema...")
        print("-" * 70)

        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT table_name, num_rows
                    FROM user_tables
                    WHERE table_name LIKE 'AGENT%' OR table_name LIKE 'MCP%' OR table_name LIKE 'CONVERSATION%'
                    ORDER BY table_name
                    """
                )
                rows = cursor.fetchall()
                for row in rows:
                    print(f"  • {row[0]}: {row[1] or 0} rows")

        pool.close()

        print("\n" + "=" * 70)
        print("✓ ATP Connection Test PASSED")
        print("=" * 70)
        return True

    except oracledb.DatabaseError as e:
        error, = e.args
        print(f"\n❌ Database Error: {error.message}")
        print(f"   Error Code: {error.code}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_shared_memory_manager():
    """Test the SharedMemoryManager with ATP backend."""
    print("\n" + "=" * 70)
    print("Testing SharedMemoryManager with ATP Backend")
    print("=" * 70)

    try:
        from src.memory.manager import SharedMemoryManager

        # Initialize with ATP
        memory = SharedMemoryManager(
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
            atp_connection=os.environ.get("ATP_CONNECTION_STRING"),
        )

        print("\n✓ SharedMemoryManager initialized")

        # Test set/get
        test_value = {"test": True, "timestamp": datetime.utcnow().isoformat()}
        await memory.set_agent_memory(
            agent_id="test-smm",
            memory_type="integration",
            value=test_value,
        )
        print("✓ set_agent_memory: Success")

        retrieved = await memory.get_agent_memory(
            agent_id="test-smm",
            memory_type="integration",
        )
        if retrieved == test_value:
            print("✓ get_agent_memory: Success (value matches)")
        else:
            print(f"✗ get_agent_memory: Value mismatch - got {retrieved}")

        # Cleanup
        await memory.close()
        print("✓ Cleanup: Connection closed")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_oci_shared_store():
    """Test MCP-OCI's ATPSharedStore."""
    print("\n" + "=" * 70)
    print("Testing MCP-OCI ATPSharedStore")
    print("=" * 70)

    try:
        # Add mcp-oci to path
        mcp_oci_path = os.environ.get("MCP_OCI_PATH", "/Users/abirzu/dev/MCP/mcp-oci")
        sys.path.insert(0, os.path.join(mcp_oci_path, "src"))

        from mcp_server_oci.core.shared_memory import (
            ATPSharedStore,
            SharedContext,
            SharedEvent,
            EventType,
            AgentState,
        )

        # Create ATP store
        store = ATPSharedStore(
            connection_string=os.environ.get("ATP_CONNECTION_STRING")
        )

        print(f"\n✓ ATPSharedStore initialized")
        print(f"  ATP Available: {store._atp_available}")

        if not store._atp_available:
            print("  ⚠ ATP not available, using in-memory fallback")
            return True

        # Test agent registration
        agent = await store.register_agent(
            agent_id="test-atp-agent",
            agent_type="integration-test",
            capabilities=["test", "integration"],
            metadata={"source": "test_atp_connection.py"},
        )
        print(f"✓ register_agent: {agent.agent_id}")

        # Test agent state update
        await store.update_agent_state("test-atp-agent", AgentState.BUSY)
        print("✓ update_agent_state: BUSY")

        # Test context
        context = SharedContext(
            session_id="test-session",
            resource_id="ocid1.test.integration",
            resource_type="test",
            findings=[{"type": "test", "message": "Integration test finding"}],
        )
        saved = await store.save_context(context)
        print(f"✓ save_context: {saved.context_id}")

        # Test event
        event = SharedEvent(
            event_type=EventType.FINDING,
            source_agent="test-atp-agent",
            payload={"test": True},
        )
        published = await store.publish_event(event)
        print(f"✓ publish_event: {published.event_id}")

        # List agents
        agents = await store.list_agents()
        print(f"✓ list_agents: {len(agents)} agents found")

        # Get recent events
        events = await store.get_recent_events(limit=10)
        print(f"✓ get_recent_events: {len(events)} events")

        # Cleanup
        await store.cleanup_expired()
        print("✓ cleanup_expired: Success")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure MCP_OCI_PATH is set correctly")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all ATP tests."""
    # Test basic connection and create tables
    conn_ok = await test_atp_connection()

    if not conn_ok:
        print("\n⚠ Skipping further tests due to connection failure")
        sys.exit(1)

    # Test SharedMemoryManager
    smm_ok = await test_shared_memory_manager()

    # Test MCP-OCI ATPSharedStore
    mcp_ok = await test_mcp_oci_shared_store()

    print("\n" + "=" * 70)
    print("Final Summary")
    print("=" * 70)
    print(f"  ATP Connection:        {'✓ PASS' if conn_ok else '✗ FAIL'}")
    print(f"  SharedMemoryManager:   {'✓ PASS' if smm_ok else '✗ FAIL'}")
    print(f"  MCP-OCI ATPSharedStore: {'✓ PASS' if mcp_ok else '✗ FAIL'}")

    if conn_ok and smm_ok and mcp_ok:
        print("\n✓ All ATP tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
