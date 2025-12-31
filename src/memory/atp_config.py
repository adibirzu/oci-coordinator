"""
OCI ATP Database Configuration.

Provides connection configuration for Oracle Autonomous Transaction Processing (ATP)
using wallet-based mTLS authentication.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ATPConfig:
    """
    Configuration for OCI ATP connection.

    Supports two modes:
    1. Wallet-based (mTLS) - Recommended for production
    2. Direct connection string - For development/testing
    """

    # Wallet configuration
    wallet_dir: str | None = None
    wallet_password: str | None = None
    tns_name: str | None = None

    # Direct connection
    connection_string: str | None = None

    # Credentials
    user: str = "ADMIN"
    password: str | None = None

    # Connection pool settings
    pool_min: int = 2
    pool_max: int = 10
    pool_increment: int = 1

    @classmethod
    def from_env(cls) -> ATPConfig:
        """
        Create configuration from environment variables.

        Environment variables:
        - ATP_WALLET_DIR: Path to extracted wallet directory
        - ATP_WALLET_PASSWORD: Wallet password
        - ATP_TNS_NAME: TNS name from tnsnames.ora (e.g., 'dbname_medium')
        - ATP_USER: Database user (default: ADMIN)
        - ATP_PASSWORD: Database password
        - ATP_CONNECTION_STRING: Alternative full connection string

        Returns:
            ATPConfig instance
        """
        return cls(
            wallet_dir=os.getenv("ATP_WALLET_DIR"),
            wallet_password=os.getenv("ATP_WALLET_PASSWORD"),
            tns_name=os.getenv("ATP_TNS_NAME"),
            connection_string=os.getenv("ATP_CONNECTION_STRING"),
            user=os.getenv("ATP_USER", "ADMIN"),
            password=os.getenv("ATP_PASSWORD"),
            pool_min=int(os.getenv("ATP_POOL_MIN", "2")),
            pool_max=int(os.getenv("ATP_POOL_MAX", "10")),
            pool_increment=int(os.getenv("ATP_POOL_INCREMENT", "1")),
        )

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid
        """
        # Check for direct connection string
        if self.connection_string:
            return True

        # Check for wallet-based configuration
        if self.wallet_dir and self.tns_name and self.password:
            wallet_path = Path(self.wallet_dir)
            if not wallet_path.exists():
                logger.error("Wallet directory not found", path=self.wallet_dir)
                return False

            # Check for required wallet files
            required_files = ["tnsnames.ora", "sqlnet.ora", "cwallet.sso"]
            for file in required_files:
                if not (wallet_path / file).exists():
                    logger.error("Required wallet file missing", file=file)
                    return False

            return True

        logger.error("Missing ATP configuration - need wallet or connection string")
        return False

    def get_dsn(self) -> str:
        """
        Get the Data Source Name for connection.

        Returns:
            DSN string for oracledb connection
        """
        if self.connection_string:
            return self.connection_string

        if self.tns_name:
            return self.tns_name

        raise ValueError("No DSN available - configure tns_name or connection_string")

    def get_connect_params(self) -> dict:
        """
        Get connection parameters for oracledb.

        Returns:
            Dictionary of connection parameters
        """
        params = {
            "user": self.user,
            "password": self.password,
        }

        if self.wallet_dir:
            params["config_dir"] = self.wallet_dir
            if self.wallet_password:
                params["wallet_password"] = self.wallet_password

        return params


async def create_atp_pool(config: ATPConfig | None = None):
    """
    Create an async connection pool for ATP.

    Args:
        config: ATPConfig instance (uses env if not provided)

    Returns:
        oracledb async connection pool

    Raises:
        ValueError: If configuration is invalid
        Exception: If connection fails
    """
    import oracledb

    if config is None:
        config = ATPConfig.from_env()

    if not config.validate():
        raise ValueError("Invalid ATP configuration")

    logger.info(
        "Creating ATP connection pool",
        tns_name=config.tns_name,
        user=config.user,
        pool_min=config.pool_min,
        pool_max=config.pool_max,
    )

    try:
        # Set wallet location for thick mode if needed
        if config.wallet_dir:
            oracledb.init_oracle_client(config_dir=config.wallet_dir)

        pool = await oracledb.create_pool_async(
            user=config.user,
            password=config.password,
            dsn=config.get_dsn(),
            min=config.pool_min,
            max=config.pool_max,
            increment=config.pool_increment,
            **({
                "config_dir": config.wallet_dir,
                "wallet_password": config.wallet_password,
            } if config.wallet_dir else {}),
        )

        logger.info("ATP connection pool created successfully")
        return pool

    except Exception as e:
        logger.error("Failed to create ATP pool", error=str(e))
        raise


async def test_atp_connection(config: ATPConfig | None = None) -> bool:
    """
    Test ATP connection.

    Args:
        config: ATPConfig instance (uses env if not provided)

    Returns:
        True if connection successful
    """
    import oracledb

    if config is None:
        config = ATPConfig.from_env()

    if not config.validate():
        logger.error("Invalid ATP configuration")
        return False

    try:
        if config.wallet_dir:
            oracledb.init_oracle_client(config_dir=config.wallet_dir)

        async with await oracledb.connect_async(
            user=config.user,
            password=config.password,
            dsn=config.get_dsn(),
            **({
                "config_dir": config.wallet_dir,
                "wallet_password": config.wallet_password,
            } if config.wallet_dir else {}),
        ) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 'ATP Connection OK' FROM DUAL")
                row = await cursor.fetchone()
                logger.info("ATP connection test successful", result=row[0])
                return True

    except Exception as e:
        logger.error("ATP connection test failed", error=str(e))
        return False


# SQL scripts for initializing ATP tables
INIT_SCHEMA_SQL = """
-- Agent Memory Table
CREATE TABLE IF NOT EXISTS agent_memory (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    key VARCHAR2(512) NOT NULL UNIQUE,
    value CLOB NOT NULL,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
    updated_at TIMESTAMP DEFAULT SYSTIMESTAMP,
    ttl_seconds NUMBER,
    CONSTRAINT agent_memory_json CHECK (value IS JSON)
);

-- Conversation History Table
CREATE TABLE IF NOT EXISTS conversation_history (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    thread_id VARCHAR2(256) NOT NULL,
    message_index NUMBER NOT NULL,
    role VARCHAR2(50) NOT NULL,
    content CLOB NOT NULL,
    metadata CLOB,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP,
    CONSTRAINT conversation_metadata_json CHECK (metadata IS JSON)
);

CREATE INDEX IF NOT EXISTS idx_conversation_thread
ON conversation_history(thread_id, message_index);

-- Agent Registry Table
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id VARCHAR2(256) PRIMARY KEY,
    role VARCHAR2(128) NOT NULL,
    definition CLOB NOT NULL,
    status VARCHAR2(50) DEFAULT 'registered',
    registered_at TIMESTAMP DEFAULT SYSTIMESTAMP,
    last_heartbeat TIMESTAMP,
    CONSTRAINT agent_definition_json CHECK (definition IS JSON)
);

-- Audit Log Table
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    agent_id VARCHAR2(256),
    action VARCHAR2(256) NOT NULL,
    request CLOB,
    response CLOB,
    duration_ms NUMBER,
    status VARCHAR2(50),
    error_message CLOB,
    created_at TIMESTAMP DEFAULT SYSTIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_agent
ON agent_audit_log(agent_id, created_at DESC);
"""


async def init_atp_schema(config: ATPConfig | None = None) -> bool:
    """
    Initialize ATP database schema.

    Creates required tables if they don't exist.

    Args:
        config: ATPConfig instance (uses env if not provided)

    Returns:
        True if successful
    """
    import oracledb

    if config is None:
        config = ATPConfig.from_env()

    if not config.validate():
        logger.error("Invalid ATP configuration")
        return False

    logger.info("Initializing ATP schema")

    try:
        if config.wallet_dir:
            oracledb.init_oracle_client(config_dir=config.wallet_dir)

        async with await oracledb.connect_async(
            user=config.user,
            password=config.password,
            dsn=config.get_dsn(),
            **({
                "config_dir": config.wallet_dir,
                "wallet_password": config.wallet_password,
            } if config.wallet_dir else {}),
        ) as conn:
            async with conn.cursor() as cursor:
                # Execute each statement separately
                for statement in INIT_SCHEMA_SQL.split(";"):
                    statement = statement.strip()
                    if statement and not statement.startswith("--"):
                        try:
                            # Oracle doesn't support IF NOT EXISTS, handle differently
                            await cursor.execute(statement)
                        except oracledb.DatabaseError as e:
                            error, = e.args
                            # ORA-00955: name already used - table exists
                            # ORA-01408: index already exists
                            if error.code not in (955, 1408):
                                raise
                await conn.commit()

        logger.info("ATP schema initialized successfully")
        return True

    except Exception as e:
        logger.error("ATP schema initialization failed", error=str(e))
        return False
