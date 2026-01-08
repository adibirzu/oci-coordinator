#!/usr/bin/env python3
"""
SelectAI Provisioning Script

Connects to ATP using wallet-based authentication and provisions
tables and data for SelectAI natural language queries.

Usage:
    python scripts/provision_selectai.py [--clean] [--create-profile]

Options:
    --clean          Drop existing tables before creating
    --create-profile Create/update the SelectAI AI profile (requires OCI creds)
    --test           Run test queries after provisioning
    --verify-only    Only verify existing setup (no changes)

Environment variables required:
    ATP_TNS_NAME     - TNS alias (e.g., dbname_high)
    ATP_USER         - Database user (e.g., ADMIN)
    ATP_PASSWORD     - Database password
    ATP_WALLET_DIR   - Path to wallet directory
    ATP_WALLET_PASSWORD - Wallet password (optional, for ewallet.pem)

For SelectAI profile creation (--create-profile):
    OCI_COMPARTMENT_ID - OCI compartment OCID for GenAI access (optional)

Note: The database must have a dynamic group and policy allowing GenAI access:
    - Dynamic group: resource.type = 'autonomousdatabase' AND resource.id = '<db_ocid>'
    - Policy: allow dynamic-group <name> to use generative-ai-family in tenancy
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import oracledb
except ImportError:
    print("Error: oracledb not installed. Run: pip install oracledb")
    sys.exit(1)


def get_connection_params():
    """Get connection parameters from environment."""
    params = {
        "tns_name": os.getenv("ATP_TNS_NAME"),
        "user": os.getenv("ATP_USER"),
        "password": os.getenv("ATP_PASSWORD"),
        "wallet_dir": os.getenv("ATP_WALLET_DIR"),
        "wallet_password": os.getenv("ATP_WALLET_PASSWORD"),
    }

    missing = [k for k, v in params.items() if not v and k != "wallet_password"]
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("\nRequired variables:")
        print("  ATP_TNS_NAME      - TNS alias (e.g., dbname_high)")
        print("  ATP_USER          - Database user (e.g., ADMIN)")
        print("  ATP_PASSWORD      - Database password")
        print("  ATP_WALLET_DIR    - Path to wallet directory")
        sys.exit(1)

    # Verify wallet directory exists
    if not os.path.isdir(params["wallet_dir"]):
        print(f"Error: Wallet directory not found: {params['wallet_dir']}")
        sys.exit(1)

    return params


def connect_to_atp(params: dict) -> oracledb.Connection:
    """Connect to ATP using wallet-based authentication."""
    print(f"\nConnecting to {params['tns_name']} as {params['user']}...")

    # Configure wallet location
    wallet_location = params["wallet_dir"]

    # Read the TNS entry from tnsnames.ora to get the full connection string
    tnsnames_path = os.path.join(wallet_location, "tnsnames.ora")
    dsn = None

    if os.path.exists(tnsnames_path):
        with open(tnsnames_path, "r") as f:
            content = f.read()
            # Parse the TNS entry
            tns_name = params["tns_name"]
            for line in content.split("\n\n"):
                if line.strip().lower().startswith(tns_name.lower()):
                    # Extract the description part
                    eq_pos = line.find("=")
                    if eq_pos > 0:
                        dsn = line[eq_pos + 1:].strip()
                        break

    print(f"Wallet location: {wallet_location}")
    print(f"DSN configured: {'Yes' if dsn else 'No (using TNS name)'}")

    try:
        # Try thin mode first (no Oracle Client needed)
        # Use oracledb.ConnectParams for better control
        connect_params = oracledb.ConnectParams(
            user=params["user"],
            password=params["password"],
            config_dir=wallet_location,
            wallet_location=wallet_location,
            wallet_password=params.get("wallet_password"),
        )

        # Use full DSN string to preserve security settings (ssl_server_dn_match, etc.)
        if dsn:
            print(f"Using full DSN from tnsnames.ora")
            connection = oracledb.connect(
                user=params["user"],
                password=params["password"],
                dsn=dsn,
                config_dir=wallet_location,
                wallet_location=wallet_location,
                wallet_password=params.get("wallet_password"),
            )
        else:
            connection = oracledb.connect(
                user=params["user"],
                password=params["password"],
                dsn=params["tns_name"],
                config_dir=wallet_location,
                wallet_location=wallet_location,
                wallet_password=params.get("wallet_password"),
            )

        print(f"Connected successfully (thin mode)")
        return connection

    except oracledb.DatabaseError as e:
        error_str = str(e)
        print(f"Thin mode connection failed: {e}")

        # Check for common issues
        if "DPY-6000" in error_str:
            print("\nPossible causes:")
            print("  1. Database might be stopped - check OCI Console")
            print("  2. IP access control - add your IP to the Access Control List")
            print("  3. VPN/firewall blocking port 1522")
            print("\nNote: ATP databases auto-stop after 7 days of inactivity.")
            print("To start: OCI Console > Autonomous Database > Start")

        raise


def clean_tables(cursor):
    """Drop existing tables."""
    print("\nCleaning existing tables...")
    tables = ["ORDER_ITEMS", "ORDERS", "PRODUCTS", "CUSTOMERS", "REGIONS"]

    for table in tables:
        try:
            cursor.execute(f"DROP TABLE {table} CASCADE CONSTRAINTS")
            print(f"  Dropped {table}")
        except oracledb.DatabaseError:
            print(f"  {table} not found (skipping)")


def create_tables(cursor):
    """Create sample tables."""
    print("\nCreating tables...")

    # Regions table
    cursor.execute("""
        CREATE TABLE regions (
            region_id       NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            region_name     VARCHAR2(100) NOT NULL,
            country         VARCHAR2(100) NOT NULL,
            timezone        VARCHAR2(50)
        )
    """)
    cursor.execute("COMMENT ON TABLE regions IS 'Geographic regions for business operations'")
    print("  Created REGIONS")

    # Customers table
    cursor.execute("""
        CREATE TABLE customers (
            customer_id     NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            customer_name   VARCHAR2(200) NOT NULL,
            email           VARCHAR2(200),
            phone           VARCHAR2(50),
            region_id       NUMBER REFERENCES regions(region_id),
            customer_type   VARCHAR2(50) CHECK (customer_type IN ('ENTERPRISE', 'SMB', 'STARTUP', 'INDIVIDUAL')),
            created_date    DATE DEFAULT SYSDATE,
            total_spend     NUMBER(15,2) DEFAULT 0,
            status          VARCHAR2(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE', 'CHURNED'))
        )
    """)
    cursor.execute("COMMENT ON TABLE customers IS 'Customer master data with spending history'")
    print("  Created CUSTOMERS")

    # Products table
    cursor.execute("""
        CREATE TABLE products (
            product_id      NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            product_name    VARCHAR2(200) NOT NULL,
            category        VARCHAR2(100),
            subcategory     VARCHAR2(100),
            unit_price      NUMBER(10,2) NOT NULL,
            cost_price      NUMBER(10,2),
            inventory_qty   NUMBER DEFAULT 0,
            status          VARCHAR2(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'DISCONTINUED', 'OUT_OF_STOCK'))
        )
    """)
    cursor.execute("COMMENT ON TABLE products IS 'Product catalog with pricing and inventory'")
    print("  Created PRODUCTS")

    # Orders table
    cursor.execute("""
        CREATE TABLE orders (
            order_id        NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            customer_id     NUMBER REFERENCES customers(customer_id),
            order_date      DATE DEFAULT SYSDATE,
            ship_date       DATE,
            total_amount    NUMBER(15,2),
            discount_pct    NUMBER(5,2) DEFAULT 0,
            status          VARCHAR2(30) DEFAULT 'PENDING'
                            CHECK (status IN ('PENDING', 'PROCESSING', 'SHIPPED', 'DELIVERED', 'CANCELLED', 'RETURNED')),
            payment_method  VARCHAR2(50),
            sales_rep       VARCHAR2(100)
        )
    """)
    cursor.execute("COMMENT ON TABLE orders IS 'Customer orders with fulfillment status'")
    print("  Created ORDERS")

    # Order items table
    cursor.execute("""
        CREATE TABLE order_items (
            item_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            order_id        NUMBER REFERENCES orders(order_id),
            product_id      NUMBER REFERENCES products(product_id),
            quantity        NUMBER NOT NULL,
            unit_price      NUMBER(10,2) NOT NULL,
            line_total      NUMBER(12,2) GENERATED ALWAYS AS (quantity * unit_price) VIRTUAL
        )
    """)
    cursor.execute("COMMENT ON TABLE order_items IS 'Line items for each order'")
    print("  Created ORDER_ITEMS")


def insert_sample_data(cursor):
    """Insert sample data."""
    print("\nInserting sample data...")

    # Regions
    regions = [
        ("West Coast", "USA", "America/Los_Angeles"),
        ("East Coast", "USA", "America/New_York"),
        ("EMEA", "UK", "Europe/London"),
        ("APAC", "Singapore", "Asia/Singapore"),
        ("LATAM", "Brazil", "America/Sao_Paulo"),
    ]
    cursor.executemany(
        "INSERT INTO regions (region_name, country, timezone) VALUES (:1, :2, :3)",
        regions
    )
    print(f"  Inserted {len(regions)} regions")

    # Customers
    customers = [
        ("Acme Corporation", "contact@acme.com", 1, "ENTERPRISE", 1250000.00, "ACTIVE"),
        ("TechStart Inc", "info@techstart.io", 1, "STARTUP", 45000.00, "ACTIVE"),
        ("Global Finance Ltd", "procurement@globalfin.com", 3, "ENTERPRISE", 2100000.00, "ACTIVE"),
        ("Smith & Associates", "orders@smithassoc.com", 2, "SMB", 78500.00, "ACTIVE"),
        ("Pacific Trading Co", "sales@pacifictrading.sg", 4, "ENTERPRISE", 890000.00, "ACTIVE"),
        ("CloudNine Solutions", "hello@cloudnine.io", 1, "STARTUP", 23000.00, "ACTIVE"),
        ("MegaRetail Group", "vendor@megaretail.com", 2, "ENTERPRISE", 3400000.00, "ACTIVE"),
        ("Latin Logistics", "compras@latinlog.br", 5, "SMB", 156000.00, "ACTIVE"),
        ("Inactive Corp", "old@inactive.com", 2, "SMB", 12000.00, "CHURNED"),
        ("Jane Doe", "jane.doe@email.com", 1, "INDIVIDUAL", 1500.00, "ACTIVE"),
    ]
    cursor.executemany(
        """INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
           VALUES (:1, :2, :3, :4, :5, :6)""",
        customers
    )
    print(f"  Inserted {len(customers)} customers")

    # Products
    products = [
        ("Enterprise Cloud License", "Software", "Licenses", 50000.00, 10000.00, 999),
        ("Professional Support Plan", "Services", "Support", 12000.00, 3000.00, 999),
        ("Data Analytics Module", "Software", "Add-ons", 15000.00, 5000.00, 999),
        ("Security Suite", "Software", "Security", 25000.00, 8000.00, 999),
        ("Training Package", "Services", "Training", 5000.00, 1500.00, 999),
        ("API Gateway License", "Software", "Integration", 8000.00, 2500.00, 999),
        ("Consulting Day", "Services", "Consulting", 2500.00, 800.00, 999),
        ("Startup Bundle", "Software", "Bundles", 10000.00, 4000.00, 999),
    ]
    cursor.executemany(
        """INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
           VALUES (:1, :2, :3, :4, :5, :6)""",
        products
    )
    print(f"  Inserted {len(products)} products")

    # Orders
    from datetime import date
    orders = [
        (1, date(2024, 10, 15), date(2024, 10, 18), 125000.00, 10, "DELIVERED", "Invoice", "Sarah Johnson"),
        (1, date(2024, 11, 20), date(2024, 11, 22), 75000.00, 5, "DELIVERED", "Invoice", "Sarah Johnson"),
        (3, date(2024, 10, 1), date(2024, 10, 5), 200000.00, 15, "DELIVERED", "Wire Transfer", "Michael Chen"),
        (5, date(2024, 11, 10), date(2024, 11, 15), 95000.00, 8, "DELIVERED", "Invoice", "Lisa Wong"),
        (7, date(2024, 12, 1), date(2024, 12, 5), 340000.00, 12, "DELIVERED", "Wire Transfer", "Sarah Johnson"),
        (2, date(2024, 12, 15), date(2024, 12, 18), 10000.00, 0, "DELIVERED", "Credit Card", "James Miller"),
        (4, date(2025, 1, 5), None, 28500.00, 5, "PROCESSING", "Invoice", "Michael Chen"),
        (6, date(2025, 1, 10), None, 15000.00, 0, "PENDING", "Credit Card", "James Miller"),
        (8, date(2025, 1, 12), None, 42000.00, 5, "SHIPPED", "Invoice", "Lisa Wong"),
        (10, date(2025, 1, 15), None, 1500.00, 0, "PENDING", "Credit Card", "James Miller"),
    ]
    cursor.executemany(
        """INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
           VALUES (:1, :2, :3, :4, :5, :6, :7, :8)""",
        orders
    )
    print(f"  Inserted {len(orders)} orders")

    # Order items
    order_items = [
        (1, 1, 2, 50000.00), (1, 2, 2, 12000.00),
        (2, 3, 3, 15000.00), (2, 5, 6, 5000.00),
        (3, 1, 3, 50000.00), (3, 4, 2, 25000.00),
        (4, 1, 1, 50000.00), (4, 2, 1, 12000.00), (4, 3, 2, 15000.00),
        (5, 1, 5, 50000.00), (5, 4, 3, 25000.00), (5, 7, 10, 2500.00),
        (6, 8, 1, 10000.00),
        (7, 3, 1, 15000.00), (7, 6, 1, 8000.00), (7, 5, 1, 5000.00),
        (8, 8, 1, 10000.00), (8, 5, 1, 5000.00),
        (9, 1, 1, 50000.00), (9, 7, 4, 2500.00),
        (10, 5, 1, 5000.00),
    ]
    cursor.executemany(
        """INSERT INTO order_items (order_id, product_id, quantity, unit_price)
           VALUES (:1, :2, :3, :4)""",
        order_items
    )
    print(f"  Inserted {len(order_items)} order items")


def verify_setup(cursor):
    """Verify table creation and data."""
    print("\nVerifying setup...")

    tables = ["REGIONS", "CUSTOMERS", "PRODUCTS", "ORDERS", "ORDER_ITEMS"]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")


def create_selectai_profile(cursor, compartment_id: str | None = None):
    """Create or update the SelectAI profile for OCI GenAI.

    Uses meta.llama-3.1-405b-instruct model which is available in eu-frankfurt-1
    (same region as the database), avoiding cross-region routing issues.
    """
    print("\nConfiguring SelectAI profile...")

    profile_name = "OCI_GENAI"

    # Check if profile already exists
    try:
        cursor.execute(
            "SELECT profile_name, status FROM USER_CLOUD_AI_PROFILES WHERE profile_name = :1",
            [profile_name]
        )
        existing = cursor.fetchone()

        if existing:
            print(f"  Profile '{profile_name}' already exists (status: {existing[1]})")
            print("  Dropping existing profile to recreate with updated settings...")
            cursor.execute(f"""
                BEGIN
                    DBMS_CLOUD_AI.DROP_PROFILE(profile_name => '{profile_name}');
                END;
            """)
            print(f"  Dropped existing profile")
    except oracledb.DatabaseError as e:
        # Table might not exist if SelectAI was never used
        if "ORA-00942" not in str(e):
            raise

    # Build the profile attributes
    # Using meta.llama-3.1-405b-instruct which is available in eu-frankfurt-1
    # This avoids the "my$cloud_domain" error that occurs with cross-region models
    attributes = {
        "provider": "oci",
        "credential_name": "OCI$RESOURCE_PRINCIPAL",
        "model": "meta.llama-3.1-405b-instruct",
        "object_list": [
            {"owner": "ADMIN", "name": "CUSTOMERS"},
            {"owner": "ADMIN", "name": "ORDERS"},
            {"owner": "ADMIN", "name": "ORDER_ITEMS"},
            {"owner": "ADMIN", "name": "PRODUCTS"},
            {"owner": "ADMIN", "name": "REGIONS"},
        ]
    }

    # Add compartment if provided (helps with resource principal auth)
    if compartment_id:
        attributes["oci_compartment_id"] = compartment_id
        print(f"  Using compartment: {compartment_id[:40]}...")
    else:
        print("  No compartment ID provided (using database's default compartment)")

    attributes_json = json.dumps(attributes)

    print(f"  Creating profile with model: {attributes['model']}")
    print(f"  Tables: {[t['name'] for t in attributes['object_list']]}")

    try:
        cursor.execute(f"""
            BEGIN
                DBMS_CLOUD_AI.CREATE_PROFILE(
                    profile_name => '{profile_name}',
                    attributes => :1
                );
            END;
        """, [attributes_json])

        # Verify creation
        cursor.execute(
            "SELECT profile_name, status FROM USER_CLOUD_AI_PROFILES WHERE profile_name = :1",
            [profile_name]
        )
        result = cursor.fetchone()

        if result:
            print(f"  Profile '{result[0]}' created successfully (status: {result[1]})")
        else:
            print("  Warning: Profile created but not visible in USER_CLOUD_AI_PROFILES")

    except oracledb.DatabaseError as e:
        error = str(e)
        if "ORA-20404" in error and "my$cloud_domain" in error:
            print("\n  ERROR: Cross-region routing issue detected!")
            print("  The model endpoint could not be resolved.")
            print("  Ensure the database and model are in the same region.")
            print("  Current model: meta.llama-3.1-405b-instruct (eu-frankfurt-1)")
        elif "ORA-20000" in error:
            print(f"\n  ERROR: GenAI access denied. Ensure IAM policy exists:")
            print(f"  allow dynamic-group <name> to use generative-ai-family in tenancy")
        else:
            print(f"\n  ERROR creating profile: {e}")
        raise


def test_selectai(cursor):
    """Test SelectAI queries (requires profile to be configured)."""
    print("\nTesting SelectAI queries...")

    try:
        # Check if profile exists
        cursor.execute("SELECT profile_name, status FROM USER_CLOUD_AI_PROFILES")
        profiles = cursor.fetchall()

        if not profiles:
            print("  No SelectAI profiles found. Create one first using:")
            print("  DBMS_CLOUD_AI.CREATE_PROFILE(profile_name => 'OCI_GENAI', ...)")
            return

        print(f"  Available profiles: {[p[0] for p in profiles]}")

        # Try a simple SelectAI query
        cursor.execute("SELECT AI showsql how many customers do we have")
        result = cursor.fetchone()
        print(f"  Test query result: {result}")

    except oracledb.DatabaseError as e:
        error = str(e)
        if "ORA-20000" in error or "ORA-06550" in error:
            print("  SelectAI not available (profile not configured or GenAI access issue)")
        else:
            print(f"  SelectAI test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Provision SelectAI tables and data")
    parser.add_argument("--clean", action="store_true", help="Drop existing tables first")
    parser.add_argument("--create-profile", action="store_true",
                        help="Create/update SelectAI profile for OCI GenAI")
    parser.add_argument("--test", action="store_true", help="Test SelectAI after provisioning")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing setup")
    parser.add_argument("--compartment-id", type=str,
                        help="OCI compartment OCID for GenAI access (or set OCI_COMPARTMENT_ID env var)")
    args = parser.parse_args()

    print("=" * 60)
    print("SelectAI Provisioning Script")
    print("=" * 60)

    # Get connection parameters
    params = get_connection_params()

    # Connect
    connection = connect_to_atp(params)
    cursor = connection.cursor()

    try:
        if args.verify_only:
            verify_setup(cursor)
            if args.test:
                test_selectai(cursor)
            return

        if args.clean:
            clean_tables(cursor)

        create_tables(cursor)
        insert_sample_data(cursor)
        connection.commit()

        verify_setup(cursor)

        # Create SelectAI profile if requested
        if args.create_profile:
            compartment_id = args.compartment_id or os.getenv("OCI_COMPARTMENT_ID")
            create_selectai_profile(cursor, compartment_id)
            connection.commit()

        if args.test:
            test_selectai(cursor)

        print("\n" + "=" * 60)
        print("Provisioning complete!")
        print("=" * 60)

        if not args.create_profile:
            print("\nNext steps:")
            print("1. Ensure IAM dynamic group and policy exist for GenAI access")
            print("2. Run with --create-profile to create the SelectAI profile")
            print("3. Test with: SELECT AI showsql how many customers do we have")
        else:
            print("\nSelectAI is ready! Test with:")
            print("  SELECT AI showsql how many customers do we have")
            print("  SELECT AI runsql show me the top 5 customers by total spend")

    finally:
        cursor.close()
        connection.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
