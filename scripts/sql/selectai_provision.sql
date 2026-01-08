-- =============================================================================
-- SelectAI Provisioning Script for OCI AI Agent Coordinator
-- =============================================================================
-- This script sets up SelectAI capabilities in an Autonomous Database
-- Run as ADMIN user with appropriate privileges
--
-- Prerequisites:
--   - OCI GenAI service enabled in your region
--   - OCI credentials (user_ocid, tenancy_ocid, private_key, fingerprint)
--   - Autonomous Database 19c or later
--
-- Usage:
--   Connect to your ATP using SQL Developer, SQLcl, or Database Actions
--   Run this script after updating the credential placeholders
-- =============================================================================

SET SERVEROUTPUT ON;
SET ECHO ON;

-- =============================================================================
-- Step 1: Clean up existing SelectAI configuration (if any)
-- =============================================================================
DECLARE
    v_count NUMBER;
BEGIN
    -- Drop existing AI profile if exists
    SELECT COUNT(*) INTO v_count FROM USER_CLOUD_AI_PROFILES WHERE profile_name = 'OCI_GENAI';
    IF v_count > 0 THEN
        DBMS_CLOUD_AI.DROP_PROFILE(profile_name => 'OCI_GENAI');
        DBMS_OUTPUT.PUT_LINE('Dropped existing OCI_GENAI profile');
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Note: No existing profile to drop');
END;
/

-- Drop existing credential if exists
BEGIN
    DBMS_CLOUD.DROP_CREDENTIAL(credential_name => 'OCI_GENAI_CRED');
    DBMS_OUTPUT.PUT_LINE('Dropped existing OCI_GENAI_CRED credential');
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('Note: No existing credential to drop');
END;
/

-- =============================================================================
-- Step 2: Drop and recreate sample tables
-- =============================================================================
BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE order_items CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN NULL;
END;
/

BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE orders CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN NULL;
END;
/

BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE products CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN NULL;
END;
/

BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE customers CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN NULL;
END;
/

BEGIN
    EXECUTE IMMEDIATE 'DROP TABLE regions CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN NULL;
END;
/

DBMS_OUTPUT.PUT_LINE('Cleaned up existing tables');

-- =============================================================================
-- Step 3: Create sample tables for SelectAI demo
-- =============================================================================

-- Regions table
CREATE TABLE regions (
    region_id       NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    region_name     VARCHAR2(100) NOT NULL,
    country         VARCHAR2(100) NOT NULL,
    timezone        VARCHAR2(50)
);

COMMENT ON TABLE regions IS 'Geographic regions for business operations';
COMMENT ON COLUMN regions.region_id IS 'Unique identifier for the region';
COMMENT ON COLUMN regions.region_name IS 'Name of the region (e.g., West Coast, EMEA)';

-- Customers table
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
);

COMMENT ON TABLE customers IS 'Customer master data with spending history';
COMMENT ON COLUMN customers.customer_name IS 'Full company or individual name';
COMMENT ON COLUMN customers.customer_type IS 'Customer segment: ENTERPRISE, SMB, STARTUP, or INDIVIDUAL';
COMMENT ON COLUMN customers.total_spend IS 'Cumulative spending in USD';

-- Products table
CREATE TABLE products (
    product_id      NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_name    VARCHAR2(200) NOT NULL,
    category        VARCHAR2(100),
    subcategory     VARCHAR2(100),
    unit_price      NUMBER(10,2) NOT NULL,
    cost_price      NUMBER(10,2),
    inventory_qty   NUMBER DEFAULT 0,
    status          VARCHAR2(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'DISCONTINUED', 'OUT_OF_STOCK'))
);

COMMENT ON TABLE products IS 'Product catalog with pricing and inventory';
COMMENT ON COLUMN products.unit_price IS 'Selling price per unit in USD';
COMMENT ON COLUMN products.cost_price IS 'Cost of goods sold per unit in USD';

-- Orders table
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
);

COMMENT ON TABLE orders IS 'Customer orders with fulfillment status';
COMMENT ON COLUMN orders.total_amount IS 'Order total in USD after discounts';
COMMENT ON COLUMN orders.discount_pct IS 'Discount percentage applied to order';

-- Order items table
CREATE TABLE order_items (
    item_id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id        NUMBER REFERENCES orders(order_id),
    product_id      NUMBER REFERENCES products(product_id),
    quantity        NUMBER NOT NULL,
    unit_price      NUMBER(10,2) NOT NULL,
    line_total      NUMBER(12,2) GENERATED ALWAYS AS (quantity * unit_price) VIRTUAL
);

COMMENT ON TABLE order_items IS 'Line items for each order';
COMMENT ON COLUMN order_items.line_total IS 'Calculated total for this line item';

DBMS_OUTPUT.PUT_LINE('Created sample tables');

-- =============================================================================
-- Step 4: Insert sample data
-- =============================================================================

-- Insert regions
INSERT INTO regions (region_name, country, timezone) VALUES ('West Coast', 'USA', 'America/Los_Angeles');
INSERT INTO regions (region_name, country, timezone) VALUES ('East Coast', 'USA', 'America/New_York');
INSERT INTO regions (region_name, country, timezone) VALUES ('EMEA', 'UK', 'Europe/London');
INSERT INTO regions (region_name, country, timezone) VALUES ('APAC', 'Singapore', 'Asia/Singapore');
INSERT INTO regions (region_name, country, timezone) VALUES ('LATAM', 'Brazil', 'America/Sao_Paulo');

-- Insert customers
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('Acme Corporation', 'contact@acme.com', 1, 'ENTERPRISE', 1250000.00, 'ACTIVE');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('TechStart Inc', 'info@techstart.io', 1, 'STARTUP', 45000.00, 'ACTIVE');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('Global Finance Ltd', 'procurement@globalfin.com', 3, 'ENTERPRISE', 2100000.00, 'ACTIVE');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('Smith & Associates', 'orders@smithassoc.com', 2, 'SMB', 78500.00, 'ACTIVE');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('Pacific Trading Co', 'sales@pacifictrading.sg', 4, 'ENTERPRISE', 890000.00, 'ACTIVE');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('CloudNine Solutions', 'hello@cloudnine.io', 1, 'STARTUP', 23000.00, 'ACTIVE');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('MegaRetail Group', 'vendor@megaretail.com', 2, 'ENTERPRISE', 3400000.00, 'ACTIVE');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('Latin Logistics', 'compras@latinlog.br', 5, 'SMB', 156000.00, 'ACTIVE');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('Inactive Corp', 'old@inactive.com', 2, 'SMB', 12000.00, 'CHURNED');
INSERT INTO customers (customer_name, email, region_id, customer_type, total_spend, status)
VALUES ('Jane Doe', 'jane.doe@email.com', 1, 'INDIVIDUAL', 1500.00, 'ACTIVE');

-- Insert products
INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
VALUES ('Enterprise Cloud License', 'Software', 'Licenses', 50000.00, 10000.00, 999);
INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
VALUES ('Professional Support Plan', 'Services', 'Support', 12000.00, 3000.00, 999);
INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
VALUES ('Data Analytics Module', 'Software', 'Add-ons', 15000.00, 5000.00, 999);
INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
VALUES ('Security Suite', 'Software', 'Security', 25000.00, 8000.00, 999);
INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
VALUES ('Training Package', 'Services', 'Training', 5000.00, 1500.00, 999);
INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
VALUES ('API Gateway License', 'Software', 'Integration', 8000.00, 2500.00, 999);
INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
VALUES ('Consulting Day', 'Services', 'Consulting', 2500.00, 800.00, 999);
INSERT INTO products (product_name, category, subcategory, unit_price, cost_price, inventory_qty)
VALUES ('Startup Bundle', 'Software', 'Bundles', 10000.00, 4000.00, 999);

-- Insert orders (Q4 2024 - Q1 2025)
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (1, DATE '2024-10-15', DATE '2024-10-18', 125000.00, 10, 'DELIVERED', 'Invoice', 'Sarah Johnson');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (1, DATE '2024-11-20', DATE '2024-11-22', 75000.00, 5, 'DELIVERED', 'Invoice', 'Sarah Johnson');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (3, DATE '2024-10-01', DATE '2024-10-05', 200000.00, 15, 'DELIVERED', 'Wire Transfer', 'Michael Chen');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (5, DATE '2024-11-10', DATE '2024-11-15', 95000.00, 8, 'DELIVERED', 'Invoice', 'Lisa Wong');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (7, DATE '2024-12-01', DATE '2024-12-05', 340000.00, 12, 'DELIVERED', 'Wire Transfer', 'Sarah Johnson');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (2, DATE '2024-12-15', DATE '2024-12-18', 10000.00, 0, 'DELIVERED', 'Credit Card', 'James Miller');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (4, DATE '2025-01-05', NULL, 28500.00, 5, 'PROCESSING', 'Invoice', 'Michael Chen');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (6, DATE '2025-01-10', NULL, 15000.00, 0, 'PENDING', 'Credit Card', 'James Miller');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (8, DATE '2025-01-12', NULL, 42000.00, 5, 'SHIPPED', 'Invoice', 'Lisa Wong');
INSERT INTO orders (customer_id, order_date, ship_date, total_amount, discount_pct, status, payment_method, sales_rep)
VALUES (10, DATE '2025-01-15', NULL, 1500.00, 0, 'PENDING', 'Credit Card', 'James Miller');

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (1, 1, 2, 50000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (1, 2, 2, 12000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (2, 3, 3, 15000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (2, 5, 6, 5000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (3, 1, 3, 50000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (3, 4, 2, 25000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (4, 1, 1, 50000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (4, 2, 1, 12000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (4, 3, 2, 15000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (5, 1, 5, 50000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (5, 4, 3, 25000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (5, 7, 10, 2500.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (6, 8, 1, 10000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (7, 3, 1, 15000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (7, 6, 1, 8000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (7, 5, 1, 5000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (8, 8, 1, 10000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (8, 5, 1, 5000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (9, 1, 1, 50000.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (9, 7, 4, 2500.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (10, 5, 1, 5000.00);

-- Update customer totals based on orders
UPDATE customers c SET total_spend = (
    SELECT NVL(SUM(o.total_amount), 0)
    FROM orders o
    WHERE o.customer_id = c.customer_id AND o.status NOT IN ('CANCELLED', 'RETURNED')
);

COMMIT;

DBMS_OUTPUT.PUT_LINE('Inserted sample data');

-- =============================================================================
-- Step 5: Create OCI credential for GenAI access
-- =============================================================================
-- IMPORTANT: Replace placeholders with your actual OCI credentials
-- Get these from OCI Console > Identity > Users > Your User > API Keys

/*
-- Uncomment and fill in your OCI credentials:

BEGIN
    DBMS_CLOUD.CREATE_CREDENTIAL(
        credential_name => 'OCI_GENAI_CRED',
        user_ocid       => 'ocid1.user.oc1..YOUR_USER_OCID',
        tenancy_ocid    => 'ocid1.tenancy.oc1..YOUR_TENANCY_OCID',
        private_key     => '-----BEGIN PRIVATE KEY-----
YOUR_PRIVATE_KEY_CONTENT_HERE
-----END PRIVATE KEY-----',
        fingerprint     => 'xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx'
    );
    DBMS_OUTPUT.PUT_LINE('Created OCI_GENAI_CRED credential');
END;
/
*/

-- =============================================================================
-- Step 6: Create SelectAI profile
-- =============================================================================
-- IMPORTANT: Only run this after creating the credential above

/*
-- Uncomment after creating credential:

BEGIN
    DBMS_CLOUD_AI.CREATE_PROFILE(
        profile_name => 'OCI_GENAI',
        attributes   => '{
            "provider": "oci",
            "credential_name": "OCI_GENAI_CRED",
            "model": "cohere.command-r-plus",
            "oci_compartment_id": "ocid1.compartment.oc1..YOUR_COMPARTMENT_OCID",
            "object_list": [
                {"owner": "ADMIN", "name": "CUSTOMERS"},
                {"owner": "ADMIN", "name": "ORDERS"},
                {"owner": "ADMIN", "name": "ORDER_ITEMS"},
                {"owner": "ADMIN", "name": "PRODUCTS"},
                {"owner": "ADMIN", "name": "REGIONS"}
            ]
        }'
    );
    DBMS_OUTPUT.PUT_LINE('Created OCI_GENAI SelectAI profile');
END;
/

-- Enable the profile (optional - makes it the default)
BEGIN
    DBMS_CLOUD_AI.SET_PROFILE(profile_name => 'OCI_GENAI');
    DBMS_OUTPUT.PUT_LINE('Set OCI_GENAI as default profile');
END;
/
*/

-- =============================================================================
-- Step 7: Verify setup
-- =============================================================================

-- Check tables
SELECT table_name, num_rows FROM user_tables WHERE table_name IN ('CUSTOMERS', 'ORDERS', 'ORDER_ITEMS', 'PRODUCTS', 'REGIONS');

-- Check data counts
SELECT 'REGIONS' as table_name, COUNT(*) as row_count FROM regions
UNION ALL
SELECT 'CUSTOMERS', COUNT(*) FROM customers
UNION ALL
SELECT 'PRODUCTS', COUNT(*) FROM products
UNION ALL
SELECT 'ORDERS', COUNT(*) FROM orders
UNION ALL
SELECT 'ORDER_ITEMS', COUNT(*) FROM order_items;

-- Check existing AI profiles (if any)
SELECT profile_name, status FROM USER_CLOUD_AI_PROFILES;

-- =============================================================================
-- Test SelectAI (after profile is created)
-- =============================================================================
/*
-- Test natural language queries:

-- Show SQL only
SELECT AI showsql how many customers do we have by region;

-- Run and return results
SELECT AI runsql what is the total revenue by customer type;

-- Get explanation
SELECT AI explainsql show me the top 5 customers by total orders;

-- Natural language response
SELECT AI narrate what were the sales trends in Q4 2024;

-- Chat about schema
SELECT AI chat what tables are available and what do they contain;
*/

DBMS_OUTPUT.PUT_LINE('SelectAI provisioning script complete!');
DBMS_OUTPUT.PUT_LINE('Next steps:');
DBMS_OUTPUT.PUT_LINE('1. Update OCI credentials in Step 5');
DBMS_OUTPUT.PUT_LINE('2. Uncomment and run Step 5 to create credential');
DBMS_OUTPUT.PUT_LINE('3. Update compartment OCID in Step 6');
DBMS_OUTPUT.PUT_LINE('4. Uncomment and run Step 6 to create AI profile');
DBMS_OUTPUT.PUT_LINE('5. Test with SELECT AI queries in Step 7');
