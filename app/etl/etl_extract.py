import os
import re
import hashlib
import argparse
import pandas as pd
import pyodbc
from pathlib import Path

# Database connection details using raw string to avoid escape warning
CONN_STRING = r"Driver={ODBC Driver 17 for SQL Server};Server=(localdb)\MSSQLLocalDB;Database=booking;Trusted_Connection=yes;"

# Default tables to extract
TABLES_TO_EXTRACT = [
    "Artist",
    "Presenter",
    "Venue",
    "BlueCard",
    "Contract",
    "GeneralLedgerJournal",
    "ContractTransaction",
    "ContractCancellation",
    "AgentPayrollLog",
    "ContractArtist"
]

def get_db_connection():
    """Establishes and returns a database connection."""
    return pyodbc.connect(CONN_STRING)

def get_column_select_clause(conn, table_name):
    """Retrieves column definitions and casts geography types to string to avoid pyodbc errors."""
    cursor = conn.cursor()
    query = f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}' ORDER BY ORDINAL_POSITION"
    cursor.execute(query)
    columns = []
    for row in cursor.fetchall():
        col_name, data_type = row[0], row[1]
        if data_type.lower() == "geography":
            columns.append(f"[{col_name}].ToString() AS [{col_name}]")
        else:
            columns.append(f"[{col_name}]")
    return ", ".join(columns)

def sanitize_column_name(name):
    """Converts camelCase/PascalCase column names to Snowflake-friendly UPPER_SNAKE_CASE."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    # Replace double underscores and convert to upper case
    return s2.upper().replace('__', '_').strip('_')

def hash_sensitive_value(val):
    """Applies SHA-256 hashing to mask sensitive byte/string data (PII)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, bytes):
        return hashlib.sha256(val).hexdigest()
    return hashlib.sha256(str(val).encode('utf-8')).hexdigest()

def extract_table(table_name, limit=None, output_dir=None):
    """Extracts a table from SQL Server, processes PII and schemas, and saves as Parquet."""
    print(f"[*] Starting extraction for table: {table_name}")
    
    conn = get_db_connection()
    try:
        # Build SELECT clause handling geography types
        select_clause = get_column_select_clause(conn, table_name)
        
        # Build query
        if limit:
            query = f"SELECT TOP {limit} {select_clause} FROM [{table_name}]"
        else:
            query = f"SELECT {select_clause} FROM [{table_name}]"
            
        df = pd.read_sql_query(query, conn)
        row_count = len(df)
        print(f"[+] Extracted {row_count} rows from [{table_name}].")
        
        if row_count == 0:
            print(f"[!] Warning: Table {table_name} is empty. Skipping file creation.")
            return
            
        # 1. Sanitize Column Names
        df.columns = [sanitize_column_name(col) for col in df.columns]
        
        # 2. Handle PII / Sensitive Data Masking
        pii_columns = ["TIN", "BANK_ACCOUNT_NUMBER", "BANK_ROUTING_NUMBER"]
        for col in pii_columns:
            if col in df.columns:
                print(f"    [-] Masking PII column: {col}")
                df[col] = df[col].apply(hash_sensitive_value)
                
        # 3. Write to Parquet
        output_file = Path(output_dir) / f"{table_name.lower()}.parquet"
        df.to_parquet(output_file, index=False, compression="snappy")
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"[+] Saved {output_file} ({file_size_mb:.2f} MB).")
        
    except Exception as e:
        print(f"[X] Error extracting table {table_name}: {e}")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Enterprise Data Consolidation - Python Ingestion Script")
    parser.add_argument("--limit", type=int, default=50000, help="Limit row count per table for development/testing (set to 0 for unlimited)")
    parser.add_argument("--table", type=str, default=None, help="Extract a single specific table")
    args = parser.parse_args()
    
    limit = args.limit if args.limit > 0 else None
    
    # Establish staging directory
    workspace_dir = Path(__file__).resolve().parents[2]
    output_dir = workspace_dir / "data" / "staged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables = [args.table] if args.table else TABLES_TO_EXTRACT
    
    print("=" * 60)
    print("ENTERPRISE DATA CONSOLIDATION - ETL EXTRACTION")
    print(f"Staging Directory: {output_dir}")
    print(f"Row Limit per table: {limit or 'UNLIMITED'}")
    print("=" * 60)
    
    for t in tables:
        extract_table(t, limit=limit, output_dir=output_dir)
        
    print("\n[+] ETL Ingestion Complete!")

if __name__ == "__main__":
    main()
