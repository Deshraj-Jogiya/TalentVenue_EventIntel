import pytest
import pandas as pd
from app.etl.etl_extract import sanitize_column_name, hash_sensitive_value

def test_sanitize_column_name():
    """Verifies that camelCase and PascalCase columns are translated to UPPER_SNAKE_CASE."""
    assert sanitize_column_name("GeneralLedgerJournalId") == "GENERAL_LEDGER_JOURNAL_ID"
    assert sanitize_column_name("AgentId") == "AGENT_ID"
    assert sanitize_column_name("ECECommissionRate") == "ECE_COMMISSION_RATE"
    assert sanitize_column_name("IsBuySell") == "IS_BUY_SELL"
    assert sanitize_column_name("MailingAddress1") == "MAILING_ADDRESS1"
    assert sanitize_column_name("TIN") == "TIN"

def test_hash_sensitive_value():
    """Verifies that sensitive data is hashed deterministically using SHA-256, and nulls are handled."""
    # Test valid string
    val_str = "123-456-789"
    hash_str = hash_sensitive_value(val_str)
    assert hash_str is not None
    assert len(hash_str) == 64  # SHA-256 is 64 hex characters
    assert hash_str == hash_sensitive_value(val_str)  # Deterministic check
    
    # Test bytes input
    val_bytes = b"\x01\x02\x03\x04"
    hash_bytes = hash_sensitive_value(val_bytes)
    assert hash_bytes is not None
    assert len(hash_bytes) == 64
    assert hash_bytes == hash_sensitive_value(val_bytes)
    assert hash_bytes != hash_str  # Different input should produce different hash
    
    # Test null handling
    assert hash_sensitive_value(None) is None
    assert hash_sensitive_value(float("nan")) is None
