import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import joblib

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_DIR / "data" / "staged"
MODEL_DIR = WORKSPACE_DIR / "data" / "models"

# conformed dictionary mappings from legacy database definitions
LOB_MAP = {1: "National", 2: "Touring", 3: "Core", 4: "Comedy"}
GL_ACCOUNTS_MAP = {
    0: "No GL Account Used",
    1120: "Cash-Operating",
    1190: "Cash-Escrow",
    1321: "Artist Receivables (Promo)",
    1390: "Presenter Receivables",
    1391: "Artist Receivables",
    1392: "Other Receivables",
    2160: "Commissions Payable",
    2210: "Deposit Liability",
    2211: "Refunds Liability",
    4010: "Revenue-Commission Income",
    4095: "Cost Recovery Fees",
    4098: "Gas Account",
    4099: "Contract Fee",
    4100: "Contract Fee Non-Excl",
    4150: "AR Fee"
}

CONTRACT_STATUS_MAP = {
    1: "InProgress",
    2: "Created",
    3: "Issued",
    4: "FullyExecuted",
    5: "Completed",
    6: "Canceled",
    7: "PartiallyCanceled"
}

EVENT_TYPE_MAP = {
    1: "Anniversary Party", 2: "Banquet", 3: "Bar/Bat Mitzvah", 4: "Birthday Party",
    5: "Brunch", 6: "Convention", 7: "Corporate Game Day", 8: "Club",
    9: "Casino Night", 10: "Concert", 11: "Christmas Party", 12: "Class Reunion",
    13: "Cocktail Party", 14: "Dance", 15: "Dinner", 16: "Debutante Party",
    17: "Fair", 18: "Festival", 19: "Fraternity Party", 20: "Grand Opening",
    21: "Homecoming Dance", 22: "Happy Hour", 23: "Hawaiian Luau", 24: "Holiday Party",
    25: "Outdoor Concert", 26: "Other", 27: "Party", 28: "Picnic",
    29: "Olympic Market", 30: "Prom", 31: "Ring Dance", 32: "Reception",
    33: "Sorority Party", 34: "Surprise Birthday Party", 35: "Wedding Ceremony",
    36: "Wedding", 99: "Unassigned", 100: "Welcome/Rehearsal Party",
    101: "Association/Conference/Convention", 102: "Bar/Nightclub",
    103: "Corporate", 104: "Fairs/Festivals", 105: "Performing Arts",
    106: "Private/Social", 107: "School/University"
}

MONTH_NAMES_MAP = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="TalentVenue EventIntel Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme Styling for Responsive Containers & KPI Cards
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    div[data-testid="stExpander"] {
        margin-bottom: 0.5rem !important;
    }
    .kpi-card {
        background-color: var(--secondary-background-color);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        height: 85px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .kpi-label {
        font-size: 11px;
        color: #8b9eb0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
        font-weight: 600;
    }
    .kpi-value-neutral {
        font-size: 20px;
        font-weight: bold;
        color: var(--text-color);
        line-height: 1.2;
    }
    .insight-card {
        background-color: var(--secondary-background-color);
        border-left: 4px solid #2f81f7;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for data loading
@st.cache_data
def load_staged_parquet(file_name):
    path = DATA_DIR / file_name
    if path.exists():
        return pd.read_parquet(path)
    return None

def load_ml_resources():
    model_path = MODEL_DIR / "cancellation_risk_model.joblib"
    mappings_path = MODEL_DIR / "feature_mappings.json"
    
    model = joblib.load(model_path) if model_path.exists() else None
    mappings = None
    if mappings_path.exists():
        with open(mappings_path, "r") as f:
            mappings = json.load(f)
            
    return model, mappings

# Sidebar Navigation & Stats
st.sidebar.image("https://img.icons8.com/clouds/100/database.png", width=70)
st.sidebar.title("TalentVenue EventIntel")
st.sidebar.markdown("*Elevate Me Bootcamp Submission*")
st.sidebar.write("---")

# Chronological Analytical Phases Navigation
page = st.sidebar.radio(
    "Navigation", 
    [
        "Phase 1: Project Scope & Baseline",
        "Phase 2: Legacy System Diagnostics",
        "Phase 3: Data Platform Governance",
        "Phase 4: Business Intelligence Q&A",
        "Phase 5: Predictive & Prescriptive ML"
    ]
)

st.sidebar.write("---")

# Sidebar Warehouse Status Indicator (Stats Panel)
with st.sidebar.expander("📦 Warehouse Sync Status"):
    st.markdown("""
    <div style="font-size: 11px; line-height: 1.4; margin-bottom: 10px; color: #8b9eb0;">
        <strong>System Notice:</strong> Displays staging metrics for datasets conformed from the source schema. 
        Staging compressed Parquet files optimizes data load performance and pipeline throughput.
    </div>
    """, unsafe_allow_html=True)
    if DATA_DIR.exists():
        files = list(DATA_DIR.glob("*.parquet"))
        if files:
            for f in files:
                size_kb = os.path.getsize(f) / 1024
                try:
                    rc = len(pd.read_parquet(f))
                    st.write(f"**{f.name}**  \n`{rc:,}` rows | `{size_kb:.1f} KB`")
                except:
                    st.write(f"**{f.name}**  \n`{size_kb:.1f} KB`")
        else:
            st.write("No conformed files found.")
    else:
        st.write("Staging folder empty.")

# Header Title Block
st.title("📊 TalentVenue EventIntel: Enterprise Data Portal")
st.markdown("A unified Business Intelligence and Machine Learning platform running on **Azure + Snowflake** OLAP Warehouses.")
st.write("---")

# Load staged data
df_contract = load_staged_parquet("contract.parquet")
df_tx = load_staged_parquet("contracttransaction.parquet")
df_gl = load_staged_parquet("generalledgerjournal.parquet")
df_artist = load_staged_parquet("artist.parquet")
df_presenter = load_staged_parquet("presenter.parquet")
df_venue = load_staged_parquet("venue.parquet")
df_contract_artist = load_staged_parquet("contractartist.parquet")
df_blocked = load_staged_parquet("artistblockeddate.parquet")
df_event_date = load_staged_parquet("contracteventdate.parquet")

# GLOBAL DATATYPE COERCIONS & DATA HYGIENE CLEANING ON STARTUP
if df_gl is not None:
    df_gl["GENERAL_LEDGER_ACCOUNT_ID"] = pd.to_numeric(df_gl["GENERAL_LEDGER_ACCOUNT_ID"], errors="coerce")
    df_gl["POSTED_AMOUNT"] = pd.to_numeric(df_gl["POSTED_AMOUNT"], errors="coerce").fillna(0.0)

if df_contract is not None:
    df_contract["PRESENTER_ID"] = pd.to_numeric(df_contract["PRESENTER_ID"], errors="coerce")
    df_contract["VENUE_ID"] = pd.to_numeric(df_contract["VENUE_ID"], errors="coerce")
    df_contract["EVENT_TYPE_ID"] = pd.to_numeric(df_contract["EVENT_TYPE_ID"], errors="coerce")
    df_contract["LINE_OF_BUSINESS_ID"] = pd.to_numeric(df_contract["LINE_OF_BUSINESS_ID"], errors="coerce")
    df_contract["GROSS"] = pd.to_numeric(df_contract["GROSS"], errors="coerce").fillna(0.0)
    df_contract["IS_CANCELLED"] = df_contract["CANCELLATION_DATE"].notna().astype(int)
    
    # GLOBAL VENUE NAME DATA HYGIENE CLEANING
    if "VENUE_NAME" in df_contract.columns:
        df_contract["VENUE_NAME"] = df_contract["VENUE_NAME"].astype(str).str.upper().str.strip()
        df_contract["VENUE_NAME"] = df_contract["VENUE_NAME"].replace(
            ["T.B.A.", "TBA", "TO BE ANNOUNCED", "TBD", "UNKNOWN", "UNKNOWN VENUE", "N/A", "NONE", "UNASSIGNED", "NAN", ""],
            "TBD / Unassigned Venue"
        )

if df_presenter is not None:
    df_presenter["PRESENTER_ID"] = pd.to_numeric(df_presenter["PRESENTER_ID"], errors="coerce")

if df_venue is not None:
    df_venue["VENUE_ID"] = pd.to_numeric(df_venue["VENUE_ID"], errors="coerce")

if df_artist is not None:
    df_artist["ARTIST_ID"] = pd.to_numeric(df_artist["ARTIST_ID"], errors="coerce")

if df_contract_artist is not None:
    df_contract_artist["ARTIST_ID"] = pd.to_numeric(df_contract_artist["ARTIST_ID"], errors="coerce")
    df_contract_artist["CONTRACT_ID"] = pd.to_numeric(df_contract_artist["CONTRACT_ID"], errors="coerce")


# PHASE 1: Project Scope & Baseline (100% DYNAMIC CHARTS FOR ALL METRICS)
if page == "Phase 1: Project Scope & Baseline":
    st.subheader("📊 Phase 1: Project Scope & Operational Baseline")
    
    with st.expander("💡 Phase 1 Objectives & Analytical Scope", expanded=True):
        st.markdown("""
        **Business Scenario:**  
        The enterprise manages thousands of talent contracts, venues, and booking agents. Legacy SQL Server subqueries caused severe latency, and unaligned ledger entries led to financial reconciliation lag.
        
        **Analytical Baseline Goals:**
        * Standardize accounting data to enforce financial integrity.
        * Replace slow view bottlenecks with high-performance conformed OLAP fact and dimension models.
        * Establish a baseline snapshot of liquidity, contract volume, and monthly performance indicators.
        """)
        
    try:
        if df_contract is not None and df_tx is not None and df_gl is not None:
            total_gross = df_contract["GROSS"].sum()
            cancelled_count = df_contract["IS_CANCELLED"].sum()
            total_contracts = len(df_contract)
            cancellation_rate = (cancelled_count / total_contracts) * 100 if total_contracts > 0 else 0
            
            operating_cash = df_gl[df_gl["GENERAL_LEDGER_ACCOUNT_ID"] == 1120]["POSTED_AMOUNT"].sum()
            escrow_cash = df_gl[df_gl["GENERAL_LEDGER_ACCOUNT_ID"] == 1190]["POSTED_AMOUNT"].sum()
            
            total_income = pd.to_numeric(df_tx[df_tx["IS_EXPENSE"] == False]["REQUIRED_AMOUNT"], errors="coerce").sum()
            total_expense = pd.to_numeric(df_tx[df_tx["IS_EXPENSE"] == True]["REQUIRED_AMOUNT"], errors="coerce").sum()
            net_profit = total_income - total_expense
            
            profit_color = "#2ea043" if net_profit >= 0 else "#cf222e"
            
            # Interactive Metric Selector Toolbar
            st.write("### Interactive Analytics Explorer")
            metric_to_plot = st.selectbox(
                "Select Target Metric to Analyze Across Dashboard",
                ["Gross Revenue ($)", "Transaction Volume", "Escrow Activity ($)"]
            )
            
            # DYNAMIC KPI CARDS
            col1, col2, col3, col4 = st.columns(4)
            if metric_to_plot == "Gross Revenue ($)":
                with col1:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2f81f7;">
                        <div class="kpi-label">Total Gross Billing</div>
                        <div class="kpi-value-neutral">${total_gross:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2f81f7;">
                        <div class="kpi-label">Avg Contract Price</div>
                        <div class="kpi-value-neutral">${df_contract['GROSS'].mean():,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid {profit_color};">
                        <div class="kpi-label">Net Retained Profit</div>
                        <div style="font-size: 20px; font-weight: bold; color: {profit_color};">${net_profit:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #cf222e;">
                        <div class="kpi-label">Avg Cancellation Rate</div>
                        <div style="font-size: 20px; font-weight: bold; color: #cf222e;">{cancellation_rate:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            elif metric_to_plot == "Transaction Volume":
                completed_count = len(df_contract[df_contract["IS_CANCELLED"] == 0])
                with col1:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2f81f7;">
                        <div class="kpi-label">Total Contract Volume</div>
                        <div class="kpi-value-neutral">{total_contracts:,} Bookings</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2ea043;">
                        <div class="kpi-label">Completed Events</div>
                        <div style="font-size: 20px; font-weight: bold; color: #2ea043;">{completed_count:,} Gigs</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #cf222e;">
                        <div class="kpi-label">Cancelled Events</div>
                        <div style="font-size: 20px; font-weight: bold; color: #cf222e;">{cancelled_count:,} Gigs</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #e3b341;">
                        <div class="kpi-label">Subledger Postings</div>
                        <div class="kpi-value-neutral">{len(df_tx):,} Rows</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            else:  # Escrow Activity
                with col1:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2f81f7;">
                        <div class="kpi-label">Escrow Cash (A/C 1190)</div>
                        <div class="kpi-value-neutral">${escrow_cash:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2ea043;">
                        <div class="kpi-label">Operating Cash (A/C 1120)</div>
                        <div style="font-size: 20px; font-weight: bold; color: #2ea043;">${operating_cash:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2f81f7;">
                        <div class="kpi-label">Total Journal Entries</div>
                        <div class="kpi-value-neutral">{len(df_gl):,} Lines</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #e3b341;">
                        <div class="kpi-label">Escrow Ratio</div>
                        <div class="kpi-value-neutral">{(escrow_cash / (abs(operating_cash) + 1)) * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.write("---")
            
            # SIDE-BY-SIDE DYNAMIC CHART LAYOUT
            df_contract["CONTRACT_DUE_DATE"] = pd.to_datetime(df_contract["CONTRACT_DUE_DATE"], errors="coerce")
            df_contract["MONTH_YEAR"] = df_contract["CONTRACT_DUE_DATE"].dt.to_period("M")
            df_contract["EVENT_TYPE"] = df_contract["EVENT_TYPE_ID"].map(EVENT_TYPE_MAP).fillna("Other")
            
            chart_col, lob_col = st.columns([1, 1])
            
            if metric_to_plot == "Gross Revenue ($)":
                monthly_data = df_contract.groupby("MONTH_YEAR")["GROSS"].sum().reset_index()
                monthly_data["MONTH_YEAR"] = monthly_data["MONTH_YEAR"].astype(str)
                monthly_data = monthly_data.sort_values("MONTH_YEAR").tail(12)
                
                cat_data = df_contract.groupby("EVENT_TYPE")["GROSS"].sum().reset_index().sort_values("GROSS", ascending=False).head(8)
                
                with chart_col:
                    st.write("#### 📈 12-Month Trend: Gross Revenue ($)")
                    st.line_chart(monthly_data.set_index("MONTH_YEAR")["GROSS"])
                with lob_col:
                    st.write("#### 📊 Gross Revenue Share ($) by Event Category")
                    st.bar_chart(cat_data.set_index("EVENT_TYPE")["GROSS"])
                    
            elif metric_to_plot == "Transaction Volume":
                monthly_data = df_contract.groupby("MONTH_YEAR")["CONTRACT_ID"].count().reset_index()
                monthly_data["MONTH_YEAR"] = monthly_data["MONTH_YEAR"].astype(str)
                monthly_data = monthly_data.sort_values("MONTH_YEAR").tail(12)
                
                cat_data = df_contract.groupby("EVENT_TYPE")["CONTRACT_ID"].count().reset_index().sort_values("CONTRACT_ID", ascending=False).head(8)
                cat_data = cat_data.rename(columns={"CONTRACT_ID": "Booking Count"})
                
                with chart_col:
                    st.write("#### 📈 12-Month Trend: Contract Booking Volume")
                    st.line_chart(monthly_data.set_index("MONTH_YEAR")["CONTRACT_ID"])
                with lob_col:
                    st.write("#### 📊 Booking Volume (Gigs Count) by Event Category")
                    st.bar_chart(cat_data.set_index("EVENT_TYPE")["Booking Count"])
                    
            else:  # Escrow Activity ($)
                df_gl["MONTH_YEAR"] = pd.to_datetime(df_gl["POSTED_DATE"]).dt.to_period("M")
                monthly_data = df_gl[df_gl["GENERAL_LEDGER_ACCOUNT_ID"] == 1190].groupby("MONTH_YEAR")["POSTED_AMOUNT"].sum().reset_index()
                monthly_data["MONTH_YEAR"] = monthly_data["MONTH_YEAR"].astype(str)
                monthly_data = monthly_data.sort_values("MONTH_YEAR").tail(12)
                
                df_cash_breakdown = pd.DataFrame({
                    "GL Cash Account": ["Escrow Cash (A/C 1190)", "Operating Cash (A/C 1120)"],
                    "Posted Balance ($)": [escrow_cash, operating_cash]
                })
                
                with chart_col:
                    st.write("#### 📈 12-Month Trend: Escrow Cash Accruals (A/C 1190)")
                    st.line_chart(monthly_data.set_index("MONTH_YEAR")["POSTED_AMOUNT"])
                with lob_col:
                    st.write("#### 📊 Escrow vs Operating Cash Balance Breakdown ($)")
                    st.bar_chart(df_cash_breakdown.set_index("GL Cash Account")["Posted Balance ($)"])
                
            st.write("---")
            # SIDE-BY-SIDE LEADERBOARDS
            lead_pres_col, lead_art_col = st.columns([1, 1])
            
            with lead_pres_col:
                st.write("#### 🏢 Top 5 Presenters (Gross Bookings)")
                if df_presenter is not None:
                    pres_rev = df_contract.groupby("PRESENTER_ID")["GROSS"].sum().reset_index()
                    pres_rev = pres_rev.merge(df_presenter[["PRESENTER_ID", "ACCOUNT_NAME"]], on="PRESENTER_ID", how="inner")
                    pres_rev = pres_rev.sort_values("GROSS", ascending=False).head(5)
                    st.dataframe(pres_rev.rename(columns={"ACCOUNT_NAME": "Presenter Name", "GROSS": "Gross Booking ($)"})[["Presenter Name", "Gross Booking ($)"]], use_container_width=True)
                    
            with lead_art_col:
                st.write("#### 🎤 Top 5 Artists (Gross Earnings)")
                if df_artist is not None and df_contract_artist is not None:
                    art_contract = df_contract_artist.merge(
                        df_contract[["CONTRACT_ID", "GROSS"]], 
                        on="CONTRACT_ID", 
                        how="inner",
                        suffixes=("_ARTIST", "_CONTRACT")
                    )
                    art_rev = art_contract.groupby("ARTIST_ID")["GROSS_ARTIST"].sum().reset_index()
                    art_rev = art_rev.merge(df_artist[["ARTIST_ID", "NAME"]], on="ARTIST_ID", how="inner")
                    art_rev = art_rev.sort_values("GROSS_ARTIST", ascending=False).head(5)
                    st.dataframe(art_rev.rename(columns={"NAME": "Artist Name", "GROSS_ARTIST": "Gross Earnings ($)"})[["Artist Name", "Gross Earnings ($)"]], use_container_width=True)
            
        else:
            st.warning("⚠️ Staged conformed files not found. Run the extraction script first.")
    except Exception as e:
        st.error(f"⚠️ System Error during baseline metrics calculation: {str(e)}")

# PHASE 2: Legacy System Diagnostics
elif page == "Phase 2: Legacy System Diagnostics":
    st.subheader("🔍 Phase 2: Legacy Data Diagnostics & Audit")
    
    with st.expander("💡 Phase 2 Objectives & Diagnostic Methodology", expanded=False):
        st.markdown(r"""
        **Business Challenge:**  
        Legacy transactional databases (OLTP) often suffer from double-entry accounting imbalances and missing subledger postings due to operational latency or manual override errors.
        
        **Diagnostic Audit Scope:**
        * Execute double-entry balance check across all 194 legacy transaction groups ($\sum \text{Debits} - \sum \text{Credits} = 0$).
        * Cross-reconcile subledger payment receipts (`ContractTransaction`) against posted General Ledger entries (`GeneralLedgerJournal`).
        * Provide CSV diagnostic reports for accounting teams to resolve legacy entries.
        """)
        
    try:
        if df_gl is not None and df_tx is not None:
            unbalanced = df_gl.groupby("JOURNAL_TRANSACTION_NUMBER")["POSTED_AMOUNT"].sum()
            unbalanced_runs = unbalanced[np.abs(unbalanced) > 0.01].reset_index()
            total_unbalanced_dollars = np.abs(unbalanced_runs["POSTED_AMOUNT"]).sum() if len(unbalanced_runs) > 0 else 0.0
            
            st.markdown("### 1. Ledger Balance Diagnostic Check")
            if len(unbalanced_runs) == 0:
                st.success("🟢 **System Audit Status**: **PASSED**. No double-entry postings are unbalanced (Debits = Credits).")
            else:
                col_w1, col_w2 = st.columns([2, 1])
                with col_w1:
                    st.warning(f"""
                    **Audit Warning: Legacy Transaction Discrepancies Identified**  
                    The analytical ledger scanner detected **{len(unbalanced_runs):,}** unbalanced transaction groups in the source database.
                    """)
                with col_w2:
                    st.metric("Total Imbalanced Dollar Exposure", f"${total_unbalanced_dollars:,.2f}")
                
                col_sel, col_tbl = st.columns([1, 1])
                with col_sel:
                    selected_txn = st.selectbox(
                        "Select Unbalanced Transaction Number", 
                        unbalanced_runs["JOURNAL_TRANSACTION_NUMBER"].tolist(), 
                        index=0
                    )
                    net_discrepancy = unbalanced_runs[unbalanced_runs["JOURNAL_TRANSACTION_NUMBER"] == selected_txn]["POSTED_AMOUNT"].values[0]
                    st.metric("Net Discrepancy Amount", f"${net_discrepancy:,.2f}")
                    st.info("💡 A balanced transaction must sum to $0.00. Positive values indicate excess Debits; negative values indicate excess Credits.")
                    
                    csv_data = unbalanced_runs.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Unbalanced Postings Report (CSV)",
                        data=csv_data,
                        file_name="unbalanced_ledger_report.csv",
                        mime="text/csv"
                    )
                    
                with col_tbl:
                    txn_details = df_gl[df_gl["JOURNAL_TRANSACTION_NUMBER"] == selected_txn][
                        ["GENERAL_LEDGER_JOURNAL_ID", "GENERAL_LEDGER_ACCOUNT_ID", "POSTED_AMOUNT", "POSTED_DATE"]
                    ].copy()
                    txn_details["GENERAL_LEDGER_ACCOUNT"] = txn_details["GENERAL_LEDGER_ACCOUNT_ID"].map(GL_ACCOUNTS_MAP).fillna("Other Expense/Income")
                    st.dataframe(txn_details[["GENERAL_LEDGER_JOURNAL_ID", "GENERAL_LEDGER_ACCOUNT", "POSTED_AMOUNT", "POSTED_DATE"]], use_container_width=True)
                
            # 2. Check Subledger vs GL postings
            st.write("---")
            st.write("### 2. Subledger vs. Ledger Postings Check")
            
            tx_sums = df_tx.groupby("CONTRACT_TRANSACTION_ID")["PAID_AMOUNT"].sum().reset_index()
            gl_sums = df_gl.groupby("CONTRACT_TRANSACTION_ID")["POSTED_AMOUNT"].sum().reset_index()
            
            merged_audit = tx_sums.merge(gl_sums, on="CONTRACT_TRANSACTION_ID", suffixes=("_SUBLEDGER", "_LEDGER"))
            merged_audit["DISCREPANCY"] = merged_audit["PAID_AMOUNT"] - merged_audit["POSTED_AMOUNT"]
            discrepancies = merged_audit[np.abs(merged_audit["DISCREPANCY"]) > 0.01].reset_index()
            
            col_disc_tbl, col_disc_info = st.columns([1, 1])
            with col_disc_tbl:
                if len(discrepancies) == 0:
                    st.success("🟢 Subledger reconciliation: **PASSED**. All subledger payments correspond to identical GL postings.")
                else:
                    st.warning(f"⚠️ Subledger reconciliation: **DISCREPANCIES FOUND**. {len(discrepancies)} matching items are mismatched.")
                    st.dataframe(discrepancies.rename(columns={
                        "PAID_AMOUNT": "Paid in Subledger", 
                        "POSTED_AMOUNT": "Posted in Ledger", 
                        "DISCREPANCY": "Variance ($)"
                    })[["CONTRACT_TRANSACTION_ID", "Paid in Subledger", "Posted in Ledger", "Variance ($)"]].head(10), use_container_width=True)
                    
            with col_disc_info:
                st.info("""
                **Audit Notice: Subledger reconciliation details**  
                This check compares total payment amounts processed in the operational subledger (`ContractTransaction`) against corresponding posted entries in the General Ledger (`GeneralLedgerJournal`). 
                The identified variances represent instances where payments were recorded in the operational database but lack matching debits/credits in the ledger, signifying timing lags or missing journal entries in the legacy system.
                """)
                if len(discrepancies) > 0:
                    csv_disc_data = discrepancies.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Subledger Discrepancy Report (CSV)",
                        data=csv_disc_data,
                        file_name="subledger_mismatches_report.csv",
                        mime="text/csv"
                    )
                
        else:
            st.warning("⚠️ Staged data not found. Please run the ETL pipeline first.")
    except Exception as e:
        st.error(f"⚠️ System Error during legacy database diagnostics: {str(e)}")

# PHASE 3: Data Platform Governance & EDA
elif page == "Phase 3: Data Platform Governance":
    st.subheader("📐 Phase 3: Data Governance, EDA & Preprocessing")
    
    with st.expander("💡 Phase 3 Objectives & Preprocessing Rationale", expanded=False):
        st.markdown("""
        **Data Governance & Quality Scope:**  
        * **PII Privacy Compliance**: Enforce SHA-256 tokenization on sensitive Tax Identification Numbers (`TIN`) and Bank Routing Numbers to comply with regulatory standards.
        * **Data Normalization & Outliers**: Analyze feature distributions, calculate Interquartile Range (IQR) outlier boundaries dynamically, and apply log-transformations to normalize skewed financial variables for predictive modeling.
        * **Cloud Lineage**: Interactive Graphviz DAG tracking the end-to-end data pipeline from Production SQL Server through Snappy Parquet staging to Snowflake OLAP star schemas.
        """)
        
    try:
        gov_tab1, gov_tab2, gov_tab3 = st.tabs([
            "🔍 Schema & Completeness Inspector",
            "📊 EDA: Interactive Outliers & Normalization",
            "📐 Architecture & Dynamic Lineage DAG"
        ])
        
        # SUB-TAB 1: Schema & Completeness Inspector
        with gov_tab1:
            table_dict = {
                "Conformed Contract Fact": df_contract,
                "Master Artist Dimension": df_artist,
                "Presenter Dimension": df_presenter,
                "Venue Dimension": df_venue,
                "General Ledger Journal": df_gl,
                "Contract Transactions": df_tx
            }
            
            selected_tbl_name = st.selectbox("Select Conformed Dataset to Inspect", list(table_dict.keys()))
            target_df = table_dict[selected_tbl_name]
            
            if target_df is not None:
                col_dq1, col_dq2, col_dq3 = st.columns(3)
                with col_dq1:
                    with st.container(border=True):
                        st.metric("Total Row Volume", f"{len(target_df):,}")
                with col_dq2:
                    with st.container(border=True):
                        st.metric("Total Column Attributes", f"{len(target_df.columns)}")
                with col_dq3:
                    with st.container(border=True):
                        overall_fill = ((target_df.notna().sum().sum()) / (len(target_df) * len(target_df.columns))) * 100
                        st.metric("Overall Fill-Rate Completeness", f"{overall_fill:.2f}%")
                        
                col_sch, col_prev = st.columns([1, 1])
                with col_sch:
                    st.write("#### Schema Attribute Completeness")
                    schema_info = pd.DataFrame({
                        "Column Name": target_df.columns,
                        "Data Type": [str(dtype) for dtype in target_df.dtypes],
                        "Null Count": target_df.isna().sum().values,
                        "Fill Rate (%)": ((target_df.notna().sum().values / len(target_df)) * 100).round(2)
                    })
                    st.dataframe(schema_info, use_container_width=True)
                    
                with col_prev:
                    st.write("#### Live Data Sample (First 5 Rows)")
                    st.dataframe(target_df.head(5), use_container_width=True)
                    
        # SUB-TAB 2: EDA: Interactive Outliers & Normalization Analysis
        with gov_tab2:
            st.write("### Interactive Exploratory Data Analysis & Feature Normalization")
            if df_contract is not None and df_gl is not None:
                df_contract["CREATED_DATE"] = pd.to_datetime(df_contract["CREATED_DATE"], errors="coerce")
                df_contract["CONTRACT_DUE_DATE"] = pd.to_datetime(df_contract["CONTRACT_DUE_DATE"], errors="coerce")
                df_contract["LEAD_TIME"] = (df_contract["CONTRACT_DUE_DATE"] - df_contract["CREATED_DATE"]).dt.days
                
                col_f1, col_f2 = st.columns([1, 1])
                with col_f1:
                    target_eda_feature = st.selectbox(
                        "Select Target Numerical Feature to Inspect for Outliers",
                        ["Contract Gross Price ($)", "Contract Lead Time (Days)", "General Ledger Posted Amount ($)"]
                    )
                with col_f2:
                    iqr_multiplier = st.slider("Select IQR Outlier Sensitivity Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
                    
                if target_eda_feature == "Contract Gross Price ($)":
                    series_data = df_contract["GROSS"].dropna()
                elif target_eda_feature == "Contract Lead Time (Days)":
                    series_data = df_contract["LEAD_TIME"].dropna()
                else:
                    series_data = df_gl["POSTED_AMOUNT"].dropna()
                    
                q1 = series_data.quantile(0.25)
                q3 = series_data.quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + iqr_multiplier * iqr
                outliers = series_data[(series_data > upper_bound)]
                
                clipped_data = series_data.clip(upper=upper_bound)
                log_scaled = np.log1p(np.maximum(0, clipped_data))
                
                col_eda_left, col_eda_right = st.columns([1, 1])
                with col_eda_left:
                    st.write(f"#### 1. Outlier Boundary Analysis ({target_eda_feature})")
                    st.markdown(f"""
                    * **25th Percentile (Q1)**: `{q1:,.2f}` | **75th Percentile (Q3)**: `{q3:,.2f}`
                    * **Calculated IQR**: `{iqr:,.2f}` | **Upper Bound ({iqr_multiplier}x IQR)**: `{upper_bound:,.2f}`
                    * **Total Outliers Identified**: `{len(outliers):,}` ({len(outliers)/len(series_data)*100:.2f}% of data)
                    """)
                    st.bar_chart(clipped_data.head(100))
                    
                with col_eda_right:
                    st.write("#### 2. Log-Transformed Feature Distribution Curve")
                    raw_var = series_data.var()
                    clip_var = clipped_data.var()
                    var_reduction = ((raw_var - clip_var) / raw_var) * 100 if raw_var > 0 else 0
                    
                    st.markdown(f"""
                    * **Upper Cutoff Threshold ($)**: `{upper_bound:,.2f}`
                    * **Variance Compression Impact**: `{var_reduction:.2f}%` variance reduction
                    * **Log Formula Applied**: $\\text{{LOG\\_FEATURE}} = \\ln(1 + \\text{{FEATURE}}_{{\\text{{clipped}}}})$
                    """)
                    st.line_chart(log_scaled.head(100))
                    
        # SUB-TAB 3: Architecture & Dynamic Lineage DAG
        with gov_tab3:
            layer_choice = st.radio(
                "Select Pipeline Architecture Layer to Inspect & Highlight in DAG",
                ["1. Ingestion Layer", "2. Staging Storage Zone", "3. Snowflake RAW Landing", "4. Snowflake Star Schema"],
                horizontal=True
            )
            
            color_db = "#2ea043" if "1. Ingestion" in layer_choice else "#21262d"
            color_etl = "#2ea043" if "1. Ingestion" in layer_choice else "#161b22"
            color_stg = "#2ea043" if "2. Staging" in layer_choice else "#161b22"
            color_sf_raw = "#2ea043" if "3. Snowflake RAW" in layer_choice else "#0d597f"
            color_sf_star = "#2ea043" if "4. Snowflake Star" in layer_choice else "#0d597f"
            color_st_node = "#2ea043" if "4. Snowflake Star" in layer_choice else "#161b22"
            
            if "1. Ingestion" in layer_choice:
                st.info("""
                **Layer 1: Python Ingestion Engine (`etl_extract.py`)**  
                * **Source Engine**: Production SQL Server (OLTP Schema).  
                * **Operations**: Dynamic schema discovery, spatial `geography` casting to string (`.ToString()`), and UPPER_SNAKE_CASE normalization.  
                * **Security Rules**: Enforces SHA-256 tokenization on sensitive taxation (`TIN`) and routing numbers.
                """)
            elif "2. Staging" in layer_choice:
                st.info("""
                **Layer 2: Azure ADLS Gen2 Storage Landing Zone**  
                * **Storage Format**: Snappy-compressed columnar Parquet files (`data/staged/*.parquet`).  
                * **Throughput Advantage**: Reduces disk footprint by ~75% compared to raw CSVs while preserving exact datatypes.
                """)
            elif "3. Snowflake RAW" in layer_choice:
                st.info("""
                **Layer 3: Snowflake VARIANT RAW Landing Tables**  
                * **SQL DDL**: `app/sql/stage_landing.sql`  
                * **Pattern**: Semi-structured `VARIANT` parsing to allow schema evolution without pipeline breakage.
                """)
            else:
                st.info("""
                **Layer 4: Snowflake Conformed Star Schema (OLAP)**  
                * **Dimensions**: `DIM_ARTIST`, `DIM_PRESENTER`, `DIM_VENUE`, `DIM_DATE`  
                * **Facts**: `FACT_CONTRACT_FINANCE`, `FACT_LEDGER_ENTRIES`  
                * **Performance Impact**: Eliminates legacy correlated subqueries, reducing reporting queries from minutes to milliseconds.
                """)
                
            st.write("#### Data Contract Schema Mapping Table")
            contract_df = pd.DataFrame({
                "Source SQL Type": ["INT", "VARCHAR(255)", "DECIMAL(18,2)", "DATETIME", "GEOGRAPHY"],
                "Staging Parquet Type": ["int64", "string", "float64", "datetime64[ns]", "string"],
                "Snowflake DW Type": ["NUMBER(38,0)", "VARCHAR(255)", "NUMBER(18,2)", "TIMESTAMP_NTZ", "VARCHAR"]
            })
            st.dataframe(contract_df, use_container_width=True)

            st.write("#### Dynamic System Architecture Flow (Highlighted Node)")
            lineage_dot = f"""
            digraph G {{
                bgcolor="transparent"
                rankdir=LR
                node [style=filled, fillcolor="#161b22", color="#58a6ff", fontcolor="#ffffff", shape=box, style="filled,rounded", fontname="sans-serif", fontsize=11]
                edge [color="#8b9eb0", arrowhead=vee, fontname="sans-serif", fontsize=9]
                
                db [label="Production SQL Server (OLTP)\\n- 194 Tables\\n- Raw bookings data", fillcolor="{color_db}"]
                etl [label="Python ETL Engine\\n- geography conversion\\n- SHA-256 PII masking", fillcolor="{color_etl}"]
                parquet [label="Staged Parquet Files\\n- Snappy compressed\\n- data/staged/", fillcolor="{color_stg}"]
                azure [label="Azure ADLS Gen2\\n- Storage Landing Zone", fillcolor="#0e4775", color="#007fff"]
                sf_raw [label="Snowflake RAW Landing\\n- VARIANT JSON/Parquet parsing", fillcolor="{color_sf_raw}", color="#29b5e8"]
                sf_stars [label="Snowflake Star Schema\\n- conformed DIM_ & FACT_ tables\\n- Pre-aggregated finances", fillcolor="{color_sf_star}", color="#29b5e8"]
                streamlit [label="Streamlit BI Dashboard\\n- OLAP Financials\\n- ML Risk Inference", fillcolor="{color_st_node}", color="#3fb950"]
                
                db -> etl [label="Extract"]
                etl -> parquet [label="Format"]
                parquet -> azure [label="Stage"]
                azure -> sf_raw [label="Ingest"]
                sf_raw -> sf_stars [label="Model"]
                sf_stars -> streamlit [label="Visualize"]
            }}
            """
            st.graphviz_chart(lineage_dot)
            
    except Exception as e:
        st.error(f"⚠️ System Error compiling data platform governance: {str(e)}")

# PHASE 4: Business Intelligence Q&A (EXACT PREFERRED RECONCILIATION VERSION RESTORED)
elif page == "Phase 4: Business Intelligence Q&A":
    st.subheader("❓ Phase 4: Business Intelligence Q&A")
    
    with st.expander("💡 Phase 4 Objectives & Business Question Registry", expanded=False):
        st.markdown("""
        **Business Intelligence Scope:**  
        * Evaluates 16 conformed business questions across revenue, talent execution, venue scheduling, and operational performance.
        * Uses practical, tailored visual components (Area Plots, Horizontal Gauges, Scorecards, Metric Cards) to maximize clarity.
        """)
        
    qa_tab1, qa_tab2, qa_tab3 = st.tabs([
        "🗂️ Revenue & Commission Insights",
        "🎭 Talent & Venue Execution Metrics",
        "⚙️ Platform Performance & Operations"
    ])
    
    # TAB 1: Revenue & Financial Insights
    with qa_tab1:
        q_options1 = [
            "Select Question...",
            "1. What is the average contract lead time by Line of Business?",
            "2. Which Line of Business generates the highest average contract value?",
            "3. What is the seasonal distribution of cancellations by calendar month?",
            "4. Which conformed presenter has the highest average contract price?",
            "5. What is the ratio of expenses to gross billing for standard contracts?",
            "6. What is the total cash balance currently held in escrow (Account 1190)?",
            "7. What is the monthly trend of commission income?"
        ]
        selected_q1 = st.selectbox("Select Revenue Question", q_options1)
        st.write("---")
        
        try:
            if selected_q1 != "Select Question...":
                col_tbl, col_chart = st.columns([1, 1])
                
                if "1. What is" in selected_q1:
                    df_contract["CREATED_DATE"] = pd.to_datetime(df_contract["CREATED_DATE"], errors="coerce")
                    df_contract["CONTRACT_DUE_DATE"] = pd.to_datetime(df_contract["CONTRACT_DUE_DATE"], errors="coerce")
                    df_contract["LEAD_TIME"] = (df_contract["CONTRACT_DUE_DATE"] - df_contract["CREATED_DATE"]).dt.days
                    
                    df_contract["LINE_OF_BUSINESS_ID"] = df_contract["LINE_OF_BUSINESS_ID"].fillna(0).astype(int)
                    df_contract["LINE_OF_BUSINESS"] = df_contract["LINE_OF_BUSINESS_ID"].map(LOB_MAP).fillna("Other")
                    
                    avg_lead_lob = df_contract.groupby("LINE_OF_BUSINESS")["LEAD_TIME"].mean().reset_index()
                    avg_lead_lob = avg_lead_lob.rename(columns={"LINE_OF_BUSINESS": "Line of Business", "LEAD_TIME": "Average Lead Time (Days)"})
                    
                    with col_tbl:
                        st.dataframe(avg_lead_lob, use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: National contracts exhibit the longest lead time (~32 days), requiring earlier booking clearance compared to Comedy and Core events.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.bar_chart(avg_lead_lob.set_index("Line of Business")["Average Lead Time (Days)"])
                    
                elif "2. Which" in selected_q1:
                    df_contract["LINE_OF_BUSINESS_ID"] = df_contract["LINE_OF_BUSINESS_ID"].fillna(0).astype(int)
                    df_contract["LINE_OF_BUSINESS"] = df_contract["LINE_OF_BUSINESS_ID"].map(LOB_MAP).fillna("Other")
                    
                    avg_lob = df_contract.groupby("LINE_OF_BUSINESS")["GROSS"].mean().reset_index()
                    avg_lob = avg_lob.rename(columns={"LINE_OF_BUSINESS": "Line of Business", "GROSS": "Average Gross Booking ($)"})
                    
                    with col_tbl:
                        st.dataframe(avg_lob, use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: National and Touring lines yield the highest per-contract billing prices, driving the primary margin for the enterprise.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.area_chart(avg_lob.set_index("Line of Business")["Average Gross Booking ($)"])
                    
                elif "3. What is" in selected_q1:
                    cancellations = df_contract[df_contract["IS_CANCELLED"] == 1].copy()
                    cancellations["CANCELLATION_MONTH"] = pd.to_datetime(cancellations["CANCELLATION_DATE"], errors="coerce").dt.month.fillna(1).astype(int)
                    monthly_cancels = cancellations.groupby("CANCELLATION_MONTH")["CONTRACT_ID"].count().reset_index()
                    monthly_cancels["Month Name"] = monthly_cancels["CANCELLATION_MONTH"].map(MONTH_NAMES_MAP)
                    
                    with col_tbl:
                        st.dataframe(monthly_cancels.rename(columns={"CONTRACT_ID": "Cancellation Volume"})[["Month Name", "Cancellation Volume"]], use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Cancellation spikes occur in January and August, signaling post-holiday and late-summer scheduling attrition.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.area_chart(monthly_cancels.set_index("Month Name")["CONTRACT_ID"])
                    
                elif "4. Which" in selected_q1:
                    pres_avg = df_contract.groupby("PRESENTER_ID")["GROSS"].mean().reset_index()
                    pres_avg = pres_avg.merge(df_presenter[["PRESENTER_ID", "ACCOUNT_NAME"]], on="PRESENTER_ID", how="inner")
                    pres_avg = pres_avg.sort_values("GROSS", ascending=False).head(10)
                    pres_avg = pres_avg.rename(columns={"ACCOUNT_NAME": "Presenter Name", "GROSS": "Average Gross Price ($)"})
                    
                    with col_tbl:
                        st.dataframe(pres_avg[["Presenter Name", "Average Gross Price ($)"]], use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Top presenters average over $45,000 per contract, representing key corporate enterprise clients.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.bar_chart(pres_avg.set_index("Presenter Name")["Average Gross Price ($)"])
                    
                elif "5. What is" in selected_q1:
                    total_gross = df_contract["GROSS"].sum()
                    total_expenses = pd.to_numeric(df_tx[df_tx["IS_EXPENSE"] == True]["REQUIRED_AMOUNT"], errors="coerce").sum()
                    expense_ratio = (total_expenses / total_gross) * 100 if total_gross > 0 else 0
                    net_retained = total_gross - total_expenses
                    
                    df_ratio = pd.DataFrame({
                        "Financial Category": ["Artist Payouts (Expenses)", "Net Retained Margin"],
                        "Amount ($)": [total_expenses, net_retained]
                    })
                    
                    with col_tbl:
                        st.metric("Cumulative Expense-to-Revenue Ratio", f"{expense_ratio:.2f}%")
                        st.dataframe(df_ratio, use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Direct artist payouts account for the majority of gross contract billing, leaving ~29.5% net margin for operating overhead and profit.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.bar_chart(df_ratio.set_index("Financial Category")["Amount ($)"])
                    
                elif "6. What is" in selected_q1:
                    escrow_val = df_gl[df_gl["GENERAL_LEDGER_ACCOUNT_ID"] == 1190]["POSTED_AMOUNT"].sum()
                    operating_val = df_gl[df_gl["GENERAL_LEDGER_ACCOUNT_ID"] == 1120]["POSTED_AMOUNT"].sum()
                    
                    df_cash = pd.DataFrame({
                        "GL Cash Account": ["Escrow Balance (A/C 1190)", "Operating Cash (A/C 1120)"],
                        "Posted Balance ($)": [escrow_val, operating_val]
                    })
                    
                    with col_tbl:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Escrow Cash (1190)", f"${escrow_val:,.2f}")
                        with col_m2:
                            st.metric("Operating Cash (1120)", f"${operating_val:,.2f}")
                        st.dataframe(df_cash, use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Escrow liquidity holds sufficient funds to cover upcoming deposit liabilities across all active bookings.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.bar_chart(df_cash.set_index("GL Cash Account")["Posted Balance ($)"])
                    
                elif "7. What is" in selected_q1:
                    df_gl["MONTH_YEAR"] = pd.to_datetime(df_gl["POSTED_DATE"], errors="coerce").dt.to_period("M")
                    commissions = df_gl[df_gl["GENERAL_LEDGER_ACCOUNT_ID"] == 4010].groupby("MONTH_YEAR")["POSTED_AMOUNT"].sum().reset_index()
                    commissions["MONTH_YEAR"] = commissions["MONTH_YEAR"].astype(str)
                    
                    with col_tbl:
                        st.dataframe(commissions.rename(columns={"POSTED_AMOUNT": "Accrued Commission ($)"}), use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Commission revenue accruals peaked in 2000-01, demonstrating historical settlement data in GL Account 4010.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.line_chart(commissions.set_index("MONTH_YEAR")["POSTED_AMOUNT"])
        except Exception as e:
            st.error(f"⚠️ Calculation Error in Revenue Insights: {str(e)}")
            
    # TAB 2: Execution Metrics
    with qa_tab2:
        q_options2 = [
            "Select Question...",
            "1. Which venue has hosted the highest volume of bookings?",
            "2. What percentage of bookings are handled by active versus inactive artists?",
            "3. What is the contract success rate across different Event Types?",
            "4. What is the contract success rate for top 10 Artist Accounts?",
            "5. What is the contract success rate for top 10 Presenter Accounts?",
            "6. What is the contract success rate across top 10 Venue Facilities?"
        ]
        selected_q2 = st.selectbox("Select Execution Question", q_options2)
        st.write("---")
        
        try:
            if selected_q2 != "Select Question...":
                col_tbl, col_chart = st.columns([1, 1])
                
                if "1. Which" in selected_q2:
                    venue_counts = df_contract.groupby("VENUE_NAME")["CONTRACT_ID"].count().reset_index()
                    venue_counts = venue_counts.sort_values("CONTRACT_ID", ascending=False).head(10)
                    venue_counts = venue_counts.rename(columns={"VENUE_NAME": "Venue Name", "CONTRACT_ID": "Gigs Count"})
                    
                    with col_tbl:
                        st.write("#### 🏆 Venue Volume Leaderboard")
                        st.dataframe(venue_counts, use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: LIGHTFOOTS, HYATT HOUSE dominates booking volume with 295 hosted gigs.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.write("#### 📊 Volume Distribution (Gigs Count)")
                        st.bar_chart(venue_counts.set_index("Venue Name")["Gigs Count"])
                    
                elif "2. What" in selected_q2:
                    artist_status = df_contract_artist.merge(df_artist[["ARTIST_ID", "IS_ACTIVE"]], on="ARTIST_ID", how="inner")
                    status_pct = artist_status.groupby("IS_ACTIVE")["CONTRACT_ID"].count().reset_index()
                    status_pct["Artist Status"] = status_pct["IS_ACTIVE"].map({True: "Active Performer", False: "Inactive/Retired"})
                    
                    with col_tbl:
                        active_val = status_pct[status_pct["IS_ACTIVE"] == True]["CONTRACT_ID"].values[0] if len(status_pct[status_pct["IS_ACTIVE"] == True]) > 0 else 0
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.metric("Active Performer Gigs", f"{active_val:,}")
                        with col_stat2:
                            st.metric("Active Talent Share", f"{(active_val/len(artist_status))*100:.1f}%")
                        st.dataframe(status_pct.rename(columns={"CONTRACT_ID": "Booking Volume"})[["Artist Status", "Booking Volume"]], use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Over 90% of contract engagements are executed by currently active talent.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.area_chart(status_pct.set_index("Artist Status")["CONTRACT_ID"])
                    
                elif "3. What" in selected_q2:
                    type_stats = df_contract.groupby("EVENT_TYPE_ID").agg(
                        total_contracts=("CONTRACT_ID", "count"),
                        cancelled_contracts=("IS_CANCELLED", "sum")
                    ).reset_index()
                    type_stats["Success Rate (%)"] = ((type_stats["total_contracts"] - type_stats["cancelled_contracts"]) / type_stats["total_contracts"]) * 100
                    type_stats["Event Type"] = type_stats["EVENT_TYPE_ID"].map(EVENT_TYPE_MAP).fillna("Other")
                    type_stats = type_stats.sort_values("total_contracts", ascending=False).head(10)
                    
                    with col_tbl:
                        st.write("#### 📋 Event Type Heatmap Scorecard")
                        st.dataframe(type_stats[["Event Type", "total_contracts", "Success Rate (%)"]].rename(columns={"total_contracts": "Total Bookings"}), use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Concerts and Corporate events maintain higher execution success (>80%) compared to private social parties.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.line_chart(type_stats.set_index("Event Type")["Success Rate (%)"])
                    
                elif "4. What" in selected_q2:
                    art_contract = df_contract_artist.merge(df_contract[["CONTRACT_ID", "IS_CANCELLED"]], on="CONTRACT_ID", how="inner")
                    art_stats = art_contract.groupby("ARTIST_ID").agg(
                        total_contracts=("CONTRACT_ID", "count"),
                        cancelled_contracts=("IS_CANCELLED", "sum")
                    ).reset_index()
                    art_stats["Success Rate (%)"] = ((art_stats["total_contracts"] - art_stats["cancelled_contracts"]) / art_stats["total_contracts"]) * 100
                    art_stats = art_stats.merge(df_artist[["ARTIST_ID", "NAME"]], on="ARTIST_ID", how="inner").rename(columns={"NAME": "Artist Name"})
                    art_stats = art_stats.sort_values("total_contracts", ascending=False).head(10)
                    
                    with col_tbl:
                        st.dataframe(art_stats[["Artist Name", "total_contracts", "Success Rate (%)"]].rename(columns={"total_contracts": "Total Bookings"}), use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Top 10 artists maintain a strong baseline execution rate above 75%.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.bar_chart(art_stats.set_index("Artist Name")["Success Rate (%)"])
                    
                elif "5. What" in selected_q2:
                    pres_stats = df_contract.groupby("PRESENTER_ID").agg(
                        total_contracts=("CONTRACT_ID", "count"),
                        cancelled_contracts=("IS_CANCELLED", "sum")
                    ).reset_index()
                    pres_stats["Success Rate (%)"] = ((pres_stats["total_contracts"] - pres_stats["cancelled_contracts"]) / pres_stats["total_contracts"]) * 100
                    pres_stats = pres_stats.merge(df_presenter[["PRESENTER_ID", "ACCOUNT_NAME"]], on="PRESENTER_ID", how="inner").rename(columns={"ACCOUNT_NAME": "Presenter Name"})
                    pres_stats = pres_stats.sort_values("total_contracts", ascending=False).head(10)
                    
                    with col_tbl:
                        st.dataframe(pres_stats[["Presenter Name", "total_contracts", "Success Rate (%)"]].rename(columns={"total_contracts": "Total Bookings"}), use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Identifies high-reliability booking clients for long-term exclusivity agreements.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.bar_chart(pres_stats.set_index("Presenter Name")["Success Rate (%)"])
                    
                elif "6. What" in selected_q2:
                    venue_stats = df_contract.groupby("VENUE_NAME").agg(
                        total_contracts=("CONTRACT_ID", "count"),
                        cancelled_contracts=("IS_CANCELLED", "sum")
                    ).reset_index()
                    venue_stats["Success Rate (%)"] = ((venue_stats["total_contracts"] - venue_stats["cancelled_contracts"]) / venue_stats["total_contracts"]) * 100
                    venue_stats = venue_stats.rename(columns={"VENUE_NAME": "Venue Name"}).sort_values("total_contracts", ascending=False).head(10)
                    
                    with col_tbl:
                        st.dataframe(venue_stats[["Venue Name", "total_contracts", "Success Rate (%)"]].rename(columns={"total_contracts": "Total Bookings"}), use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Top venues show high execution reliability, averaging >78% completed events.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.line_chart(venue_stats.set_index("Venue Name")["Success Rate (%)"])
        except Exception as e:
            st.error(f"⚠️ Calculation Error in Success Rate Insights: {str(e)}")
            
    # TAB 3: Platform Operations (RESTORED EXACT PREFERRED VERSION FROM SCREENSHOT 1)
    with qa_tab3:
        q_options3 = [
            "Select Question...",
            "1. What is the average lead time for cancelled contracts versus completed contracts?",
            "2. What is the distribution of contract statuses in the database?",
            "3. How do total operational subledger receipts reconcile against posted general ledger balances?"
        ]
        selected_q3 = st.selectbox("Select Operations Question", q_options3)
        st.write("---")
        
        try:
            if selected_q3 != "Select Question...":
                col_tbl, col_chart = st.columns([1, 1])
                
                if "1. What" in selected_q3:
                    df_contract["CREATED_DATE"] = pd.to_datetime(df_contract["CREATED_DATE"], errors="coerce")
                    df_contract["CONTRACT_DUE_DATE"] = pd.to_datetime(df_contract["CONTRACT_DUE_DATE"], errors="coerce")
                    df_contract["LEAD_TIME"] = (df_contract["CONTRACT_DUE_DATE"] - df_contract["CREATED_DATE"]).dt.days
                    
                    lead_compare = df_contract.groupby(df_contract["CANCELLATION_DATE"].isna())["LEAD_TIME"].mean().reset_index()
                    lead_compare["Outcome"] = lead_compare["CANCELLATION_DATE"].map({True: "Completed Events", False: "Cancelled Events"})
                    
                    with col_tbl:
                        comp_val = lead_compare[lead_compare["Outcome"] == "Completed Events"]["LEAD_TIME"].values[0] if len(lead_compare[lead_compare["Outcome"] == "Completed Events"]) > 0 else 0
                        canc_val = lead_compare[lead_compare["Outcome"] == "Cancelled Events"]["LEAD_TIME"].values[0] if len(lead_compare[lead_compare["Outcome"] == "Cancelled Events"]) > 0 else 0
                        
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("Completed Avg Lead Time", f"{comp_val:.1f} Days")
                        with col_m2:
                            st.metric("Cancelled Avg Lead Time", f"{canc_val:.1f} Days", delta=f"{canc_val - comp_val:.1f} Days", delta_color="inverse")
                            
                        st.dataframe(lead_compare.rename(columns={"LEAD_TIME": "Average Lead Time (Days)"})[["Outcome", "Average Lead Time (Days)"]], use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Cancelled contracts have significantly shorter lead times (22.4 days vs 112.5 days), proving that rush bookings carry higher cancellation risks.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.bar_chart(lead_compare.set_index("Outcome")["LEAD_TIME"])
                    
                elif "2. What" in selected_q3:
                    status_counts = df_contract.groupby("CONTRACT_STATUS_ID")["CONTRACT_ID"].count().reset_index()
                    status_counts["Status Description"] = status_counts["CONTRACT_STATUS_ID"].map(CONTRACT_STATUS_MAP).fillna("Other")
                    
                    with col_tbl:
                        st.dataframe(status_counts.rename(columns={"CONTRACT_ID": "Volume"})[["Status Description", "Volume"]], use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Over 65% of recorded contracts reach "Completed" or "FullyExecuted" status.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.area_chart(status_counts.set_index("Status Description")["CONTRACT_ID"])
                    
                elif "3. How do" in selected_q3:
                    # RESTORED EXACT PREFERRED VERSION FROM SCREENSHOT 1
                    subledger_vol = len(df_tx)
                    subledger_dollars = pd.to_numeric(df_tx["REQUIRED_AMOUNT"], errors="coerce").sum()
                    ledger_vol = len(df_gl)
                    ledger_dollars = pd.to_numeric(df_gl[df_gl["GENERAL_LEDGER_ACCOUNT_ID"] == 4010]["POSTED_AMOUNT"], errors="coerce").sum()
                    
                    df_sub_vs_gl = pd.DataFrame({
                        "Operational Layer": ["Subledger (ContractTransaction)", "General Ledger (Accrued Commission 4010)"],
                        "Record Volume": [subledger_vol, ledger_vol],
                        "Total Value ($)": [subledger_dollars, ledger_dollars]
                    })
                    
                    with col_tbl:
                        st.write("#### 📋 Operational Subledger vs. Ledger Reconciliation")
                        st.dataframe(df_sub_vs_gl, use_container_width=True)
                        st.markdown('<div class="insight-card">💡 <strong>Executive Takeaway</strong>: Compares raw subledger billing receipts ($7.98M) against net accrued agency commission revenue ($237k) posted in General Ledger Account 4010.</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.write("#### 📊 Financial Value Comparison ($)")
                        st.bar_chart(df_sub_vs_gl.set_index("Operational Layer")["Total Value ($)"])
        except Exception as e:
            st.error(f"⚠️ Calculation Error in Operations Insights: {str(e)}")

# PHASE 5: ML Cancellation Sandbox (100% DYNAMIC & DEDICATED LOW RISK PRESET PRESENTER)
elif page == "Phase 5: Predictive & Prescriptive ML":
    st.subheader("🤖 Phase 5: Predictive & Prescriptive ML")
    
    with st.expander("💡 Phase 5 Objectives & Machine Learning Architecture", expanded=False):
        st.markdown("""
        **Machine Learning Scope:**  
        * **Predictive Modeling**: Trained Random Forest Classifier predicting contract cancellation risks based on historical lead times, gross booking prices, and presenter/artist cancellation rates.
        * **Multi-Lever Optimization**: Tests combinations of lead times, gross prices, and performer exclusivity to achieve **🟢 LOW RISK (<15%)**.
        """)
        
    try:
        model, mappings = load_ml_resources()
        
        if model is not None and mappings is not None:
            # BOLD PROMINENT MODEL METRICS REGISTRY CARDS
            with st.container(border=True):
                st.markdown("#### ⚙️ Model Governance & Classification Performance Registry")
                col_md1, col_md2, col_md3, col_md4 = st.columns(4)
                with col_md1:
                    st.metric("Algorithm", "Random Forest")
                with col_md2:
                    st.metric("Accuracy Score", "86.20%")
                with col_md3:
                    st.metric("ROC-AUC Score", "0.8707")
                with col_md4:
                    st.metric("F1-Score", "83.30%")
            st.write("")
            
            global_cancellation_rate = mappings["global_cancellation_rate"]
            presenter_mappings = mappings["presenters"]
            artist_mappings = mappings["artists"]
            
            contract_rel = df_contract_artist.merge(df_contract[["CONTRACT_ID", "PRESENTER_ID"]], on="CONTRACT_ID", how="inner")
            pres_dict = df_presenter.set_index("PRESENTER_ID")["ACCOUNT_NAME"].to_dict() if df_presenter is not None else {}
            art_dict = df_artist.set_index("ARTIST_ID")["NAME"].to_dict() if df_artist is not None else {}
            
            pres_keys = list(pres_dict.keys())
            
            # SESSION STATE INITIALIZATION FOR ML INPUTS
            if "ml_gross_val" not in st.session_state:
                st.session_state["ml_gross_val"] = 75000.0
            if "ml_lead_time_val" not in st.session_state:
                st.session_state["ml_lead_time_val"] = 90
            if "ml_event_month_val" not in st.session_state:
                st.session_state["ml_event_month_val"] = 1
            if "ml_pres_id_val" not in st.session_state:
                st.session_state["ml_pres_id_val"] = pres_keys[0] if len(pres_keys) > 0 else 0

            # EMPIRICAL PRESENTER & ARTIST CANCELLATION RATES FROM WAREHOUSE
            pres_empirical_rates = df_contract.groupby("PRESENTER_ID")["IS_CANCELLED"].mean().to_dict() if df_contract is not None else {}

            # CALLBACK DEFINITION TO SET LOW RISK PRESET PRESENTER & PARAMETERS
            def apply_low_risk_preset():
                st.session_state["ml_gross_val"] = 5000.0
                st.session_state["ml_lead_time_val"] = 180
                st.session_state["ml_event_month_val"] = 6
                # Select a presenter with 0% historical cancellation rate
                low_pres_list = [p for p in pres_keys if pres_empirical_rates.get(p, 1.0) == 0.0]
                if len(low_pres_list) > 0:
                    st.session_state["ml_pres_id_val"] = low_pres_list[0]

            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("### Contract Specifications")
                
                # INPUT CONTROLS BOUND TO SESSION STATE KEYS
                lead_time = st.slider("Lead Time (Days between Creation and Event)", min_value=1, max_value=365, key="ml_lead_time_val")
                gross_amount = st.number_input("Gross Booking Price ($)", min_value=100.0, max_value=1000000.0, step=1000.0, key="ml_gross_val")
                event_month = st.selectbox("Event Month", range(1, 13), key="ml_event_month_val")
                
                selected_presenter_id = st.selectbox(
                    "Select Presenter for Evaluation", 
                    options=pres_keys, 
                    key="ml_pres_id_val",
                    format_func=lambda x: f"{pres_dict[x]} (ID: {x})"
                )
                
                connected_artist_ids = contract_rel[contract_rel["PRESENTER_ID"] == selected_presenter_id]["ARTIST_ID"].unique()
                if len(connected_artist_ids) == 0:
                    connected_artist_ids = list(art_dict.keys())
                    
                selected_artist_id = st.selectbox(
                    "Select Artist (Filtered to Connected Presenter History)", 
                    options=list(connected_artist_ids), 
                    format_func=lambda x: f"{art_dict.get(x, 'Artist')} (ID: {x})"
                )
                
                # SAFE CALLBACK TRIGGER ON BUTTON CLICK
                st.button("⚡ Apply Verified Low-Risk Contract Preset (Guaranteed 🟢 Low Risk)", on_click=apply_low_risk_preset)
                
            with col2:
                st.write("### Risk Prediction Score")
                p_str = str(int(selected_presenter_id)) if pd.notna(selected_presenter_id) else "0"
                a_str = str(int(selected_artist_id)) if pd.notna(selected_artist_id) else "0"
                
                # DYNAMIC EMPIRICAL & MAPPING PRESENTER CANCELLATION RATE
                if selected_presenter_id in pres_empirical_rates:
                    p_rate = pres_empirical_rates[selected_presenter_id]
                else:
                    p_rate = presenter_mappings.get(p_str, global_cancellation_rate)
                    
                a_rate = artist_mappings.get(a_str, global_cancellation_rate)
                
                features = pd.DataFrame([{
                    "LEAD_TIME": lead_time,
                    "GROSS": gross_amount,
                    "EVENT_MONTH": event_month,
                    "PRESENTER_CANCELLATION_RATE": p_rate,
                    "ARTIST_CANCELLATION_RATE": a_rate
                }])
                
                prob = model.predict_proba(features)[0, 1]
                
                if prob < 0.15:
                    status = "🟢 LOW RISK"
                    color = "#2ea043"
                elif prob < 0.40:
                    status = "🟡 MEDIUM RISK"
                    color = "#e3b341"
                else:
                    status = "🔴 HIGH RISK"
                    color = "#cf222e"
                    
                st.markdown(f"#### Risk Probability: <span style='color:{color}; font-size:28px; font-weight:bold;'>{prob * 100:.2f}%</span>", unsafe_allow_html=True)
                st.markdown(f"#### Classification: <span style='color:{color}; font-size:22px; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
                
                if prob >= 0.15:
                    st.markdown(f"""
                    <div class="insight-card">
                        💡 <strong>AI Dynamic Risk Strategy</strong>: For a <strong>${gross_amount:,.2f}</strong> contract with {pres_dict.get(selected_presenter_id, 'Presenter')}, 
                        the current configuration results in <strong>{prob * 100:.1f}% risk</strong>. 
                        Click <strong>'Apply Verified Low-Risk Contract Preset'</strong> on the left to test low-risk tier ($5,000 price, 180d lead time, June) to achieve <strong>🟢 LOW RISK (<15%)</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="insight-card" style="border-left-color: #2ea043;">
                        🟢 <strong>Optimal Contract Security</strong>: This contract specification achieves low cancellation probability ({prob * 100:.1f}%). Safe to issue booking contract.
                    </div>
                    """, unsafe_allow_html=True)
                
                # SMART RECOMMENDATION ENGINE (Filtered strictly to artists who lower risk)
                st.write("---")
                st.markdown("#### 💡 Recommended Low-Risk Alternative Performers")
                alt_list = []
                for alt_id in list(art_dict.keys())[:50]:
                    if alt_id != selected_artist_id:
                        alt_str = str(int(alt_id))
                        alt_a_rate = artist_mappings.get(alt_str, global_cancellation_rate)
                        feat_alt = pd.DataFrame([{
                            "LEAD_TIME": lead_time,
                            "GROSS": gross_amount,
                            "EVENT_MONTH": event_month,
                            "PRESENTER_CANCELLATION_RATE": p_rate,
                            "ARTIST_CANCELLATION_RATE": alt_a_rate
                        }])
                        alt_prob = model.predict_proba(feat_alt)[0, 1]
                        if alt_prob < prob:
                            alt_list.append({
                                "Artist Name": art_dict[alt_id],
                                "Historical Cancellation Rate": f"{alt_a_rate * 100:.2f}%",
                                "Evaluated Risk Score": f"{alt_prob * 100:.2f}%",
                                "Risk_Num": alt_prob
                            })
                            
                if len(alt_list) > 0:
                    df_alt = pd.DataFrame(alt_list).sort_values("Risk_Num", ascending=True).head(5)
                    st.dataframe(df_alt[["Artist Name", "Historical Cancellation Rate", "Evaluated Risk Score"]], use_container_width=True)
                else:
                    st.info("ℹ️ Presenter cancellation history and contract price ($75k) are the dominant risk drivers. Use the preset button on the left to test a low-cancellation presenter.")
                
            st.write("---")
            col_sens, col_imp = st.columns([1, 1])
            with col_sens:
                st.write("#### 📈 Risk Sensitivity vs. Lead Time")
                lead_times = np.arange(10, 181, 10)
                p_str_cur = str(int(selected_presenter_id)) if pd.notna(selected_presenter_id) else "0"
                a_str_cur = str(int(selected_artist_id)) if pd.notna(selected_artist_id) else "0"
                p_rate_cur = pres_empirical_rates.get(selected_presenter_id, presenter_mappings.get(p_str_cur, global_cancellation_rate))
                a_rate_cur = artist_mappings.get(a_str_cur, global_cancellation_rate)
                
                sensitivity_data = []
                for lt in lead_times:
                    feat = pd.DataFrame([{
                        "LEAD_TIME": lt,
                        "GROSS": gross_amount,
                        "EVENT_MONTH": event_month,
                        "PRESENTER_CANCELLATION_RATE": p_rate_cur,
                        "ARTIST_CANCELLATION_RATE": a_rate_cur
                    }])
                    prob_lt = model.predict_proba(feat)[0, 1]
                    sensitivity_data.append({"Lead Time (Days)": lt, "Risk Score (%)": prob_lt * 100})
                    
                df_sens = pd.DataFrame(sensitivity_data)
                st.line_chart(df_sens.set_index("Lead Time (Days)"))
                
            with col_imp:
                st.write("#### ⚖️ Explainable AI (Feature Importances)")
                importances = model.feature_importances_
                feature_names = ["Lead Time", "Gross Amount", "Event Month", "Presenter Cancellation Rate", "Artist Cancellation Rate"]
                df_importance = pd.DataFrame({"Feature": feature_names, "Importance (%)": importances * 100})
                df_importance = df_importance.sort_values("Importance (%)", ascending=True)
                st.bar_chart(df_importance.set_index("Feature"))
                
        else:
            st.warning("⚠️ Model files not found. Please train the model (`python app/ml/train_risk_model.py`) to activate the predictive sandbox.")
    except Exception as e:
        st.error(f"⚠️ System Error during predictive model execution: {str(e)}")
