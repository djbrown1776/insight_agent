import os

import duckdb
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool

load_dotenv()

st.set_page_config(page_title="Oil & Gas Production Dashboard", layout="wide")

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
SCHEMA = os.getenv("SCHEMA")

FORBIDDEN_KEYWORDS = [
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "TRUNCATE",
    "ALTER",
    "ATTACH",
    "COPY",
]


# ---------------------------------------------------------------------------
# DuckDB connection, attached to Postgres. Cached as a resource so Streamlit
# reuses one connection across reruns/interactions instead of reconnecting
# on every button click or chat message.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_connection():
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL postgres;")
    con.execute("LOAD postgres;")
    pg_attach_str = (
        f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} "
        f"host={DB_HOST} port={DB_PORT}"
    )
    con.execute(f"ATTACH '{pg_attach_str}' AS pg (TYPE postgres, READ_ONLY);")
    return con


con = get_connection()


# ---------------------------------------------------------------------------
# Agent tools (duplicated from app.py on purpose — this file has no
# dependency on it, for a standalone demo).
# ---------------------------------------------------------------------------
@tool
def get_db_schema() -> str:
    """
    Returns the schema of the Postgres database (via the 'pg' DuckDB attachment),
    including table names and column definitions. Use this first to understand
    available tables before writing SQL.
    """
    query = f"""
        SELECT table_name, column_name, data_type
        FROM pg.information_schema.columns
        WHERE table_schema = '{SCHEMA}'
        ORDER BY table_name, ordinal_position;
    """
    try:
        return con.sql(query).df().to_string(index=False)
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"


@tool
def run_duckdb_sql(query: str, save_as: str = "") -> str:
    """
    Executes a read only DuckDB SQL query. This is the ONLY tool you need for
    both fetching data and analyzing it. DuckDB is attached to Postgres as
    the 'pg' catalog, so:

        - Query live Postgres tables:   SELECT * FROM pg.<schema>.<table>
        - Query a saved local file:     SELECT * FROM read_parquet('/tmp/foo.parquet')
        - Join Postgres + local files IN ONE QUERY:
              SELECT w.*, r.county_label
              FROM pg.<schema>.wells w
              JOIN read_csv('/tmp/county_lookup.csv') r ON w.county = r.county_code

    IMPORTANT INSTRUCTIONS FOR THE AGENT:
    1. Do all grouping/aggregation/filtering/joins in SQL here rather than
       pulling data into pandas and chaining .groupby() — it's faster and
       less error-prone.
    1a. oilprod/gasprod/waterprod/year are stored as TEXT in Postgres. Always
        cast with CAST(NULLIF(col, '') AS INTEGER) before doing arithmetic —
        never sum or plot them as raw text.
    2. Only pass `save_as` (a filename like 'production_by_county.parquet')
       when you want to persist the FULL result to /tmp/ for later reuse.
       Otherwise the result is not saved and you only get the preview below.
    3. This tool returns a string preview of the top 10 rows plus row count.
    4. To plot with matplotlib/seaborn, load the saved file back with:
           df = duckdb.sql("SELECT * FROM read_parquet('/tmp/<name>.parquet')").df()
       (duckdb.sql(...) returns a relation — call .df() to get a DataFrame.)

    Args:
        query: A DuckDB SELECT statement (may reference pg.<schema>.<table> and/or local files).
        save_as: Optional filename (parquet) under /tmp/ to persist the full result to.
    """
    if any(kw in query.upper() for kw in FORBIDDEN_KEYWORDS):
        return "Error: Non-SELECT or destructive statements are forbidden."

    try:
        rel = con.sql(query)
        df = rel.df()

        if save_as:
            path = f"/tmp/{save_as}"
            rel.to_parquet(path)
            location_note = f"Full result saved to '{path}' ({len(df)} rows)."
        else:
            location_note = f"Query returned {len(df)} rows (not persisted — pass save_as to keep it)."

        return f"{location_note}\nPreview:\n{df.head(10).to_string()}"
    except Exception as e:
        return f"DuckDB Execution Error: {str(e)}"


# ---------------------------------------------------------------------------
# Agent, cached so it's built once per session, not on every rerun.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    model = LiteLLMModel(
        model_id="openrouter/qwen/qwen-2.5-72b-instruct",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_tokens=8096,
    )
    return CodeAgent(
        tools=[get_db_schema, run_duckdb_sql],
        model=model,
        additional_authorized_imports=[
            "duckdb",
            "pandas",
            "numpy",
            "matplotlib.pyplot",
            "seaborn",
            "scipy.stats",
        ],
        max_steps=12,
        verbosity_level=2,
    )


agent = get_agent()


# ---------------------------------------------------------------------------
# Cached data loaders — the fast path. Refreshing re-runs this SQL directly
# against Postgres; it does NOT call the LLM agent. Only the chat panel at
# the bottom calls the agent.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_top_operators(prod_column: str, limit: int = 5):
    """prod_column must be 'oilprod' or 'gasprod' — never pass raw user input
    here, since this builds a SQL string directly."""
    query = f"""
        SELECT w.company_name AS operator,
               SUM(CAST(NULLIF(p.{prod_column}, '') AS INTEGER)) AS total
        FROM pg.{SCHEMA}.production p
        JOIN pg.{SCHEMA}.wells w ON p.api_wellno = w.api_wellno
        GROUP BY w.company_name
        ORDER BY total DESC
        LIMIT {limit}
    """
    return con.sql(query).df()


@st.cache_data(ttl=300)
def load_yearly_trend():
    query = f"""
        SELECT CAST(NULLIF(p.year, '') AS INTEGER) AS year,
               SUM(CAST(NULLIF(p.oilprod, '') AS INTEGER)) AS total_oil,
               SUM(CAST(NULLIF(p.gasprod, '') AS INTEGER)) AS total_gas
        FROM pg.{SCHEMA}.production p
        WHERE NULLIF(p.year, '') IS NOT NULL
        GROUP BY year
        ORDER BY year
    """
    return con.sql(query).df()


def clear_dashboard_cache():
    load_top_operators.clear()
    load_yearly_trend.clear()


# ---------------------------------------------------------------------------
# Header + refresh
# ---------------------------------------------------------------------------
header_col, refresh_col = st.columns([6, 1])
with header_col:
    st.title("Oil & Gas Production Dashboard")
with refresh_col:
    st.write("")  # vertical alignment spacer
    if st.button("🔄 Refresh", use_container_width=True):
        clear_dashboard_cache()
        st.rerun()

# ---------------------------------------------------------------------------
# Top 5 oil operators / top 5 gas operators, side by side
# ---------------------------------------------------------------------------
oil_col, gas_col = st.columns(2)

with oil_col:
    st.subheader("Top 5 Oil Operators")
    oil_df = load_top_operators("oilprod")
    if oil_df.empty:
        st.info("No oil production data available.")
    else:
        fig_oil = go.Figure(
            go.Bar(x=oil_df["operator"], y=oil_df["total"], marker_color="firebrick")
        )
        fig_oil.update_layout(
            yaxis_title="Total Oil (bbl)", xaxis_title=None, margin=dict(t=10)
        )
        st.plotly_chart(fig_oil, use_container_width=True)

with gas_col:
    st.subheader("Top 5 Gas Operators")
    gas_df = load_top_operators("gasprod")
    if gas_df.empty:
        st.info("No gas production data available.")
    else:
        fig_gas = go.Figure(
            go.Bar(x=gas_df["operator"], y=gas_df["total"], marker_color="steelblue")
        )
        fig_gas.update_layout(
            yaxis_title="Total Gas (mcf)", xaxis_title=None, margin=dict(t=10)
        )
        st.plotly_chart(fig_gas, use_container_width=True)

# ---------------------------------------------------------------------------
# Oil vs. gas trend over time (oil = red, gas = blue)
# ---------------------------------------------------------------------------
st.subheader("Oil vs. Gas Production Trend")
trend_df = load_yearly_trend()
if trend_df.empty:
    st.info("No yearly production data available.")
else:
    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=trend_df["year"],
            y=trend_df["total_oil"],
            mode="lines+markers",
            name="Oil",
            line=dict(color="red"),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=trend_df["year"],
            y=trend_df["total_gas"],
            mode="lines+markers",
            name="Gas",
            line=dict(color="blue"),
        )
    )
    fig_trend.update_layout(
        xaxis_title="Year", yaxis_title="Total Production", margin=dict(t=10)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Chat panel — the slow path. Only this calls the CodeAgent.
# ---------------------------------------------------------------------------
st.subheader("Ask a question about this data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("e.g. Which wells have the steepest oil decline this year?")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            answer = str(agent.run(question))
        st.markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))
