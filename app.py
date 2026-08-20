import os

import duckdb
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from smolagents import CodeAgent, LiteLLMModel, tool

load_dotenv()

console = Console()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
SCHEMA = os.getenv("SCHEMA")

# ---------------------------------------------------------------------------
# 1. DuckDB connection.
# ---------------------------------------------------------------------------
con = duckdb.connect(database=":memory:")
con.execute("INSTALL postgres;")
con.execute("LOAD postgres;")

PG_ATTACH_STR = (
    f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} "
    f"host={DB_HOST} port={DB_PORT}"
)

con.execute(f"ATTACH '{PG_ATTACH_STR}' AS pg (TYPE postgres, READ_ONLY);")

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
# 2. Schema Discovery Tool
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
    2. Only pass `save_as` (a filename like 'production_by_county.parquet')
       when you want to persist the FULL result to /tmp/ for later reuse
       (e.g. across multiple analysis steps or for the charting step).
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
# 4. Configure the LLM via OpenRouter
# ---------------------------------------------------------------------------
model = LiteLLMModel(
    model_id="openrouter/qwen/qwen-2.5-72b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=8096,
)

# ---------------------------------------------------------------------------
# 5. Initialize the CodeAgent
# ---------------------------------------------------------------------------
agent = CodeAgent(
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

prompt = f"""
1. Inspect the database schema to see all available tables — you should find at
   least 'wells' and 'production'. Tables live under the 'pg' catalog, schema '{SCHEMA}'
   (e.g. pg.{SCHEMA}.wells, pg.{SCHEMA}.production).
2. Using a single DuckDB SQL query, join wells and production on the API well
   number (api_wellno) to bring operator, location (county/town), and well status
   into the production data. Save the result with save_as='wells_production.parquet'.
3. For the most recent 4 years available, calculate total oil (OILPROD), gas (GAS),
   and water production by county and by operator using DuckDB SQL (GROUP BY) against
   the saved parquet file — not pandas groupby chains. To handle empty strings, Use this exact syntax for casting: CAST(NULLIF(oilprod, '') AS INTEGER).
   Do this for gas and water as well.Identify the top five producing counties/operators.
4. Using DuckDB SQL (window functions / GROUP BY, e.g. LAG() for year-over-year gaps,
   or COUNT(DISTINCT year) per well), flag wells with inconsistent or missing
   production reporting (e.g. gaps in years, wells marked active with zero reported
   production).
5. Load the relevant aggregated result into pandas (via duckdb.sql(...).df()) and
   save a bar chart of top five operators by total oil production to
   './oil_gas_production_trend.png' (save to the current working directory, DO NOT use /tmp/).
   Use DuckDB SQL to perform the aggregation and casting to integers before passing the final summarized data into Pandas for charting. Do not use Pandas `.groupby()`.
6. IMPORTANT: Format your final answer as clean, readable Markdown. Use headings,
   bullet points, and formatted code blocks for the SQL. Include a dedicated text section where you explicitly write out the notable year-over-year trends from your analysis.
   DO NOT return a raw Python dictionary.Include the exact DuckDB SQL queries you ran for Step 3 and Step 4 in your final Markdown output inside SQL code blocks. Do not hide your work.
"""

response = agent.run(prompt)

console.print("\n[bold green]--- FINAL REPORT ---[/bold green]\n")
console.print(Markdown(str(response)))
