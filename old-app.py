import os

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from smolagents import CodeAgent, LiteLLMModel, tool
from sqlalchemy import create_engine, text

load_dotenv()

console = Console()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
SCHEMA = os.getenv("SCHEMA")

# 1. Database Connection (PostgreSQL)
DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URI)


# 2. Schema Discovery Tool (Gives the agent context on tables/columns)
@tool
def get_db_schema() -> str:
    """
    Returns the schema of the PostgreSQL database, including table names and column definitions.
    Use this first to understand available tables before writing SQL.
    """
    target_schema = SCHEMA
    query = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = :schema
    ORDER BY table_name, ordinal_position;
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"schema": target_schema})

        return df.to_string(index=False)
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"


# 3. Safe SQL Execution Tool (Returns results as a Pandas DataFrame string)
@tool
def execute_sql(query: str) -> str:
    """
    Executes a read-only SQL query against the PostgreSQL database.

        IMPORTANT INSTRUCTIONS FOR THE AGENT:
        1. This tool returns a string preview of the top 10 rows.
        2. The FULL query result is automatically saved locally to '/tmp/latest_query_result.parquet'.
        3. To perform pandas analysis or create charts, you MUST load the full data in your Python code using:
           df = pd.read_parquet("/tmp/latest_query_result.parquet")

        Args:
            query: The standard PostgreSQL SELECT query to run.
    """
    # Basic guardrail against accidental destructive operations
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER"]
    if any(kw in query.upper() for kw in forbidden_keywords):
        return "Error: Non-SELECT or destructive statements are forbidden."

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            # Save raw slice locally so the CodeAgent can load it with pandas if needed
            df.to_parquet("/tmp/latest_query_result.parquet")
            return f"Query returned {len(df)} rows. Preview:\n{df.head(10).to_string()}"
    except Exception as e:
        return f"SQL Execution Error: {str(e)}"


# 4. Configure the LLM via OpenRouter
model = LiteLLMModel(
    model_id="openrouter/qwen/qwen-2.5-72b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=8096,
)

# 5. Initialize the CodeAgent
# We authorize data science libraries so the agent can write Python transformations directly
agent = CodeAgent(
    tools=[get_db_schema, execute_sql],
    model=model,
    additional_authorized_imports=[
        "pandas",
        "numpy",
        "matplotlib.pyplot",
        "seaborn",
        "scipy.stats",
    ],
    max_steps=12,  # Error Recovery
    verbosity_level=2,  # Thinking out loud
)

prompt = """
1. Inspect the database schema to see all available tables — you should find at
   least 'wells' (well master data) and 'production' (annual oil/gas/water production
   by well).
2. Join 'wells' and 'production' on the API well number (api_wellno) to bring
   operator, location (county/town), and well status into the production data.
3. For the most recent 4 years available, calculate total oil (OILPROD), gas (GAS),
   and water production by county and by operator, and identify the top producing
   counties/operators and any notable year-over-year trends.
4. Using pandas, flag wells with inconsistent or missing production reporting
   (e.g. gaps in years, wells marked active with zero reported production).
5. Save a bar chart of top counties or operators by total oil production to
   ./oil_gas_production_trend.png' (save to the current working directory, DO NOT use /tmp/).
7. IMPORTANT: Format your final answer as clean, readable Markdown. Use headings,
   bullet points, and formatted code blocks for the SQL. DO NOT return a raw Python dictionary.
"""

response = agent.run(prompt)

console.print("\n[bold green]--- FINAL REPORT ---[/bold green]\n")
console.print(Markdown(str(response)))
