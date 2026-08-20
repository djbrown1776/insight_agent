import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", 1000)

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_engine():
    """Create and return a PostgreSQL SQLAlchemy engine using .env credentials."""
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}",
        isolation_level="AUTOCOMMIT",
    )


def close_engine(engine):
    """Close all pooled connections."""
    engine.dispose()
    print("Disconnected")


# ---------------------------------------------------------------------------
# Core query runner
# ---------------------------------------------------------------------------


def run_query(engine, query, limit=None, max_rows=1000_000, force=False):
    """Run raw SQL and return a DataFrame. If limit is set, wraps the
    query to only pull that many rows. Otherwise guards against
    accidentally pulling more than max_rows."""
    if limit is not None:
        query = f"SELECT * FROM ({query.strip().rstrip(';')}) AS sub LIMIT {limit}"

    with engine.connect() as conn:
        result = conn.execute(text(query))
        cols = result.keys()
        rows = result.fetchmany(max_rows + 1)
        if len(rows) > max_rows and not force:
            raise MemoryError(
                f"Query returned more than {max_rows} rows. "
                f"Add limit=N, or call with force=True if you really need the full pull."
            )
        return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# In-flight query management (check / cancel running queries in Postgres)
# ---------------------------------------------------------------------------


def list_inflight(engine):
    """Show currently running queries under your user in PostgreSQL."""
    query = """--sql
        SELECT pid, query_start AS starttime, query AS text
        FROM pg_stat_activity
        WHERE usename = current_user
          AND state = 'active'
    """
    return run_query(engine, query)


def cancel_all_inflight(engine):
    """Cancel all active queries under your user (excluding this call itself)."""
    with engine.connect() as conn:
        my_pid = conn.execute(text("SELECT pg_backend_pid()")).scalar()

    df = list_inflight(engine)
    df = df[df["pid"] != my_pid]

    if len(df) == 0:
        print("Nothing else is running. You're clear.")
        return 0

    with engine.connect() as conn:
        for pid in df["pid"]:
            conn.execute(text(f"SELECT pg_cancel_backend({pid});"))
    print(f"Canceled {len(df)} queries.")
    return len(df)
