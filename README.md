# InsightAgent 🤖📊

An autonomous Business Intelligence and Data Engineering agent for querying and reporting on oil and gas production data.

## What It Does
* **Data Ingestion:** Streams multi-year public well and production records directly into PostgreSQL.
* **Fast Query Engine:** Attaches DuckDB to PostgreSQL in read-only mode to run high-speed in-memory analytical SQL without loading large datasets into Python memory.
* **Autonomous Analysis:** Uses an AI CodeAgent to inspect schemas, execute DuckDB queries, flag reporting anomalies, and generate summary reports.
* **Interactive UI:** A Streamlit dashboard with KPI charts and an interactive AI data assistant.

## Tech Stack
Python, DuckDB, PostgreSQL, Hugging Face `smolagents`, LiteLLM, Streamlit, Plotly.
