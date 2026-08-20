import io
import os
import zipfile

import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
SCHEMA = os.getenv("SCHEMA", "raw_oil_gas")

WELLS_URL = "https://www.dec.ny.gov/fs/data/wellDOS.zip"
PRODUCTION_URL_TEMPLATE = (
    "https://www.dec.ny.gov/fs/projects/oilandgasdata/Prod{year}.zip"
)
YEAR_START = 2000
YEAR_END = 2024  # last year NYSDEC has published as of writing

HEADERS = {"User-Agent": "Mozilla/5.0 (data ingestion script)"}


def clean_columns(columns: pd.Index) -> list[str]:
    return [
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
        for c in columns
    ]


def download_zip(url: str) -> zipfile.ZipFile:
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(resp.content))


def csv_members(zf: zipfile.ZipFile) -> list[str]:
    return [n for n in zf.namelist() if n.lower().endswith(".csv")]


def load_csv_member(
    zf: zipfile.ZipFile,
    member: str,
    table_name: str,
    engine,
    raw_conn,
    cursor,
    create_table: bool,
) -> int:
    """Stream one CSV member from an open zip into a Postgres table via COPY.
    Returns the number of rows loaded."""

    # 1. Sample the CSV to infer columns / dtypes and (optionally) create the table
    with zf.open(member) as f:
        sample_df = pd.read_csv(f, nrows=1000, low_memory=False)
    sample_df.columns = clean_columns(sample_df.columns)

    if create_table:
        # Land everything as TEXT — inferring numeric dtypes from a 1,000-row sample
        # is unsafe: full files often contain sentinel strings like "NA" in columns
        # that look numeric in the sample, which breaks COPY with a cast error, and
        # numeric inference can also strip leading zeros from ID-like columns
        # (e.g. api_wellno). Cast to proper types downstream once you can see the
        # full data, in a view or a transform step.
        cols_sql = ", ".join(f'"{c}" TEXT' for c in sample_df.columns)
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.{table_name} CASCADE;"))
            conn.execute(text(f"CREATE TABLE {SCHEMA}.{table_name} ({cols_sql});"))

    # 2. Re-open the member and stream the full contents via COPY, skipping the header row
    with zf.open(member) as raw:
        text_stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        header_line = text_stream.readline()
        col_count = len(header_line.strip().split(","))

        copy_sql = f"""
            COPY {SCHEMA}.{table_name} FROM STDIN WITH (
                FORMAT csv,
                DELIMITER ',',
                NULL ''
            );
        """
        cursor.copy_expert(sql=copy_sql, file=text_stream)
    raw_conn.commit()

    # Row count for logging
    with engine.begin() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.{table_name};"))
        return result.scalar()


def main():
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))

    raw_conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    cursor = raw_conn.cursor()

    # --- Table 1: wells master data (single file, ~40k wells) ---
    print(f"\n--- Downloading wells master data ---")
    wells_zip = download_zip(WELLS_URL)
    wells_members = [n for n in csv_members(wells_zip) if "wellspublic" in n.lower()]
    if not wells_members:
        raise RuntimeError(
            f"No wellspublic.csv found in {WELLS_URL}. Contents: {wells_zip.namelist()}"
        )

    rows = load_csv_member(
        wells_zip,
        wells_members[0],
        table_name="wells",
        engine=engine,
        raw_conn=raw_conn,
        cursor=cursor,
        create_table=True,
    )
    print(f"✓ Loaded {SCHEMA}.wells: {rows:,} rows.")

    # --- Table 2: annual production data (one zip per year, appended into one table) ---
    print(f"\n--- Downloading production data, {YEAR_START}-{YEAR_END} ---")
    first_year_loaded = False
    for year in range(YEAR_START, YEAR_END + 1):
        url = PRODUCTION_URL_TEMPLATE.format(year=year)
        try:
            prod_zip = download_zip(url)
        except requests.exceptions.HTTPError as e:
            print(f"  {year}: skipped ({e})")
            continue

        for member in csv_members(prod_zip):
            rows = load_csv_member(
                prod_zip,
                member,
                table_name="production",
                engine=engine,
                raw_conn=raw_conn,
                cursor=cursor,
                create_table=not first_year_loaded,
            )
            first_year_loaded = True
            print(f"  {year} ({member}): table now has {rows:,} total rows.")

    cursor.close()
    raw_conn.close()
    print(
        f"\nAll NYSDEC oil & gas data loaded into schema '{SCHEMA}': tables 'wells' and 'production'."
    )
    print("Join key: wells.api_wellno <-> production.api_wellno")


if __name__ == "__main__":
    main()
