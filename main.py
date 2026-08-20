# %%
from db import (
    cancel_all_inflight,
    close_engine,
    get_engine,
    list_inflight,
    run_query,
)

engine = get_engine()
print("Engine created!")

# %%
# ---------------------------------------------------------------------------
# Main query
# ---------------------------------------------------------------------------
query = """--sql
      select *
      from raw_geology.train
      where 1 = 1
      limit 100
"""
df = run_query(engine, query, limit=None)
# df.loc[:, df.columns.str.startswith('first36')] # column starts with
# df.shape[0] # Row and column count
# df[(df['cost_center'] != df['well_cc']) & ~(df['cost_center'].isna() & df['well_cc'].isna())] # miss match
# df[df['api14'].duplicated(keep=False)] # duplicates
# df['well_cc'].isnull().sum() # count of nulls
df.head(10)

# ---------------------------------------------------------------------------
# Check / kill anything running under your user
# ---------------------------------------------------------------------------
list_inflight(engine)

# %%
cancel_all_inflight(engine)

# %%
# ---------------------------------------------------------------------------
# Done — close connection
# ---------------------------------------------------------------------------
close_engine(engine)
