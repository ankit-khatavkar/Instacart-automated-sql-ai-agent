# Generic SQL Insight Agent

A reusable, natural-language analytics agent. Ask a plain English question about
any SQL database, and it writes its own SQL, runs it, and explains the results —
with an automatically-chosen table or chart.

Built on a fully free stack: any SQL database (DuckDB, Postgres, MySQL, Redshift,
etc.) + Google Gemini's free API tier for the "thinking" part.

---

## How it works, in one sentence

You ask a question → Gemini reads your database schema and decides what SQL to
write → your code runs that SQL against the real database → Gemini reads the
results and explains them in plain English → the result is displayed as text,
a table, and (when it makes sense) a chart.

Nothing in this pipeline is scripted or pre-written per question — the AI writes
new SQL for every question, so this works on any database schema without
modification.

---

## Files

| File | Purpose |
|---|---|
| `generic_sql_agent.py` | The agent itself — schema discovery, the SQL tool, the Gemini loop, and visualization |
| `instacart.duckdb` (example) | A local DuckDB database file — swap this for any other database via `DB_CONNECTION_STRING` |

---

## One-time setup

### 1. Install dependencies

```bash
pip install google-genai sqlalchemy pandas duckdb duckdb-engine matplotlib jinja2
```

What each library is for:

| Library | Why it's needed |
|---|---|
| `google-genai` | Talks to the Gemini API |
| `sqlalchemy` | Universal database connector — lets the same code talk to DuckDB, Postgres, MySQL, Redshift, etc. |
| `pandas` | Stores and displays query results |
| `duckdb` | The database engine itself (if using DuckDB) |
| `duckdb-engine` | Teaches SQLAlchemy how to speak DuckDB specifically |
| `matplotlib` | Draws bar charts |
| `jinja2` | Required by pandas' `.style` (used for clean table rendering) |

### 2. Get a free Gemini API key

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with a Google account
3. Click **Create API key** (no credit card required)
4. Copy the key (looks like `AIzaSy...`)

**Never commit this key to a file or share a screenshot containing it.** Always
set it as an environment variable, not hardcoded in the script.

### 3. Point the script at your database

Open `generic_sql_agent.py` and find this line:

```python
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING", "duckdb:///instacart.duckdb")
```

The only thing that ever needs to change between projects is the connection
string. SQLAlchemy reads the prefix (`duckdb://`, `postgresql://`, etc.) and
automatically knows how to talk to that specific database — nothing else in
the script needs to change.

| Database | Example connection string |
|---|---|
| DuckDB | `duckdb:///instacart.duckdb` |
| SQLite | `sqlite:///mydata.db` |
| Postgres / Neon | `postgresql://user:password@host:5432/dbname` |
| MySQL | `mysql+pymysql://user:password@host:3306/dbname` |
| Redshift | `redshift+psycopg2://user:password@cluster-host:5439/dbname` (needs `pip install sqlalchemy-redshift psycopg2-binary`) |

You can either edit the default in the script, or set it as an environment
variable each session (recommended, since it avoids editing the file):

```python
os.environ["DB_CONNECTION_STRING"] = "postgresql://user:password@host:5432/dbname"
```

---

## Running it — step by step (Jupyter)

**Step 1 — Install packages (once per environment):**
```python
!pip install google-genai sqlalchemy pandas duckdb duckdb-engine matplotlib jinja2
```

**Step 2 — Set your working folder and API key, before importing the script:**
```python
import os

os.chdir("/path/to/folder/containing/generic_sql_agent.py/and/your/database")
os.environ["GEMINI_API_KEY"] = "AIzaSy...your-real-key..."

print("Now working in:", os.getcwd())
```

> **Why order matters:** importing the script actually *runs* it top to bottom
> once, including the lines that check for the API key and connect to the
> database. If you import before setting these, the script will fail or
> connect to the wrong thing. Always set the folder and key *first*.

**Step 3 — Import and ask a question:**
```python
import pandas as pd
import generic_sql_agent as agent

agent.show_answer("Which department has the highest reorder rate?")
```

This single call handles everything: asks the AI, runs the SQL, prints the
plain-English answer, and draws a chart or stat card if the data is verified
to match the answer (see **Visualization logic** below).

---

## Enabling / Disabling SQL script printing (Verbose mode)

By default, the agent runs quietly — you only see the final plain-English
answer, table, and chart. The underlying `[tool call] run_sql(...)` lines
(showing exactly what SQL the AI wrote) are suppressed.

```python
VERBOSE = False   # default — no SQL/tool-call noise printed
```

If you want to see the AI's reasoning and the exact SQL it wrote (useful for
debugging, or to explain "how" the agent got an answer), turn it on before
asking a question:

```python
agent.VERBOSE = True
agent.show_answer("your question here")
```

Turn it back off (`agent.VERBOSE = False`) once you're done debugging, to
keep future output clean.

---

## Visualization logic

After every answer, the agent tries to display the underlying data as
visually as possible — but only when a visualization would actually be
useful and verified, never a forced/misleading one. The decision happens
in this order:

1. **Verification check (runs first).** Before anything is drawn, the
   script confirms that the data it's about to visualize is consistent
   with the numbers the AI actually wrote in its plain-English answer. If
   they don't reasonably agree, the visualization is skipped and a small
   note is shown instead. Full details in **Verification: catching stale
   or mismatched chart data**, below. The text answer itself is never
   affected by this check.

2. **Bar chart** — tried next, once verification passes. Only used if the
   result has **2 or more rows**, at least one text/label column, and at
   least one number column. (A single-row result is intentionally excluded
   here — a bar chart with one bar isn't useful. That case falls through
   to option 3 instead.)

3. **Stat cards** — tried only if the bar chart didn't apply, and only if
   the result is **exactly one row**. Each numeric column in that row
   becomes its own card. For example, a basket-size summary with mean,
   median, min, max, and standard deviation would render as 5 separate
   cards side by side. Numbers are lightly formatted for readability:
   commas for whole numbers, percentages for values between 0 and 1,
   and 2 decimal places otherwise.

4. **Skip entirely** — if the result matches neither shape (for example,
   many rows with multiple numeric columns and no single clear label
   column, or some other unusual shape), no chart is drawn at all. The
   plain-English answer is still shown either way, so you're never left
   with zero output — just no forced or misleading chart.

This logic lives in these functions inside `generic_sql_agent.py`, if you
ever want to inspect or extend it:
- `_chart_matches_answer(df, answer_text)` — the verification check (step 1)
- `_render_bar_chart(df, question)` — step 2
- `_render_stat_cards(df)` — step 3
- `show_answer(question)` — the function that ties all of the above together, in order

---

## Verification: catching stale or mismatched chart data

**The problem this solves:** the AI agent is allowed to run more than one
SQL query per question (for example, it may double-check its own result
with a follow-up query). The chart/card visualization is built from
whichever query result looks most like the "real" answer — but that's a
heuristic, not a guarantee. Occasionally, a stray follow-up query (e.g. a
sanity-check `SELECT COUNT(*) ...`) can end up being what gets charted,
while the AI's *written* answer is still correct. Without a check, this
shows up as a chart or stat card with a wrong number, even though the text
above it is right.

**The fix:** before drawing any chart or card, the script checks whether
the numbers it's about to visualize actually appear in the AI's own text
answer. If they don't reasonably agree, the visualization is skipped —
you'll see a small note instead:

> *(Visualization skipped — couldn't confirm it matches the answer above.
> The numbers in the text are the ones to trust.)*

**How the matching works, concretely:**
1. `_extract_numbers_from_text(answer)` — pulls every number out of the
   AI's written answer, handling commas (`1,234`), percentages (`83%`),
   and plain decimals (`10.09`).
2. `_extract_numbers_from_df(df)` — pulls every numeric value out of the
   data about to be charted.
3. `_numbers_match(a, b)` — compares two numbers with a small tolerance,
   so `10.09` (as rounded in the text) still counts as matching `10.088883`
   (the raw, unrounded database value).
4. `_chart_matches_answer(df, answer)` — requires at least 50% of the
   chart's numbers to be found somewhere in the text. If fewer than half
   match, the chart is skipped.

**What this guarantees, and what it doesn't:**
- ✅ The plain-English text answer is **always shown**, completely
  unaffected by this check — it is displayed before any chart logic runs.
- ✅ This reliably catches the specific failure mode above: a chart
  showing numbers that have nothing to do with what the AI actually said.
- ⚠️ This is a heuristic safety net, not a mathematical proof of
  correctness. It's possible (though less likely) for a chart to
  coincidentally share enough numbers with unrelated text and pass when it
  shouldn't, or for a valid chart to be skipped if the AI's text
  summarizes rather than restates every number. When in doubt, the text
  answer is always the one to trust — the chart is a convenience on top
  of it, not a separate source of truth.

You can test this logic without spending any Gemini API calls, using fake
data:

```python
import pandas as pd
import generic_sql_agent as agent

# a mismatch that should be caught (chart skipped)
fake_answer = "Average: 10.09 items, Median: 8, Minimum: 1, Maximum: 145"
fake_df = pd.DataFrame([{"count_star()": 0}])
print(agent._chart_matches_answer(fake_df, fake_answer))  # False

# a genuine match (chart allowed through)
fake_df2 = pd.DataFrame([{"mean": 10.09, "median": 8, "min": 1, "max": 145}])
print(agent._chart_matches_answer(fake_df2, fake_answer))  # True
```

---

### Testing the chart/card rendering itself, without using API calls

Since Gemini's free tier has a limited number of requests per day, you can
also test how the charts/cards look using fake data, without asking the AI
anything:

```python
import pandas as pd
import generic_sql_agent as agent

# test stat cards (single row, multiple numbers)
agent.LAST_RESULT_DF = pd.DataFrame([{
    "mean_basket_size": 10.3, "median_basket_size": 9, "min_basket_size": 1,
    "max_basket_size": 145, "stddev_basket_size": 6.8
}])
agent._render_stat_cards(agent.LAST_RESULT_DF)

# test bar chart (multiple rows, one label + one number column)
agent.LAST_RESULT_DF = pd.DataFrame([
    {"product_name": "Banana", "total_orders": 491291},
    {"product_name": "Organic Strawberries", "total_orders": 275577},
    {"product_name": "Limes", "total_orders": 146660},
])
agent._render_bar_chart(agent.LAST_RESULT_DF, "Top products (test)")
```

---

## How the schema is discovered (why this works on any database)

Instead of manually describing your tables and columns, the script asks the
database itself:

```python
def describe_schema(engine, max_tables=MAX_TABLES_IN_SCHEMA) -> str:
    inspector = inspect(engine)
    table_names = inspector.get_table_names()[:max_tables]
    ...
```

This uses SQLAlchemy's `inspect()` to list every table and column that
actually exists, and builds a plain-text description automatically. Point
the script at a different database, and this description regenerates
itself — nothing needs to be rewritten by hand.

`MAX_TABLES_IN_SCHEMA` (default 30) caps how many tables get described, to
avoid an enormous schema description on a large shared database (e.g. a
company Redshift warehouse with hundreds of tables).

---

## Safety guardrail

The AI is only ever allowed to write read-only `SELECT` queries. Before
running any SQL, the tool checks for and blocks destructive keywords:

```python
forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
```

This matters because, unlike a version with pre-written fixed queries, this
version lets the AI write arbitrary SQL — this guardrail ensures it can only
ever *read* data, never modify or delete it.

---

## Known limitations

- **Free tier rate limits.** Gemini's free tier allows a limited number of
  requests per day (currently 20/day for `gemini-3.6-flash` at time of
  writing). Each question can use several requests internally (the AI may
  try more than one SQL query before settling on a final answer). Check
  current usage/limits at [ai.dev/rate-limit](https://ai.dev/rate-limit).
- **Large databases** may need `MAX_TABLES_IN_SCHEMA` increased, or schema
  filtering added, to avoid an overly large prompt.
- **CSV files are not databases.** This script needs an actual SQL database
  to connect to. To use a folder of CSVs, load them into DuckDB first
  (a few lines of one-time setup — see project notes for the loading
  pattern used to build `instacart.duckdb`).

---

## Quick reference — full run sequence

```python
# 1. Install (once per environment)
!pip install google-genai sqlalchemy pandas duckdb duckdb-engine matplotlib jinja2

# 2. Set folder + key BEFORE importing
import os
os.chdir("/path/to/your/project/folder")
os.environ["GEMINI_API_KEY"] = "AIzaSy...your-real-key..."

# 3. Import and use
import pandas as pd
import generic_sql_agent as agent

agent.show_answer("your question here")

# Optional: see the SQL the AI wrote
agent.VERBOSE = True
```
