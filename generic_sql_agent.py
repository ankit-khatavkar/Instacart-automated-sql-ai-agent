"""
Generic SQL Insight Agent (v2 -- quiet mode + visualization support)
----------------------------------------------------------------------
Same as generic_sql_agent.py, with two additions:

1. VERBOSE = False by default -- tool calls / SQL are no longer printed.
   Set VERBOSE = True (or agent.VERBOSE = True in a notebook) to see them again.

2. Every run_sql call stores its result as a pandas DataFrame in LAST_RESULT_DF,
   so after asking a question you can immediately turn the underlying data into
   a clean table or chart -- see show_answer() at the bottom for a ready-made
   helper, or just use LAST_RESULT_DF yourself.

SETUP: same as generic_sql_agent.py
  pip install google-genai sqlalchemy pandas duckdb duckdb-engine matplotlib
  export GEMINI_API_KEY="..."

USAGE (in Jupyter)
  import generic_sql_agent as agent
  agent.show_answer("What are the top 10 most ordered products?")
"""

import os
import sys
import json

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from google import genai
from google.genai import types

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING", "duckdb:///instacart.duckdb")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.6-flash"
MAX_ROWS_RETURNED = 50
MAX_TABLES_IN_SCHEMA = 30

# Set this to True (agent.VERBOSE = True) if you want to see the SQL/tool calls again
VERBOSE = False

if not GEMINI_API_KEY:
    sys.exit("Set GEMINI_API_KEY (get a free key at https://aistudio.google.com/app/apikey)")

engine = create_engine(DB_CONNECTION_STRING)
client = genai.Client(api_key=GEMINI_API_KEY)

# Holds the DataFrame from the most recent run_sql call, so you can
# visualize it right after asking a question.
LAST_RESULT_DF = None
LAST_SQL = None

# Holds every query result from the current question (reset at the start of
# each ask_agent call). Used to recover the "real" answer even if the AI
# runs an extra verification/sanity-check query after already getting the
# real result -- see _pick_best_result() below.
_QUERY_HISTORY = []


# ----------------------------------------------------------------------
# AUTO-DISCOVER THE SCHEMA
# ----------------------------------------------------------------------

def describe_schema(engine, max_tables=MAX_TABLES_IN_SCHEMA) -> str:
    inspector = inspect(engine)
    table_names = inspector.get_table_names()[:max_tables]
    lines = ["Database schema (auto-discovered):\n"]
    for table in table_names:
        columns = inspector.get_columns(table)
        col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        lines.append(f"- {table}({col_desc})")
    return "\n".join(lines)


SCHEMA_NOTES = describe_schema(engine)


# ----------------------------------------------------------------------
# THE ONE TOOL: run arbitrary read-only SQL
# ----------------------------------------------------------------------

def run_sql(sql: str) -> str:
    """Run a read-only SQL query against the connected database and return rows as JSON."""
    global LAST_RESULT_DF, LAST_SQL

    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    if any(sql.strip().lower().startswith(word) for word in forbidden):
        return json.dumps({"error": "Only read-only SELECT queries are allowed."})

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
    except Exception as e:
        return json.dumps({"error": str(e)})

    # remember the full (untrimmed) result for visualization
    LAST_RESULT_DF = df.copy()
    LAST_SQL = sql
    _QUERY_HISTORY.append(df.copy())  # keep every result from this question, not just the last

    if len(df) > MAX_ROWS_RETURNED:
        df = df.head(MAX_ROWS_RETURNED)
    return df.to_json(orient="records")


TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="run_sql",
            description="Run a read-only SQL SELECT query against the connected database and return the results.",
            parameters={
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "A valid SQL SELECT query, with a LIMIT clause."}},
                "required": ["sql"],
            },
        )
    ])
]

PYTHON_FUNCTIONS = {"run_sql": run_sql}


# ----------------------------------------------------------------------
# AGENT LOOP
# ----------------------------------------------------------------------

def _pick_best_result():
    """
    Choose the most likely 'real answer' DataFrame from this question's
    query history. Heuristic: prefer the result with the most columns
    (a genuine stats/summary/ranking result usually has several columns;
    a one-off verification query like SELECT COUNT(*) typically has just
    one). Ties broken by picking the earliest such result, since a later
    single-column query is more likely to be a sanity check than the
    original answer.
    """
    if not _QUERY_HISTORY:
        return None
    return max(_QUERY_HISTORY, key=lambda df: df.shape[1])


def ask_agent(question: str, max_turns: int = 6) -> str:
    global LAST_RESULT_DF
    _QUERY_HISTORY.clear()  # start fresh for this question

    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a data analyst agent. You have one tool, run_sql, which lets you "
            "query the database described below. Write correct SQL for the question, "
            "call the tool, then explain the results in plain English with concrete "
            "numbers.\n\n" + SCHEMA_NOTES
        ),
        tools=TOOLS,
    )

    for _ in range(max_turns):
        response = client.models.generate_content(
            model=MODEL_NAME, contents=contents, config=config,
        )
        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_call = None
        for part in candidate.content.parts:
            if part.function_call:
                function_call = part.function_call
                break

        if not function_call:
            LAST_RESULT_DF = _pick_best_result()
            return response.text

        fn_name = function_call.name
        fn_args = dict(function_call.args) if function_call.args else {}
        if VERBOSE:
            print(f"  [tool call] {fn_name}({fn_args})")

        result = PYTHON_FUNCTIONS[fn_name](**fn_args) if fn_name in PYTHON_FUNCTIONS else json.dumps({"error": "unknown tool"})

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=fn_name, response={"result": result})],
            )
        )

    LAST_RESULT_DF = _pick_best_result()
    return "Reached max tool-call turns without a final answer."


# ----------------------------------------------------------------------
# VERIFICATION: check that the chart/card data actually matches numbers
# the AI stated in its own text answer, before displaying anything visual.
# This doesn't guarantee correctness -- it's a heuristic safety net, not a
# proof -- but it catches the common failure mode where a stray follow-up
# query leaves stale/wrong data sitting in LAST_RESULT_DF.
# ----------------------------------------------------------------------

import re


def _extract_numbers_from_text(text: str) -> list:
    """Pull every number out of a text answer, normalized to plain floats.
    Handles commas (1,234), percentages (83%), and plain decimals (10.09)."""
    numbers = []
    for match in re.findall(r"-?\d[\d,]*\.?\d*%?", text):
        cleaned = match.replace(",", "")
        is_percent = cleaned.endswith("%")
        cleaned = cleaned.rstrip("%")
        if cleaned in ("", "-", "."):
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if is_percent:
            value = value / 100.0
        numbers.append(value)
    return numbers


def _extract_numbers_from_df(df: pd.DataFrame) -> list:
    """Pull every numeric cell value out of a result DataFrame."""
    numeric_cols = df.select_dtypes(include="number").columns
    values = []
    for col in numeric_cols:
        values.extend(df[col].dropna().tolist())
    return [float(v) for v in values]


def _numbers_match(a: float, b: float) -> bool:
    """True if two numbers are close enough to count as 'the same', allowing
    for rounding differences (e.g. 10.09 vs 10.088883)."""
    tolerance = max(0.015, abs(b) * 0.01)  # 1% relative, or 0.015 absolute for small numbers
    return abs(a - b) <= tolerance


def _chart_matches_answer(df: pd.DataFrame, answer_text: str, min_overlap_ratio: float = 0.5) -> bool:
    """
    Returns True if a reasonable share of the chart/card's numbers also
    appear in the AI's own text answer -- i.e. the visualization and the
    explanation are telling a consistent story. Returns False (skip the
    visualization) if they don't sufficiently agree, or if there's nothing
    numeric to check.
    """
    chart_numbers = _extract_numbers_from_df(df)
    if not chart_numbers:
        return False

    text_numbers = _extract_numbers_from_text(answer_text)
    if not text_numbers:
        return False

    matched = sum(
        1 for c in chart_numbers if any(_numbers_match(c, t) for t in text_numbers)
    )
    return (matched / len(chart_numbers)) >= min_overlap_ratio


# ----------------------------------------------------------------------
# VISUALIZATION HELPERS (Jupyter-friendly)
# ----------------------------------------------------------------------

def _format_number(x) -> str:
    """Light formatting: commas for big numbers, 1-2 decimals for fractions/rates."""
    try:
        if float(x).is_integer():
            return f"{int(x):,}"
        if abs(x) < 1:
            return f"{x:.1%}" if 0 <= x <= 1 else f"{x:.2f}"
        return f"{x:,.2f}"
    except (ValueError, TypeError):
        return str(x)


def _render_bar_chart(df: pd.DataFrame, question: str) -> bool:
    """Try a horizontal bar chart. Returns True if it drew something, False if the shape doesn't fit."""
    import matplotlib.pyplot as plt

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    label_cols = df.select_dtypes(exclude="number").columns.tolist()

    # need at least 2 rows to make a bar comparison meaningful, one label col, one number col
    if len(numeric_cols) < 1 or len(label_cols) < 1 or len(df) < 2:
        return False

    label_col = label_cols[0]
    value_col = numeric_cols[0]
    plot_df = df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(plot_df))))
    ax.barh(plot_df[label_col].astype(str), plot_df[value_col])
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(question, fontsize=11)
    plt.tight_layout()
    plt.show()
    return True


def _render_stat_cards(df: pd.DataFrame) -> bool:
    """
    Fallback for single-row results (e.g. one overall stat, or one row with several
    metrics side by side) -- shows each numeric column as its own big-number card
    instead of a chart. Returns True if it drew something, False otherwise.
    """
    from IPython.display import display, HTML

    if len(df) != 1:
        return False

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return False

    label_cols = df.select_dtypes(exclude="number").columns.tolist()
    subtitle = " · ".join(str(df.iloc[0][c]) for c in label_cols) if label_cols else ""

    cards_html = ""
    for col in numeric_cols:
        value = _format_number(df.iloc[0][col])
        label = col.replace("_", " ").title()
        cards_html += f"""
        <div style="display:inline-block; min-width:150px; margin:6px; padding:16px 20px;
                    border:1px solid #ddd; border-radius:10px; text-align:center;
                    font-family:sans-serif; background:#fafafa;">
            <div style="font-size:24px; font-weight:700; color:#222;">{value}</div>
            <div style="font-size:12px; color:#666; margin-top:4px;">{label}</div>
        </div>"""

    subtitle_html = f'<div style="font-family:sans-serif; color:#888; font-size:12px; margin-bottom:4px;">{subtitle}</div>' if subtitle else ""
    display(HTML(f'<div>{subtitle_html}<div>{cards_html}</div></div>'))
    return True


def show_answer(question: str, chart: bool = True, max_chart_rows: int = 15):
    """
    Ask the agent a question and display:
      1. The plain-English answer (always shown)
      2. A visualization, but ONLY if its numbers are consistent with what
         the AI actually wrote in its text answer. Priority order:
           a. Horizontal bar chart, if the result has 2+ rows with a label + number column
           b. Stat card(s), if the result is a single summary row
           c. Nothing -- if neither shape fits, OR if the chart's numbers
              don't reasonably match the text answer, skip the visualization
              rather than risk showing something misleading.

    Run this directly in a Jupyter cell (relies on IPython display).
    Returns nothing (so Jupyter doesn't auto-print the raw answer text a
    second time below the rendered output). If you need the raw answer
    string in code, use ask_agent(question) directly instead.
    """
    from IPython.display import display, Markdown

    answer = ask_agent(question)
    display(Markdown(f"### {question}\n\n{answer}"))

    df = LAST_RESULT_DF
    if df is None or df.empty:
        return

    # ---- verification: only visualize if the chart data agrees with the text ----
    if chart and not _chart_matches_answer(df, answer):
        display(Markdown(
            "*(Visualization skipped — couldn't confirm it matches the answer above. "
            "The numbers in the text are the ones to trust.)*"
        ))
        return

    # ---- visualization: try bar chart first, then stat cards, else skip silently ----
    if chart:
        if len(df) <= max_chart_rows and _render_bar_chart(df, question):
            pass
        elif _render_stat_cards(df):
            pass
        # else: no chart-friendly shape -- nothing more to draw

    return  # explicit None -- prevents Jupyter's auto-display of a return value
