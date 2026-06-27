"""
Analyst Copilot — Streamlit chat interface for a tool-reasoning analytics agent.
Run: streamlit run app.py
"""
import os
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyst Copilot",
    page_icon="📊",
    layout="wide",
)

# ── API key guard ────────────────────────────────────────────────────────────
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not API_KEY or API_KEY == "your-key-here":
    st.error("Set ANTHROPIC_API_KEY in your .env file and restart.")
    st.stop()

# ── Lazy imports (keep startup fast) ────────────────────────────────────────
import pandas as pd
from src.data_gen import load_or_generate
from src import tools as tool_store
from src.agent import build_agent, run_query

# ── Dataset ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dataset…")
def get_dataset() -> pd.DataFrame:
    df = load_or_generate("data/superstore.csv")
    tool_store.init_tools(df)
    return df


@st.cache_resource(show_spinner="Initialising agent…")
def get_agent():
    return build_agent(API_KEY)


df = get_dataset()
executor = get_agent()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, tool_trace, figures}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Analyst Copilot")
st.caption(
    f"Tool-reasoning analytics agent · {len(df):,} rows · "
    "Synthetic Superstore dataset (2021–2024)"
)

# ── Example question buttons ──────────────────────────────────────────────────
EXAMPLES = [
    "Which region is most profitable?",
    "Is discount hurting profit?",
    "What's the sales trend over time?",
]

cols = st.columns(len(EXAMPLES))
for col, q in zip(cols, EXAMPLES):
    if col.button(q, use_container_width=True):
        st.session_state.pending_question = q

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Render figures inline
            if msg.get("figures"):
                for label, img_bytes in msg["figures"]:
                    st.image(img_bytes, caption=label, use_container_width=True)
            # Tool trace expander
            if msg.get("tool_trace"):
                with st.expander("🔍 Tools used — reasoning trace", expanded=False):
                    for i, call in enumerate(msg["tool_trace"], 1):
                        st.markdown(f"**Step {i} · `{call['tool']}`**")
                        st.code(call["args"], language="json")
                        if call.get("output_snippet"):
                            st.caption("Output preview:")
                            st.code(call["output_snippet"][:300], language="text")
                        st.markdown("---")

# ── Input handling ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a business question about the dataset…")

# Merge pending example-button question
question = user_input or st.session_state.pending_question
if st.session_state.pending_question and user_input != st.session_state.pending_question:
    st.session_state.pending_question = ""

if question:
    st.session_state.pending_question = ""

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            tool_store.reset_run()
            try:
                answer, tool_trace = run_query(executor, question)
            except Exception as e:
                answer = f"Agent error: {e}"
                tool_trace = []

            figures = tool_store.get_figures()

        st.markdown(answer)

        if figures:
            for label, img_bytes in figures:
                st.image(img_bytes, caption=label, use_container_width=True)

        if tool_trace:
            with st.expander("🔍 Tools used — reasoning trace", expanded=True):
                for i, call in enumerate(tool_trace, 1):
                    st.markdown(f"**Step {i} · `{call['tool']}`**")
                    st.code(call["args"], language="json")
                    if call.get("output_snippet"):
                        st.caption("Output preview:")
                        st.code(call["output_snippet"][:300], language="text")
                    st.markdown("---")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "tool_trace": tool_trace,
            "figures": figures,
        }
    )
    st.rerun()

# ── Sidebar: dataset overview ─────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Dataset at a glance")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", len(df.columns))
    st.metric("Date range", f"{df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")
    st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    st.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    st.markdown("---")
    st.caption("**Available columns**")
    st.caption(", ".join(df.columns.tolist()))
    st.markdown("---")
    st.caption("**Available tools**")
    tool_names = [
        "describe_dataset", "investigate_distribution", "group_compare",
        "correlate", "top_n", "filter_count", "trend_over_time",
    ]
    for t in tool_names:
        st.caption(f"• `{t}`")
