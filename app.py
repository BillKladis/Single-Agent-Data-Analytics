"""
Analyst Copilot - tool-reasoning analytics agent with selectable architectures.
Run: streamlit run app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Analyst Copilot", page_icon="bar_chart", layout="wide")

API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not API_KEY or API_KEY == "your-key-here":
    st.error("Set ANTHROPIC_API_KEY in your .env file and restart.")
    st.stop()

import pandas as pd

from src import tools as tool_store
from src.data_gen import load_or_generate
from src.strategies import STRATEGIES, STRATEGY_LABELS


@st.cache_resource(show_spinner="Loading dataset...")
def get_dataset() -> pd.DataFrame:
    df = load_or_generate("data/superstore.csv")
    tool_store.init_tools(df)
    return df


@st.cache_resource(show_spinner="Initialising agent...")
def get_agent(strategy_short: str):
    return STRATEGIES[strategy_short](API_KEY)


df = get_dataset()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

# ----- Sidebar -----
with st.sidebar:
    st.subheader("Agent architecture")
    strategy_short = st.radio(
        "Reasoning strategy",
        options=list(STRATEGIES.keys()),
        format_func=lambda s: STRATEGY_LABELS[s],
        index=0,
        help="All three share the same 13-tool toolbox and system prompt - only "
             "the control flow differs. See the README for the benchmark study.",
    )
    st.divider()
    st.subheader("Dataset at a glance")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Date range", f"{df['Order_Date'].min().date()} - {df['Order_Date'].max().date()}")
    st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    st.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    st.divider()
    st.caption("**Toolbox (13 primitives)**")
    st.caption(", ".join(f"`{t}`" for t in tool_store.TOOL_NAMES))

agent = get_agent(strategy_short)

# ----- Header -----
st.title("Analyst Copilot")
st.caption(
    f"Tool-reasoning analytics agent | {len(df):,} rows | "
    f"strategy: **{STRATEGY_LABELS[strategy_short]}** | synthetic Superstore (2021-2024)"
)

EXAMPLES = [
    "Which region is most profitable?",
    "Is discount hurting profit?",
    "Does profit differ significantly across categories?",
    "Which category-region segments lose money?",
    "What's the sales trend over time?",
    "Do a few sub-categories drive most sales?",
]
cols = st.columns(3)
for i, q in enumerate(EXAMPLES):
    if cols[i % 3].button(q, use_container_width=True, key=f"ex_{i}"):
        st.session_state.pending_question = q

st.divider()


def render_trace(tool_trace, expanded):
    with st.expander("Tools used - reasoning trace", expanded=expanded):
        for i, call in enumerate(tool_trace, 1):
            st.markdown(f"**Step {i} - `{call['tool']}`**")
            st.code(call["args"], language="json")
            if call.get("output_snippet"):
                st.caption("Output preview:")
                st.code(call["output_snippet"][:300], language="text")
            st.markdown("---")


def render_metrics(meta):
    c = st.columns(4)
    c[0].metric("LLM calls", meta["n_llm_calls"])
    c[1].metric("Tool calls", meta["n_tool_calls"])
    c[2].metric("Tokens", f"{meta['total_tokens']:,}")
    c[3].metric("Latency", f"{meta['latency_s']:.1f}s")


# ----- History -----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("meta"):
                render_metrics(msg["meta"])
            for label, img in msg.get("figures", []):
                st.image(img, caption=label, use_container_width=True)
            if msg.get("tool_trace"):
                render_trace(msg["tool_trace"], expanded=False)

# ----- Input -----
user_input = st.chat_input("Ask a business question about the dataset...")
question = user_input or st.session_state.pending_question

if question:
    st.session_state.pending_question = ""
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner(f"Reasoning with {STRATEGY_LABELS[strategy_short]}..."):
            tool_store.reset_run()
            res = agent.run(question)
            figures = tool_store.get_figures()

        answer = res.answer or f"Agent error: {res.error}"
        meta = {
            "n_llm_calls": res.n_llm_calls, "n_tool_calls": len(res.tool_calls),
            "total_tokens": res.total_tokens, "latency_s": res.latency_s,
        }
        st.markdown(answer)
        render_metrics(meta)
        for label, img in figures:
            st.image(img, caption=label, use_container_width=True)
        if res.tool_calls:
            render_trace(res.tool_calls, expanded=True)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "tool_trace": res.tool_calls,
        "figures": figures, "meta": meta,
    })
    st.rerun()
