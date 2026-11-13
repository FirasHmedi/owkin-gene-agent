"""Web chat UI. Run with: streamlit run streamlit_app.py"""

from __future__ import annotations

import json
import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from gene_agent.agent import Agent, Reply
from gene_agent.errors import friendly_error
from gene_agent.tools import available_cancers

load_dotenv()

EXAMPLES = [
    "How can you help me?",
    "What are the main genes involved in lung cancer?",
    "What is the median value expression of genes involved in breast cancer?",
    "What is the median value expression of genes involved in esophageal cancer?",
]

st.set_page_config(page_title="Gene data assistant", page_icon="🧬")
st.title("Gene data assistant")
st.caption("Ask questions about the cancer gene dataset in plain language.")


def render_tool_calls(reply: Reply) -> None:
    """Show which dataset lookups produced the answer."""
    if not reply.tool_calls:
        return
    count = len(reply.tool_calls)
    with st.expander(f"How this answer was produced ({count} data lookup{'s' if count != 1 else ''})"):
        for call in reply.tool_calls:
            st.markdown(f"**{call.name}**")
            st.json({"input": call.input})
            if call.is_error:
                st.warning(call.output)
            else:
                st.json(json.loads(call.output))


with st.sidebar:
    st.subheader("Dataset")
    st.write("Cancer indications covered:")
    st.write(", ".join(available_cancers()))

    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.text_input(
        "Anthropic API key",
        type="password",
        help="Used for this browser session only. Set ANTHROPIC_API_KEY to skip this field.",
    )

    if st.button("New conversation"):
        st.session_state.pop("agent", None)
        st.session_state.pop("history", None)
        st.rerun()

if not api_key:
    st.info("Enter an Anthropic API key in the sidebar, or set ANTHROPIC_API_KEY, to start.")
    st.stop()

if "agent" not in st.session_state or st.session_state.get("agent_key") != api_key:
    st.session_state.agent = Agent(client=anthropic.Anthropic(api_key=api_key))
    st.session_state.agent_key = api_key
    st.session_state.history = []

history = st.session_state.history

for question, reply in history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(reply.text)
        render_tool_calls(reply)

clicked = None
if not history:
    st.write("Try one of these:")
    for example in EXAMPLES:
        if st.button(example):
            clicked = example

question = st.chat_input("Ask about a cancer or its genes") or clicked

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Looking that up..."):
            try:
                reply = st.session_state.agent.ask(question)
            except Exception as exc:  # shown to the user, never a stack trace
                st.error(friendly_error(exc))
                st.stop()
        st.markdown(reply.text)
        render_tool_calls(reply)
    history.append((question, reply))
    if clicked:
        st.rerun()  # clear the example buttons
