"""Web chat UI. Run with: streamlit run streamlit_app.py"""

from __future__ import annotations

import json

import streamlit as st

from gene_agent.agent import Agent, Reply
from gene_agent.errors import friendly_error
from gene_agent.tools import available_cancers

EXAMPLES = [
    "How can you help me?",
    "What are the main genes involved in lung cancer?",
    "What is the median value expression of genes involved in breast cancer?",
    "What is the median value expression of genes involved in esophageal cancer?",
]

st.set_page_config(page_title="Gene data assistant", page_icon="🧬")
st.title("Gene data assistant")
st.caption("Ask questions about the cancer gene dataset in plain language.")


def render_reply(reply: Reply) -> None:
    st.markdown(reply.text)
    if reply.ungrounded_numbers():
        st.warning("Some figures in the text above are not in the dataset. Rely on the data shown below.")
    data = reply.data_markdown()
    if data:
        with st.container(border=True):
            st.caption("From the dataset")
            st.markdown(data)
    if reply.tool_calls:
        count = len(reply.tool_calls)
        with st.expander(f"How this answer was produced ({count} data lookup{'s' if count != 1 else ''})"):
            for call in reply.tool_calls:
                st.markdown(f"**{call.name}**")
                st.json({"input": call.input})
                if call.is_error:
                    st.warning(call.output)
                else:
                    st.json(json.loads(call.output))


if "agent" not in st.session_state:
    agent = Agent()
    try:
        bar = st.progress(0.0, text=f"Preparing the local model ({agent.model}). The first run downloads it.")
        agent.ensure_model(lambda done, total: bar.progress(min(done / total, 1.0)))
        bar.empty()
    except Exception as exc:
        st.error(friendly_error(exc))
        st.stop()
    st.session_state.agent = agent
    st.session_state.history = []

agent = st.session_state.agent
history = st.session_state.history

with st.sidebar:
    st.subheader("Dataset")
    st.write("Cancer indications covered:")
    st.write(", ".join(available_cancers()))
    st.subheader("Model")
    st.write(f"`{agent.model}`, running locally. Nothing leaves this machine.")
    if st.button("New conversation"):
        agent.reset()
        history.clear()
        st.rerun()

for question, reply in history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render_reply(reply)

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
                reply = agent.ask(question)
            except Exception as exc:  # shown to the user, never a stack trace
                st.error(friendly_error(exc))
                st.stop()
        render_reply(reply)
    history.append((question, reply))
    if clicked:
        st.rerun()  # clear the example buttons
