import streamlit as st
import pandas as pd
from utils import load_csv, generate_summary, answer_question

st.set_page_config(page_title="CSV Chat — Gemini", layout="wide")
st.title("chat with your csv")

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.filename = None
    st.session_state.messages = []
    st.session_state.summary_done = False

uploaded = st.file_uploader("upload a csv", type="csv")

if uploaded:
    if st.session_state.filename != uploaded.name:
        df = load_csv(uploaded)
        st.session_state.df = df
        st.session_state.filename = uploaded.name
        st.session_state.messages = []
        st.session_state.summary_done = False

    df = st.session_state.df

    col1, col2, col3 = st.columns(3)
    col1.metric("rows", df.shape[0])
    col2.metric("columns", df.shape[1])
    col3.metric("file", uploaded.name)

    with st.expander("preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    if not st.session_state.summary_done:
        with st.chat_message("assistant"):
            with st.spinner("reading your data..."):
                summary = generate_summary(df)
            st.markdown(summary)
            st.session_state.messages.append(
                {"role": "assistant", "content": summary}
            )
        st.session_state.summary_done = True

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("ask about your data..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                history = [
                    {"content": m["content"], "response": m["content"]}
                    for m in st.session_state.messages
                    if m["role"] == "assistant"
                ]
                answer = answer_question(df, prompt, history)
            st.markdown(answer)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

else:
    st.info("upload a csv to start.")
    st.session_state.messages = []
    st.session_state.summary_done = False
