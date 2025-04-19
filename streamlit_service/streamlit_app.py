import os
import requests
import streamlit as st

INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8003/api/v1/ingest")
ASK_URL = os.getenv("ASK_URL", "http://localhost:8004/api/v1/ask")
ARTIST_URL = os.getenv("ARTIST_URL", "http://localhost:8003/api/v1/artist")

st.set_page_config(page_title="Concert‑RAG UI", page_icon="🎤", layout="centered")
st.title("🎤 Concert‑Tour Knowledge Bot")

mode = st.radio("Choose action", ["Add document", "Ask question", "Artist lookup (online)"])

if mode == "Add document":
    st.markdown("### Paste concert tour document (plain text)")
    doc_text = st.text_area("Document", height=250)
    if st.button("Ingest") and doc_text.strip():
        with st.spinner("Uploading & summarising …"):
            resp = requests.post(INGEST_URL, json={"text": doc_text})
        if resp.status_code == 201:
            st.success("Document ingested! Summary:")
            st.write(resp.json()["summary"])
        else:
            st.error(resp.json().get("detail", resp.text))

elif mode == "Ask question":
    st.markdown("### Ask about upcoming tours (2025‑2026)")
    q = st.text_input("Question", placeholder="Where will BTS perform in 2026?")
    if st.button("Ask") and q.strip():
        with st.spinner("Thinking …"):
            resp = requests.post(ASK_URL, json={"question": q})
        st.write(resp.json()["answer"])

else:  # Artist lookup
    st.markdown("### Live search for an artist’s concerts (bonus mode)")
    name = st.text_input("Artist name", placeholder="Taylor Swift")
    if st.button("Search") and name.strip():
        with st.spinner("Searching online …"):
            resp = requests.get(ARTIST_URL, params={"name": name})
        st.write(resp.json()["answer"])
