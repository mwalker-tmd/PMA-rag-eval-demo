"""Streamlit entry point (driving adapter). Run with: uv run streamlit run app.py"""

import streamlit as st

st.set_page_config(page_title="RAG Pipeline Chat", page_icon="💬", layout="wide")

pages = [
    st.Page("src/rag_chatbot/ui/chat_page.py", title="Chat", icon="💬", url_path="chat", default=True),
    st.Page("src/rag_chatbot/ui/test_sets_page.py", title="Test Sets", icon="🧪", url_path="test-sets"),
    st.Page("src/rag_chatbot/ui/evaluations_page.py", title="Evaluations", icon="📊", url_path="evaluations"),
]
st.navigation(pages).run()
