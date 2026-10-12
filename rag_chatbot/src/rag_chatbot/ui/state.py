"""Shared Streamlit state: one cached container per server process."""

from __future__ import annotations

import streamlit as st

from rag_chatbot.container import ConfigError, Container, build_container


@st.cache_resource(show_spinner=False)
def get_container() -> Container:
    # One container per process: embedded Qdrant allows a single client per index directory,
    # and the container's JobRunner must be shared by every page and browser tab.
    return build_container()


def container_or_stop() -> Container:
    """The app's container, or a configuration error on the page (and stop)."""
    try:
        return get_container()
    except ConfigError as e:
        st.error(f"Configuration error: {e}")
        st.info("Set the value in `rag_chatbot/.env` (see `.env.sample`), then restart the app.")
        st.stop()
