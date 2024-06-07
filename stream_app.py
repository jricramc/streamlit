import streamlit as st

st.set_page_config(
    page_title="Multi-Agent Debate",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

PAGES = {
    "Main Simulation": "main_page",
    "Input Question": "input_page",
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]

if page == "main_page":
    import stream_main
    stream_main.run()
elif page == "input_page":
    import stream_input
    stream_input.run()
