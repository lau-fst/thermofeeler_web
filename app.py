import streamlit as st
import requests

st.set_page_config(page_title="ThermoFeeler", page_icon="🌡",
        layout="centered",
        initial_sidebar_state="auto")

title = """<p style="font-family:'Tangerine'; color:Red; font-size:42px;">ThermoFeeler</p>"""
st.markdown(title, unsafe_allow_html=True)

st.markdown("""Enter a twitter query""")
query= st.text_input('Example : Apple', 'Apple')

with st.spinner('Wait for it...'):
    url = f'https://thermofeeler-6hn6fqkota-uc.a.run.app/predict_query?query={query}&max_results=10'
    response = requests.get(url).json()[1]
st.success('Done!')

col1, col2, col3, col4 = st.columns(4)
col1.write(f"Total number of tweets retrieved : {response['total']}")
col2.write(f"Total number of negative tweets  : {response['negative total']}")
col3.write(f"Total number of neutral tweets : {response['neutral total']}")
col4.write(f"Total number of positive tweets : {response['positive total']}")
