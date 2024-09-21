import streamlit as st

# Columns
col1, col2, col3 = st.columns(3)
with col1:
    st.write("This is column 1")
with col2:
    st.write("This is column 2")
with col3:
    st.write("This is column 3")

# Expander
with st.expander("Click to expand"):
    st.write("This content is hidden by default")

# Sidebar
st.sidebar.title("Sidebar Title")
st.sidebar.write("This content appears in the sidebar")

# Tabs
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("This is tab 1")
with tab2:
    st.write("This is tab 2")
