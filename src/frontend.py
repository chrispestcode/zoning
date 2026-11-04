import streamlit as st
import requests
import re
import base64
from io import BytesIO
from PIL import Image
from streamlit_image_zoom import image_zoom

# --- Configuration ---
API_URL = "http://localhost:8003/search"
PAGE_TITLE = "Semantic Search"
PAGE_ICON = "🔍"

# --- Page Setup ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.title(f"{PAGE_ICON} {PAGE_TITLE}")

# --- Custom CSS for Button Styling ---
st.markdown(
    """
    <style>
    /* Remove or repurpose .custom-button if not used elsewhere */
    /*
    .custom-button {
        width: 100%;
        height: 50px; 
        font-size: 16px; 
        background-color: #ADD8E6; 
        color: black; 
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .custom-button:hover {
        background-color: #87CEEB; 
    }
    */

    /* Style all streamlit buttons */
    .stButton>button {
        width: 100%;
        height: 50px; /* Consistent height */
        font-size: 16px; /* Adjust font size */
        background-color: #ADD8E6; /* Light Blue Background */
        color: black; /* Black text */
        border: none; /* Remove default border */
        border-radius: 5px; /* Rounded corners */
        cursor: pointer;
        transition: background-color 0.3s ease; /* Smooth hover transition */
    }
    .stButton>button:hover {
        background-color: #87CEEB; /* Slightly darker blue on hover */
    }

    /* Keep the text input style if you still want it, otherwise remove */
    /*
    div[data-testid="stTextInput"] input {
        background-color: #e6f2ff; 
        border-radius: 5px; 
    }
    */
    </style>
    """,
    unsafe_allow_html=True
)

@st.dialog("Form Details", width="large")
def enlarge_form(image, form_title, page_number):
    st.subheader(form_title)
    st.text(f"Page Number: {page_number}")
    try:
        st.image(image)
    except Exception as e:
        st.error(f"Error displaying form image: {e}")

# --- Session State Initialization ---
if "results" not in st.session_state:
    st.session_state.results = None
if "query" not in st.session_state:
    st.session_state.query = ""

# --- Search Input and Controls ---
query = st.text_input(
    "Enter your search query:",
    value=st.session_state.query,  # Use session state for persistence
    key="query_input",
    placeholder="Search for documents, images, tables..."
)

# Update session state query if text input changes
if query != st.session_state.query:
    st.session_state.query = query

col1, col2 = st.columns([1, 1])  # Equal column widths for uniform button sizes
with col1:
    search_button = st.button("Search", use_container_width=True)  # Ensure container width is used
with col2:
    clear_button = st.button("Clear Results", use_container_width=True)  # Match size with Search button

# --- Event Handling ---

# Handle Clear Button
if clear_button:
    st.session_state.results = None
    st.session_state.query = ""
    st.rerun()

# Handle Search Button
if search_button and st.session_state.query:
    query_text = st.session_state.query
    with st.spinner(f"Searching for: {query_text}..."):
        try:
            response = requests.post(API_URL, json={"query": query_text})
            response.raise_for_status()
            st.session_state.results = response.json().get("response", {})
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend API: {e}")
            st.session_state.results = None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.results = None
elif search_button and not st.session_state.query:
    st.warning("Please enter a search query.")

# --- Results Display ---
st.divider() # Separator before results area

if st.session_state.results is not None:
    text_response = st.session_state.results.get("text_response", {})
    scores = []
    # Check if results were found
    if not text_response: # Assuming if scores is empty, results are empty
         st.info("No results found for your query.")
    else:
        st.write(text_response)
        query_documents = st.session_state.results.get("query_documents")
        query_forms = st.session_state.results.get("query_forms")
        st.subheader(f"Search Results ({len(query_documents)} found):")
        # # Iterate through results
        # for idx, (score, content, doc_name, type_name) in enumerate(zip(scores, contents, doc_names, types)):
        #     alt_text = alt_texts[idx]
        #     extra = extra_content[idx]
        for idx, document in enumerate(query_documents):
            with st.container(border=True):
                st.subheader(f"Result #{idx + 1}")
                document_data = query_documents.get(document)
                st.caption(
                    f"Document: {document_data.get('FileName')}"
                    if document_data and document_data.get("FileName")
                    else f"Document #{document}: Document name is not available."
                )

        cols_per_row = 3
        idx = 0
        for row_start in range(0, len(query_forms), cols_per_row):
            row_forms = query_forms[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_forms))
            for col, form in zip(cols, row_forms):
                with col:
                    with st.container(border=True):
                        st.text(f"{form.get('FormTitle', 'Form Result')} (Page {form.get('PageNo', 'N/A')})")
                        try:
                            #st.text(form.get('Base64Encoding', ''))
                            delimeter = form.get('Base64Encoding', '').find(',')
                            image_enc = form.get('Base64Encoding', '')[delimeter+1:]
                            image_data = base64.b64decode(image_enc)
                            buffer = BytesIO(image_data)
                            image = Image.open(buffer)
                            image.load()
                            buffer.close()
                            image_zoom(
                                image=image,
                                mode="dragmove",  # Other options: "mousemove", "scroll", "both", "default"
                                size=300,         # Display size of the image
                                zoom_factor=1.0,  # Factor to zoom by
                                keep_resolution=True # Use original resolution for zooming
                            )
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                            
                        if st.button(f"View Form Details", key=f"form_{idx}"):
                            enlarge_form(
                                image=image,
                                form_title=form.get('FormTitle', 'Form Result'),
                                page_number=form.get('PageNo', 'N/A')
                            )
                idx += 1

else:
    # Initial state before any search
    st.info("Enter a query and click 'Search' to begin.")
