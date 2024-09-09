import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
import PyPDF2

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set the path to the PDF folder
PDF_FOLDER = "./pdf_catalogs"  # Update this path as needed

# Initialize session state
if 'furniture_database' not in st.session_state:
    st.session_state.furniture_database = pd.DataFrame(columns=['item', 'description', 'pdf_name'])
if 'pdf_contents' not in st.session_state:
    st.session_state.pdf_contents = {}

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([w for w in word_tokens if w.isalnum() and w not in stop_words])

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def process_pdf(pdf_name, content):
    try:
        # Simple parsing (you'd want to make this more robust)
        items = content.split('\n\n')
        new_rows = []
        for item in items:
            if len(item.strip()) > 0:
                lines = item.split('\n', 1)
                item_name = lines[0].strip()
                description = lines[1].strip() if len(lines) > 1 else ""
                new_rows.append({'item': item_name, 'description': description, 'pdf_name': pdf_name})
        
        new_df = pd.DataFrame(new_rows)
        st.session_state.furniture_database = pd.concat([st.session_state.furniture_database, new_df], ignore_index=True)
        return len(items)
    except Exception as e:
        st.error(f"Error processing PDF {pdf_name}: {str(e)}")
        return 0

def match_brief(brief):
    if st.session_state.furniture_database.empty:
        return []
    
    brief_processed = preprocess_text(brief)
    st.session_state.furniture_database['processed_description'] = st.session_state.furniture_database['description'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(st.session_state.furniture_database['processed_description'])
    brief_vector = vectorizer.transform([brief_processed])
    
    cosine_similarities = cosine_similarity(brief_vector, tfidf_matrix).flatten()
    
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    results = st.session_state.furniture_database.iloc[top_indices]
    
    return results

st.title('Furniture Catalog Matcher')

# Read PDFs from the folder
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

# Display list of available PDFs
st.subheader("Available Catalogs")
for pdf_name in pdf_files:
    st.write(f"- {pdf_name}")

# Button to process all PDFs
if st.button("Process All Catalogs"):
    st.session_state.furniture_database = pd.DataFrame(columns=['item', 'description', 'pdf_name'])
    st.session_state.pdf_contents = {}
    total_items = 0
    for pdf_name in pdf_files:
        file_path = os.path.join(PDF_FOLDER, pdf_name)
        content = read_pdf(file_path)
        st.session_state.pdf_contents[pdf_name] = content
        items_processed = process_pdf(pdf_name, content)
        total_items += items_processed
        st.success(f"Processed {pdf_name}. Items added: {items_processed}")
    st.success(f"Total items in database: {total_items}")

# Text area for client brief
client_brief = st.text_area("Enter client brief:")

if st.button("Find Matches"):
    if st.session_state.furniture_database.empty:
        st.error("Please process the catalogs first.")
    elif not client_brief:
        st.error("Please enter a client brief.")
    else:
        matches = match_brief(client_brief)
        st.subheader("Top Matches:")
        for idx, row in matches.iterrows():
            st.text(f"{idx+1}. {row['item']} (from {row['pdf_name']})")
            st.text(f"   Description: {row['description'][:200]}...")
            st.text("---")

# Display database stats
if not st.session_state.furniture_database.empty:
    st.sidebar.subheader("Database Stats")
    st.sidebar.text(f"Total items: {len(st.session_state.furniture_database)}")
    st.sidebar.text(f"Catalogs processed: {len(st.session_state.pdf_contents)}")

# Option to clear the database
if st.sidebar.button("Clear Database"):
    st.session_state.furniture_database = pd.DataFrame(columns=['item', 'description', 'pdf_name'])
    st.session_state.pdf_contents = {}
    st.sidebar.success("Database cleared.")

# Display the full database
if not st.session_state.furniture_database.empty:
    if st.checkbox("Show full database"):
        st.dataframe(st.session_state.furniture_database)

# Option to view raw PDF content
if st.session_state.pdf_contents:
    pdf_to_view = st.selectbox("Select a catalog to view its content:", list(st.session_state.pdf_contents.keys()))
    if st.button("View Catalog Content"):
        st.text_area("Catalog Content:", st.session_state.pdf_contents[pdf_to_view], height=300)