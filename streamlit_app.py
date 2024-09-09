import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
import PyPDF2
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set NLTK data path
nltk_data_path = os.path.expanduser("~/nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
    nltk.download('stopwords', quiet=True, download_dir=nltk_data_path)

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set the path to the PDF folder
PDF_FOLDER = "./pdf_catalogs"  # Update this path as needed

# Initialize session state
if 'furniture_database' not in st.session_state:
    st.session_state.furniture_database = pd.DataFrame(columns=['name', 'type', 'description', 'sizes', 'finishes', 'price', 'pdf_name'])
if 'pdf_contents' not in st.session_state:
    st.session_state.pdf_contents = {}

def parse_furniture_item(text):
    item = {
        'name': '',
        'type': '',
        'description': '',
        'sizes': '',
        'finishes': '',
        'price': ''
    }
    
    # Try to extract name (assuming it's the first line)
    lines = text.split('\n')
    if lines:
        item['name'] = lines[0].strip()
    
    # Look for specific patterns
    price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', text)
    if price_match:
        item['price'] = price_match.group()
    
    weight_match = re.search(r'Weight Capacity:\s*(\d+kg)', text)
    if weight_match:
        item['description'] = f"Weight Capacity: {weight_match.group(1)}"
    
    # Extract any other information you can identify
    # ...

    return item

def process_pdf(pdf_name, content):
    try:
        # Split content into items (adjust based on your PDF structure)
        items = re.split(r'\n(?=[\w\s]{3,50}:)', content)
        new_rows = []
        for item_text in items:
            if len(item_text.strip()) > 0:
                furniture_item = parse_furniture_item(item_text)
                furniture_item['pdf_name'] = pdf_name
                new_rows.append(furniture_item)
        
        new_df = pd.DataFrame(new_rows)
        st.session_state.furniture_database = pd.concat([st.session_state.furniture_database, new_df], ignore_index=True)
        return len(new_rows)
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_name}: {str(e)}")
        st.error(f"Error processing PDF {pdf_name}. Please check the log for details.")
        return 0

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def match_brief(brief):
    if st.session_state.furniture_database.empty:
        return []
    
    brief_processed = preprocess_text(brief)
    st.session_state.furniture_database['processed_text'] = st.session_state.furniture_database.apply(
        lambda row: preprocess_text(' '.join([str(row['name']), str(row['type']), str(row['description']), str(row['sizes']), str(row['finishes'])])),
        axis=1
    )
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(st.session_state.furniture_database['processed_text'])
    brief_vector = vectorizer.transform([brief_processed])
    
    cosine_similarities = cosine_similarity(brief_vector, tfidf_matrix).flatten()
    
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    results = st.session_state.furniture_database.iloc[top_indices]
    
    return results

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([w for w in word_tokens if w.isalnum() and w not in stop_words])

st.title('Furniture Catalog Matcher')

# Read PDFs from the folder
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

# Display list of available PDFs
st.subheader("Available Catalogs")
for pdf_name in pdf_files:
    st.write(f"- {pdf_name}")

# Button to process all PDFs
if st.button("Process All Catalogs"):
    st.session_state.furniture_database = pd.DataFrame(columns=['name', 'type', 'description', 'sizes', 'finishes', 'price', 'pdf_name'])
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
            st.markdown(f"**{idx+1}. {row['name']}** (from {row['pdf_name']})")
            for field in ['type', 'description', 'sizes', 'finishes', 'price']:
                if row[field]:
                    st.markdown(f"**{field.capitalize()}:** {row[field]}")
            st.markdown("---")

# Display database stats
if not st.session_state.furniture_database.empty:
    st.sidebar.subheader("Database Stats")
    st.sidebar.text(f"Total items: {len(st.session_state.furniture_database)}")
    st.sidebar.text(f"Catalogs processed: {len(st.session_state.pdf_contents)}")

# Option to clear the database
if st.sidebar.button("Clear Database"):
    st.session_state.furniture_database = pd.DataFrame(columns=['name', 'type', 'description', 'sizes', 'finishes', 'price', 'pdf_name'])
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