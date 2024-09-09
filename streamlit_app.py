import streamlit as st
from pypdf import PdfReader
import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize session state
if 'furniture_database' not in st.session_state:
    st.session_state.furniture_database = pd.DataFrame(columns=['item', 'description'])

def preprocess_text(text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([w for w in word_tokens if w.isalnum() and w not in stop_words])

def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Simple parsing (you'd want to make this more robust)
        items = re.split(r'\n\n', text)
        for item in items:
            if len(item.strip()) > 0:
                # Assume first line is item name, rest is description
                lines = item.split('\n', 1)
                item_name = lines[0].strip()
                description = lines[1].strip() if len(lines) > 1 else ""
                new_row = pd.DataFrame({'item': [item_name], 'description': [description]})
                st.session_state.furniture_database = pd.concat([st.session_state.furniture_database, new_row], ignore_index=True)
        return len(items)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return 0

def match_brief(brief):
    if st.session_state.furniture_database.empty:
        return []
    
    # Preprocess the brief and furniture descriptions
    brief_processed = preprocess_text(brief)
    st.session_state.furniture_database['processed_description'] = st.session_state.furniture_database['description'].apply(preprocess_text)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(st.session_state.furniture_database['processed_description'])
    brief_vector = vectorizer.transform([brief_processed])
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(brief_vector, tfidf_matrix).flatten()
    
    # Get top 5 matches
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    results = st.session_state.furniture_database.iloc[top_indices]
    
    return results

st.title('Furniture Catalog Matcher')

# File uploader for PDFs
uploaded_files = st.file_uploader("Choose PDF catalog files", accept_multiple_files=True, type="pdf")
if uploaded_files:
    total_items = 0
    for file in uploaded_files:
        items_processed = process_pdf(file)
        total_items += items_processed
    st.success(f"Processed {len(uploaded_files)} PDF catalogs. Total items added: {total_items}")

# Text area for client brief
client_brief = st.text_area("Enter client brief:")

if st.button("Find Matches"):
    if st.session_state.furniture_database.empty:
        st.error("Please upload PDF catalogs first.")
    elif not client_brief:
        st.error("Please enter a client brief.")
    else:
        matches = match_brief(client_brief)
        st.subheader("Top Matches:")
        for idx, row in matches.iterrows():
            st.text(f"{idx+1}. {row['item']}")
            st.text(f"   Description: {row['description'][:200]}...")
            st.text("---")

# Display database stats
if not st.session_state.furniture_database.empty:
    st.sidebar.subheader("Database Stats")
    st.sidebar.text(f"Total items: {len(st.session_state.furniture_database)}")

# Option to clear the database
if st.sidebar.button("Clear Database"):
    st.session_state.furniture_database = pd.DataFrame(columns=['item', 'description'])
    st.sidebar.success("Database cleared.")

# Display the full database
if not st.session_state.furniture_database.empty:
    if st.checkbox("Show full database"):
        st.dataframe(st.session_state.furniture_database)