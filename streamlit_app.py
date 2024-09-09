import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import io

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize session state
if 'furniture_database' not in st.session_state:
    st.session_state.furniture_database = pd.DataFrame(columns=['item', 'description', 'pdf_name'])
if 'pdf_contents' not in st.session_state:
    st.session_state.pdf_contents = {}

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([w for w in word_tokens if w.isalnum() and w not in stop_words])

def process_pdf(file):
    try:
        pdf_content = file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Store PDF content in session state
        st.session_state.pdf_contents[file.name] = text
        
        # Simple parsing (you'd want to make this more robust)
        items = text.split('\n\n')
        new_rows = []
        for item in items:
            if len(item.strip()) > 0:
                # Assume first line is item name, rest is description
                lines = item.split('\n', 1)
                item_name = lines[0].strip()
                description = lines[1].strip() if len(lines) > 1 else ""
                new_rows.append({'item': item_name, 'description': description, 'pdf_name': file.name})
        
        new_df = pd.DataFrame(new_rows)
        st.session_state.furniture_database = pd.concat([st.session_state.furniture_database, new_df], ignore_index=True)
        return len(items)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
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

# File uploader for PDFs
uploaded_files = st.file_uploader("Choose PDF catalog files", accept_multiple_files=True, type="pdf")
if uploaded_files:
    total_items = 0
    for file in uploaded_files:
        if file.name not in st.session_state.pdf_contents:
            items_processed = process_pdf(file)
            total_items += items_processed
            st.success(f"Processed {file.name}. Items added: {items_processed}")
        else:
            st.info(f"{file.name} already processed. Skipping.")
    st.success(f"Total items in database: {len(st.session_state.furniture_database)}")

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
            st.text(f"{idx+1}. {row['item']} (from {row['pdf_name']})")
            st.text(f"   Description: {row['description'][:200]}...")
            st.text("---")

# Display database stats
if not st.session_state.furniture_database.empty:
    st.sidebar.subheader("Database Stats")
    st.sidebar.text(f"Total items: {len(st.session_state.furniture_database)}")
    st.sidebar.text(f"PDFs processed: {len(st.session_state.pdf_contents)}")

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
    pdf_to_view = st.selectbox("Select a PDF to view its content:", list(st.session_state.pdf_contents.keys()))
    if st.button("View PDF Content"):
        st.text_area("PDF Content:", st.session_state.pdf_contents[pdf_to_view], height=300)