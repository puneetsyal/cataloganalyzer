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

# Set NLTK data path and download data (as before)
# ...

# Initialize session state (as before)
# ...

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

# Rest of the functions (read_pdf, match_brief) remain the same
# ...

st.title('Furniture Catalog Matcher')

# PDF processing and matching logic (as before)
# ...

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

# Rest of the Streamlit UI code (as before)
# ...