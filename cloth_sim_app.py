import streamlit as st


# from flask import Flask, jsonify, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import unicodedata
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    stop = stopwords.words('english')
    lem = WordNetLemmatizer()
    text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore').lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [lem.lemmatize(w) for w in words if w not in stop]

def concatenate_text_columns(df, column_names, new_column_name):
    df[new_column_name] = df[column_names].apply(lambda x: ' '.join(x), axis=1)
    return df

def apply_preprocessing(df, text_column):
    df['preprocessed_text'] = df[text_column].apply(preprocess_text)
    df['preprocessed_text'] = df['preprocessed_text'].apply(lambda x: ' '.join(x))
    return df

# Load the data
df = pd.read_csv('shoppers_stop (1).csv')

# Load the pre-calculated embeddings
embeddings = np.load('ss_embeddings.npy')

def calculate_cosine_similarity(df, text_column, search_keyword):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_pairs = [(text, search_keyword) for text in df[text_column]]
    # embeddings = model.encode(sentence_pairs)
    # # Save the embeddings to a NumPy array or a file
    # np.save('ss_embeddings.npy', embeddings)
    search_keyword_embedding = model.encode([(search_keyword, search_keyword)])[0]
    search_keyword_embedding = search_keyword_embedding.reshape(1, -1)
    cosine_similarities = cosine_similarity(embeddings, search_keyword_embedding)
    df['bert_cosine_similarity'] = cosine_similarities.flatten()
    return df

def cloth_similarity(df, search_keyword, top_N, text_column_names=["name", "desc", "class"]): 
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df["Product_Description"] = df["desc"].apply(str)
    df["Product_Name"] = df["name"].apply(str)
    df["Product_Class"] = df["class"].apply(str)
    df = concatenate_text_columns(df, text_column_names, "Full_Des")
    df = apply_preprocessing(df, "Full_Des")  
    df = df[df['preprocessed_text'].map(len) > 0]
    df = calculate_cosine_similarity(df, "preprocessed_text", search_keyword) 
    df_sorted = df.sort_values('bert_cosine_similarity', ascending=False)
    df_top_n = df_sorted.head(top_N)
    df_ranked_list = df_top_n[['name', 'link', 'bert_cosine_similarity']]
    return df_ranked_list 

@st.cache_data
def search_links(keyword):
    res = cloth_similarity(df, keyword, top_N=5, text_column_names=["name", "desc", "class"])
    # Perform search logic here
    # You can use the keyword to fetch relevant links from a database or perform web scraping
    # Return a list of links
    print(res)
    return res

# Set page title
st.title("Clothing Similarity Search")

# Add a search bar
search_input = st.text_input("Enter your search keyword")

# Check if the search button is clicked
if st.button("Search"):
    if search_input:
        # Perform search and get links
        links = search_links(search_input)

         # Display the filtered DataFrame
        st.dataframe(links)

        # Display the number of results
        st.markdown(f"Number of results: {len(links)}")

        # Display the links
        st.markdown("### Search Results:")
        for link in links['link']:
            st.markdown(f"- [{link}]({link})")

    else:
        st.warning("Please enter a search keyword")
