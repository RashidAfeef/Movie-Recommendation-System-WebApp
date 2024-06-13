import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the model
model = load_model('movie_recommendation_model.h5')

# Load the vectorizer
vectorizer_model_path = 'vectorizer.pkl'
with open(vectorizer_model_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Load the scaler
scaler_model_path = 'scaler.pkl'
with open(scaler_model_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the movie dataset
movie_df = pd.read_csv("imdb_top_1000.csv")
movie_df['Content'] = movie_df['Genre'] + ' ' + movie_df['Overview']

# Normalize the ratings
movie_df[['IMDB_Rating', 'Meta_score']] = scaler.transform(movie_df[['IMDB_Rating', 'Meta_score']])

# Define the recommendation function
def recommend_movies(movie_name, movie_df, vectorizer, model, top_n=10):
    input_movie = movie_df[movie_df['Series_Title'] == movie_name]
    
    if input_movie.empty:
        return f"Movie '{movie_name}' not found in the dataset."
    
    input_content = input_movie['Content'].values[0]
    input_content_sequence = vectorizer(tf.convert_to_tensor([input_content]))
    input_embedding = model.predict([input_content_sequence, input_movie[['IMDB_Rating', 'Meta_score']].values])
    input_embedding = input_embedding / np.linalg.norm(input_embedding, axis=1, keepdims=True)
    all_embeddings = model.predict([vectorizer(movie_df['Content']), movie_df[['IMDB_Rating', 'Meta_score']].values])
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    
    cosine_similarities = np.dot(all_embeddings, input_embedding.T).flatten()
    similar_indices = np.argsort(cosine_similarities)[::-1][1:top_n+1]
    similar_movies = movie_df.iloc[similar_indices]
    
    return similar_movies

# Fetch movie ID from TMDB
def fetch_movie_id(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&query={title}"
    response = requests.get(url)
    data = response.json()
    if data['results']:
        return data['results'][0]['id']
    return None

# Fetch movie poster from TMDB
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return full_path

# Custom CSS for fonts, colors, and layout
st.markdown("""
    <style>
    .header {
        font-size: 40px;
        color: #FF6347;
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
        margin-top: 20px;
    }
    .subheader {
        font-size: 20px;
        color: #4682B4;
        text-align: center;
        font-family: 'Arial Narrow', sans-serif;
        margin-bottom: 20px;
            
    }
    .caption {
        font-size: 16px;
        color: #6A5ACD;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .movie-title {
        color: #2E8B57;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 5px;
    }
    .movie-overview {
        color: #FFFF;
        margin-bottom: 10px;
    }
    .movie-genres {
        color: #4682B4;
        margin-bottom: 10px;
    }
    .movie-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        border: 2px solid #ccc;
        border-radius: 10px;
        background-color: rgba(0, 0, 0, 1);
        padding: 15px;
    }
    .movie-poster {
        margin-right: 20px;
        border-radius: 10px;
    }
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: 100% 100%;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Movie Recommendation</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter your Movie Name</div>', unsafe_allow_html=True)

# Auto-suggest feature for movie titles
movie_titles = movie_df['Series_Title'].tolist()
title = st.selectbox("Movie Title", movie_titles)

if st.button("Show Recommendations"):
    recommendations = recommend_movies(title, movie_df, vectorizer, model)
    
    if isinstance(recommendations, str):
        st.write(recommendations)
    elif not recommendations.empty:
        st.subheader('Recommendations:')
        for idx, row in recommendations.iterrows():
            movie_id = fetch_movie_id(row['Series_Title'])
            if movie_id:
                poster_url = fetch_poster(movie_id)
                st.markdown(f"""
                    <div class="movie-container">
                        <img src="{poster_url}" width="100" class="movie-poster">
                        <div>
                            <div class="movie-title">{row['Series_Title']}</div>
                            <div class="movie-genres">Genres: {row['Genre']}</div>
                            <div class="movie-overview">Overview: {row['Overview']}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="movie-container">
                        <div class="movie-title">{row['Series_Title']}</div>
                        <div class="movie-genres">Genres: {row['Genre']}</div>
                        <div class="movie-overview">Overview: {row['Overview']}</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.write("No similar movies found.")
