from flask import Flask, render_template, request

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the CSV file into a DataFrame
df = pd.read_csv('main_data.csv')  # Replace 'your_movie_data.csv' with the actual file path

# Select the relevant columns for the recommendation model
selected_columns = ['movie_title', 'genres', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
df = df[selected_columns]

# Fill any missing values with an empty string
df = df.fillna('')

# Combine relevant columns into a single column for text processing
df['combined_features'] = df['genres'] + ' ' + df['director_name'] + ' ' + df['actor_1_name'] + ' ' + df['actor_2_name'] + ' ' + df['actor_3_name']

# Create a TF-IDF Vectorizer to convert text data into numerical vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features into a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Calculate cosine similarity between movies based on TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies(movie_title, cosine_sim=cosine_sim):
    # Check if the movie title exists in the DataFrame
    if movie_title not in df['movie_title'].values:
        return []

    # Get the index of the movie that matches the title
    idx = df[df['movie_title'] == movie_title].index[0]

    # Get the pairwise similarity scores with other movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 similar movies (excluding the input movie itself)
    sim_scores = sim_scores[1:2]

    # Get the movie indices
    movie_indices = [score[0] for score in sim_scores]

    # Return the titles of the recommended movies
    return df['movie_title'].iloc[movie_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    input_movies = []
    for i in range(5):
        movie_title = request.form.get(f'movie{i+1}')
        input_movies.append(movie_title)

    recommendations_list = {}
    for movie in input_movies:
        recommendations_list[movie] = recommend_movies(movie)
        

    return render_template('recommendations.html', recommendations=recommendations_list)

if __name__ == '__main__':
    app.run(debug=True)
