import pandas as pd 
import numpy as np
from scipy.sparse import csr_matrix as sparse_matrix
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from flask import Flask,request,render_template
from pathlib import Path

app = Flask(__name__)

THIS_FOLDER = Path(__file__).parent.resolve()
books_info_file = THIS_FOLDER / "books_info_svd.csv"
books_ratings_file = THIS_FOLDER / "book_ratings_svd.csv"
# Load the dataset
book_info = pd.read_csv(books_info_file)
book_ratings = pd.read_csv(books_ratings_file)

# Define function to build a user-book matrix based on book ratings.
book_id_mapping = {book_id: idx for idx, book_id in enumerate(book_info['book_id'].unique())}
user_id_mapping = {user_id: idx for idx, user_id in enumerate(book_ratings['user_id'].unique())}
	
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bookurl', methods=['GET', 'POST'])
def bookurl():
    if request.method == 'POST':
        #return render_template('bookurl.html', shortcode=request.form['userid'])
        user_id=request.form['userid']
        booklist=getbooks(user_id)
        recommended_books=book_info[book_info.book_id.isin(booklist)].values.tolist()
        return render_template('bookurl.html', results=recommended_books)
    elif request.method == 'GET':
        return 'A GET request was made'
    else:
        return 'Not a valid request method for this route'
	
def build_user_book_matrix(book_info, book_ratings,user_id_mapping,book_id_mapping):
    
    # Create mappings for book IDs and user IDs
    #book_id_mapping = {book_id: idx for idx, book_id in enumerate(book_info['book_id'].unique())}
    #user_id_mapping = {user_id: idx for idx, user_id in enumerate(book_ratings['user_id'].unique())}

    num_users = len(user_id_mapping)
    num_books = len(book_id_mapping)
    user_book_matrix = np.zeros((num_users, num_books))
    
    # Fill the user-book matrix with ratings
    try:
        for _, row in book_ratings.iterrows():
            user_id = user_id_mapping[row['user_id']]
            book_id = book_id_mapping[row['book_id']]
            rating = row['rating']
            user_book_matrix[user_id, book_id] = rating
    except Exception:
        print("exception")
        pass

    return user_book_matrix
    
# Define function to perfrom SVC on the user_book matrix and train a nearest neighbors model

def train_models(user_book_matrix, num_latent_factors=10, num_neighbors=5):
    
    # Perform Truncated SVD on the user-book matrix
    svd = TruncatedSVD(n_components=num_latent_factors, random_state=42)
    svd_model = svd.fit_transform(user_book_matrix)
    
    # Train a nearest neighbors model on the SVD-transformed matrix
    knn_model = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
    knn_model.fit(svd_model)

    return svd_model, knn_model
    
# Use model to generate book recommendations for a given user

def get_recommendations(user_id, svd_model, knn_model,user_id_mapping, book_info, book_ratings, num_recommendations=10):
    
    # Get the embedding of the target user
    user_embedding = svd_model[user_id].reshape(1, -1)
    
    # Find similar users based on the nearest neighbors model
    _, indices = knn_model.kneighbors(user_embedding)
    similar_user_ids = indices.flatten()
    #print("similar_user_ids")
    #print(similar_user_ids)
    recommendations = []
    for similar_user_id in similar_user_ids:
        similar_user_id_mapping=[k for k, v in user_id_mapping.items() if v == similar_user_id][0]
        rated_books = set(book_ratings.loc[book_ratings['user_id'] == similar_user_id_mapping, 'book_id'])
        unrated_books = [book_id for book_id in book_info['book_id'].unique() if book_id not in rated_books]
        recommendations.extend(book_id for book_id in rated_books if book_id not in recommendations)
        if len(recommendations) >= num_recommendations:
            break
    recommendations = recommendations[:num_recommendations]
    recommended_books = [book_info.loc[book_info['book_id'] == book_id, 'title'].values[0] for book_id in recommendations]
    return recommendations

user_book_matrix = build_user_book_matrix(book_info, book_ratings,user_id_mapping,book_id_mapping)
svd_model, knn_model = train_models(user_book_matrix)
	
# Build the user-book matrix and train the models
def getbooks(user_id):

    #print(svd_model)
    #print(knn_model)
    # Get book recommendations for a specific user and print the recommended book titles
    #user_id = 0 #an example
    recommendations = get_recommendations(int(user_id), svd_model, knn_model, user_id_mapping,book_info, book_ratings)
    #print("Recommended books:")
    #for book_title in recommendations:
        #print(book_title)
    return recommendations

if __name__ == '__main__':  
    app.run(debug=True)