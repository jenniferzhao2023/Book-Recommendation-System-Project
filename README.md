# Final-Project-Book-Recommendation-System

## Project/Goals
The book recommendation system provides personalized recommendations based on user interests and preferences.  Additionally, data wrangling and exploratory data analysis will be utilized to draw insights about users’ reading preferences and current trends in the book market.

Key Goals of the Project:
    Base-Case Recommendations for New Users: The recommendation engine will provide initial recommendations for new users based on their past ratings and/or other relevant keywords, ensuring relevant content suggestions.

    Personalized Recommendations through Collaborative Filtering: By utilizing user ID and search preferences, collaborative filtering techniques will generate personalized recommendations for active users based on their activity history and search preferences, enhancing the user experience.

## Dataset:
The dataset contains book information along with user ratings and is available from Kaggle.
    1.	Books.csv: book information(book title, authors, publish year). It consists of 27K rows.
    2.	Ratings.csv: book rating description(user_id, book_id, rating). It consists of 27K rows
    3.	Users.csv: user descriptions (user_id, location, age). It consists of 160K rows
    Due to the large data size, a subset was created by sampling 13,000 records from the original dataset.
## TECH STACK
    Language: Python
    Libraries: Pandas, NumPy, Seaborn
    Sklearn Modules: TfidfVectorizer, cosine_similarity, TruncatedSVD, NearestNeighbors, Sparse_matrix
    Web Tools: Flask, HTML, Streamlit
    Models: KNN (K-Mean) and Truncated SVD Latent Factors model
## Process:
    
    1.Performmed EDA: 
        Conduct EDA to identify duplicates, drop unnecessary columns, and rename column names. Add a new column for modified book titles and use visualizationsto gain insights from the dataset.
        Average Rating and Number of Ratings: Calculate and add the average rating and number of ratings columns for each book and add an average rating column.
        Dataset Integration: Merge the rating dataset and book dataset to combine book information and user ratings.
    2. Data Analysis and Visualization
    3.Model building: 
        Popularity_based: List the top 20 popular books with the most ratings and higher average ratings.
        Content-Based Recommendation: Recommend books to users based on book similarity using Cosine Similarity to calculate the similarity of book features.
        Collaborative Recommendation with SVD:  Apply collaborative filtering techniques to discover patterns and similarities among users. Utilize user-item ratings and interactions to identify users with similar preferences.
    4. Web Deployment: Use Flask, HTML, Streamlit to deploy the model
## Future Goals
Enhancing Content-Based Book Recommendations with a Rich Dataset and Stable Neural Networks.
	1.Dataset: Dataset: Seek a comprehensive dataset with more features (genres, descriptions) to improve the model's performance.
	2.Building a more robust model by leveraging the PyTorch and TensorFlow deep learning libraries.

