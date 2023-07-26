# Final-Project-Book-Recommendation-System

## Project/Goals
The book recommendation system provides personalized recommendations based on user interests and preferences. Additionally, data wrangling and exploratory data analysis will be utilized to draw insights about users’ reading preferences and current trends in the book market.

Key Goals of the Project:
1. Base-Case Recommendations for New Users: The recommendation engine will provide initial recommendations for new users based on their past ratings and/or other relevant keywords, ensuring relevant content suggestions.
2. Personalized Recommendations through Collaborative Filtering: By utilizing user ID and search preferences, collaborative filtering techniques will generate personalized recommendations for active users based on their activity history and search preferences, enhancing the user experience.
## Dataset:
The dataset contains book information along with user ratings and is available from Kaggle.
1. Books.csv: book information(book title, authors, publish year). It consists of 27K rows.
2. Ratings.csv: book rating description(user_id, book_id, rating). It consists of 27K rows
3. Users.csv: user descriptions (user_id, location, age). It consists of 160K rows
Due to the large data size, a subset was created by sampling 13,000 records from the original dataset.
## TECH STACK
Language: Python
Libraries: Panda, Numpy, Seaborn
Sklearn Modules: TfidfVectorizer, cosine_sililarity, TruncatedSVD, NearestNeighbors, Sparse_matrix
Web Tools: Flask, HTML
Models: KNN(K-Mean) and Truncated SVD Latent Factors Model
   
## Process:
1. Perfrommed EDA:
   Identify duplicates, check missing values, drop unnecessary columns, and rename column names.
   Calculate and add the average rating and number of rating columns for each book.
   Dataset integration: Merge the rating dataset and book dataset to combine book information and user ratings.
2. Data Analysis and Visualization: Data visualization to gain insights from the dataset.
3. Model building:
   Polularity_based recommendations: List the top 20 popular books with the most ratings and higher average ratings.
   Content_based recommendation: Recommend books to users based on book similarity using Cosine similarity to calculate the similarity of book features.
   Collaborative recommendation with KNN and SVD: apply collaborative filtering techniques to discover patterns and similarities among users. Utilize user-item ratings and interactions to identify users with similar preferences.
4. Web Deployment: Use Flask and HTML to deploy the website.
## Future Goals
Enhancing Content-Based Book Recommendations with a Rich Dataset and Stable Neural Networks.
1. Dataset: Dataset: Seek a comprehensive dataset with more features (genres, descriptions) to improve the model's performance.
2. Building a more robust model by leveraging the PyTorch and TensorFlow deep learning libraries.

