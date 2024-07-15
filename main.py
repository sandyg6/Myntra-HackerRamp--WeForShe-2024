from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import io
import base64
app = Flask(__name__)
nltk.download('stopwords')
myntra_df = pd.read_csv('FashionDataset.csv')
reviews_df = pd.read_csv('MyntraReview.csv')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

myntra_df['description'] = myntra_df['description'].astype(str).apply(clean_text)
myntra_df['Sentiment'] = myntra_df['avg_rating'].apply(lambda x: 1 if x > 3 else 0)
reviews_df['review_text'] = reviews_df['Review'].astype(str).apply(clean_text)
reviews_df['Sentiment'] = reviews_df['Sentiment'].apply(lambda x: 1 if x > 3 else 0)
reviews_df.rename(columns={'review_text': 'description'}, inplace=True)

combined_df = pd.concat([myntra_df[['description', 'Sentiment']], reviews_df[['description', 'Sentiment']]])
X_train, X_test, y_train, y_test = train_test_split(combined_df['description'], combined_df['Sentiment'], test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Print classification report
print(classification_report(y_test, y_pred))

# Clustering function
def cluster_reviews(reviews, n_clusters=2):
    tfidf_matrix = vectorizer.fit_transform(reviews)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    return kmeans, tfidf_matrix

# Cluster positive and negative reviews separately
positive_reviews = combined_df[combined_df['Sentiment'] == 1]['description']
negative_reviews = combined_df[combined_df['Sentiment'] == 0]['description']

kmeans_positive, tfidf_positive = cluster_reviews(positive_reviews, n_clusters=5)
kmeans_negative, tfidf_negative = cluster_reviews(negative_reviews, n_clusters=5)

# Plot clusters function
def plot_clusters(kmeans, tfidf_matrix, title):
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(tfidf_matrix.toarray())
    colors = ["r", "b", "c", "y", "m"]
    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmeans.labels_])
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    # Save plot to memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Encode plot to base64 string
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8').replace('\n', '')
    plt.close()
    return plot_data

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_sentiment():
    description = request.form['text']
    description = clean_text(description)
    X_new = vectorizer.transform([description])
    sentiment = model.predict(X_new)[0]
    sentiment_text = "Positive" if sentiment == 1 else "Negative"
    return render_template('index.html', sentiment=sentiment_text)

@app.route('/clusters')
def show_clusters():
    plot_positive = plot_clusters(kmeans_positive, tfidf_positive, "Positive Reviews Clusters")
    plot_negative = plot_clusters(kmeans_negative, tfidf_negative, "Negative Reviews Clusters")
    return render_template('clusters.html', plot_positive=plot_positive, plot_negative=plot_negative)

if __name__ == '__main__':
    app.run(debug=True)
