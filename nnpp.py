import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import numpy as np

file_path = r"C:\Users\imsan\OneDrive\Desktop\movie recommendation system\dataset.csv"
dataset = pd.read_csv(file_path)

dataset['genre'].fillna('Unknown', inplace=True)
dataset['overview'].fillna('', inplace=True)
dataset['genre'] = dataset['genre'].str.lower()

dataset['weighted_vote'] = dataset['vote_average'] * dataset['vote_count']

X = dataset[['vote_average', 'vote_count', 'popularity']]
y = np.where(dataset['popularity'] > dataset['popularity'].mean(), 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X_train)

_, indices = knn.kneighbors(X_test)
y_pred = np.zeros(len(X_test))
for i, idx in enumerate(indices):
    avg_vote = X_train.iloc[idx]['vote_average'].mean()
    y_pred[i] = 1 if avg_vote > X_train['vote_average'].mean() else 0

conf_matrix_collab = confusion_matrix(y_test, y_pred)
class_report_collab = classification_report(y_test, y_pred)


tfidf_genres = TfidfVectorizer(stop_words='english')
tfidf_matrix_genres = tfidf_genres.fit_transform(dataset['genre'])

tfidf_overview = TfidfVectorizer(stop_words='english')
tfidf_matrix_overview = tfidf_overview.fit_transform(dataset['overview'])

tfidf_combined = np.hstack([tfidf_matrix_genres.toarray(), tfidf_matrix_overview.toarray()])

X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(tfidf_combined, y, test_size=0.3, random_state=42)

knn_cb = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn_cb.fit(X_train_cb)

_, indices_cb = knn_cb.kneighbors(X_test_cb)
y_pred_cb = np.zeros(len(X_test_cb))
for i, idx in enumerate(indices_cb):
    avg_vote = dataset.iloc[idx]['vote_average'].mean()
    y_pred_cb[i] = 1 if avg_vote > dataset['vote_average'].mean() else 0

conf_matrix_content = confusion_matrix(y_test_cb, y_pred_cb)
class_report_content = classification_report(y_test_cb, y_pred_cb)

ratings = dataset[['id', 'vote_average', 'vote_count']]

svd = TruncatedSVD(n_components=50, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix_overview)

X_train_svd, X_test_svd, y_train_svd, y_test_svd = train_test_split(svd_matrix, y, test_size=0.3, random_state=42)

knn_svd = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn_svd.fit(X_train_svd)

_, indices_svd = knn_svd.kneighbors(X_test_svd)
y_pred_svd = np.zeros(len(X_test_svd))
for i, idx in enumerate(indices_svd):
    avg_vote = dataset.iloc[idx]['vote_average'].mean()
    y_pred_svd[i] = 1 if avg_vote > dataset['vote_average'].mean() else 0

conf_matrix_svd = confusion_matrix(y_test_svd, y_pred_svd)
class_report_svd = classification_report(y_test_svd, y_pred_svd)

print("Confusion Matrix - Collaborative Filtering:\n", conf_matrix_collab)
print("Classification Report - Collaborative Filtering:\n", class_report_collab)

print("Confusion Matrix - Content-Based Filtering:\n", conf_matrix_content)
print("Classification Report - Content-Based Filtering:\n", class_report_content)

print("Confusion Matrix - SVD:\n", conf_matrix_svd)
print("Classification Report - SVD:\n", class_report_svd)

import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


plot_confusion_matrix(conf_matrix_collab, "Confusion Matrix - Collaborative Filtering")

plot_confusion_matrix(conf_matrix_content, "Confusion Matrix - Content-Based Filtering")

plot_confusion_matrix(conf_matrix_svd, "Confusion Matrix - SVD")
