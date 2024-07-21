# train_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the CSV file
df = pd.read_csv('data.csv')

# Preprocess data
X = df['text']
y = df['generated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Save vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Train and save models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='linear'),
    'Naive Bayes': MultinomialNB()
}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    joblib.dump(model, f'{name}.pkl')

print("Models and vectorizer saved successfully.")
