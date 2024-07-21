from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
app = Flask(__name__)

# Load the CSV file
df = pd.read_csv('data.csv')

# Load models and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
models = {
    'Logistic Regression': joblib.load('Logistic Regression.pkl'),
    'Random Forest': joblib.load('Random Forest.pkl'),
    'SVM': joblib.load('SVM.pkl'),
    'Naive Bayes': joblib.load('Naive Bayes.pkl')
}

# Preprocess data for accuracy calculation
X = df['text']
y = df['generated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_vec = vectorizer.transform(X_test)

@app.route('/')
def index():
    return render_template('index.html', prediction=None, plot_url=None, 
                           head_data=None, tail_data=None, 
                           performance_data=None)


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    model_choice = request.form['model']
    viz_choice = request.form['visualization']
    show_head = request.form.get('head')
    show_tail = request.form.get('tail')
    show_performance = request.form.get('performance')

    # Model Selection
    model = models.get(model_choice, models['Logistic Regression'])

    # Prediction
    text_vec = vectorizer.transform([text])
    prediction_num = model.predict(text_vec)[0]
    prediction = "Human-generated" if prediction_num == 0 else "AI-generated"
    accuracy = accuracy_score(y_test, model.predict(X_test_vec))

    # Visualization
    plot_url = None
    if viz_choice == 'Pie Chart':
        plt.figure(figsize=(6, 6))
        df['generated'].value_counts().plot.pie(autopct='%1.1f%%')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

    elif viz_choice == 'Box Plot':
        plt.figure(figsize=(6, 6))
        df.boxplot(column=['generated'])
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

    # Prepare Data for Display
    head_data = df.head().to_html() if show_head else None
    tail_data = df.tail().to_html() if show_tail else None
    performance_data = f"Accuracy: {accuracy:.2f}" if show_performance else None

    return render_template('index.html', prediction=prediction, plot_url=plot_url,
                           head_data=head_data, tail_data=tail_data, 
                           performance_data=performance_data)

if __name__ == '__main__':
    app.run(debug=True)
