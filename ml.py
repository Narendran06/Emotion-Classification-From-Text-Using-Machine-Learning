import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import string

# Ensure nltk data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Expanded sample dataset
def create_sample_data():
    data = {
        'text': [
            'I am so happy today!', 
            'I feel very sad and down.', 
            'This is a wonderful day!', 
            'I am very angry about the situation.', 
            'I am feeling great and excited!', 
            'This is a terrible experience.', 
            'I am so thrilled and happy!', 
            'I am feeling depressed and sad.', 
            'What a fantastic event!', 
            'I am furious with the outcome.',
            'The food was excellent but the service was terrible.',
            'I am elated to receive this award!',
            'I am frustrated with the new policies.',
            'The movie was boring and uninteresting.',
            'I am overjoyed with the progress we have made.',
            'I am anxious about the upcoming test.',
            'The concert was absolutely thrilling!',
            'I feel disappointed by the recent news.'
        ],
        'emotion': [
            'happy', 
            'sad', 
            'happy', 
            'angry', 
            'happy', 
            'sad', 
            'happy', 
            'sad', 
            'happy', 
            'angry',
            'mixed',
            'happy',
            'angry',
            'negative',
            'happy',
            'anxious',
            'happy',
            'disappointed'
        ]
    }
    return pd.DataFrame(data)

# Build a pipeline with text preprocessing and classification
def build_pipeline():
    text_preprocessor = Pipeline(steps=[
        ('vectorizer', TfidfVectorizer(preprocessor=preprocess_text, max_features=1500))
    ])
    
    model = Pipeline(steps=[
        ('preprocessor', text_preprocessor),
        ('classifier', MultinomialNB())
    ])
    
    return model

# Function to plot learning curves
def plot_learning_curves(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=2, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5))
    
    plt.figure(figsize=(10, 7))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Main function
def main():
    data = create_sample_data()
    
    if data.isnull().sum().any():
        print("Missing values found. Dropping them...")
        data = data.dropna()
    
    X = data['text']
    y = data['emotion']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to compare
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Machine': SVC(probability=True)
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        pipeline = Pipeline([
            ('preprocessor', TfidfVectorizer(preprocessor=preprocess_text, max_features=1500)),
            ('classifier', model)
        ])
        
        # Cross-validation
        try:
            scores = cross_val_score(pipeline, X, y, cv=2, scoring='accuracy')
            print(f"{name} Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
        except ValueError as e:
            print(f"Error during cross-validation for {name}: {e}")
            continue
        
        # Fit and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {accuracy:.2f}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
        
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
        cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
        print("Confusion Matrix:\n", cm)
        
        # Plot Confusion Matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.show()
        
        # Plot Learning Curves
        plot_learning_curves(pipeline, X, y)
        
        # ROC Curve (only if there are more than 2 classes and a reasonable number of samples)
        if len(pipeline.classes_) > 2 and len(X_test) > 2:
            y_prob = pipeline.predict_proba(X_test)
            print(f"\ny_prob shape: {y_prob.shape}, y_test dummies shape: {pd.get_dummies(y_test).shape}")
            y_test_dummies = pd.get_dummies(y_test)
            for i in range(min(y_prob.shape[1], y_test_dummies.shape[1])):  # Ensure index is within bounds
                fpr, tpr, _ = roc_curve(y_test_dummies.iloc[:, i], y_prob[:, i])
                plt.figure(figsize=(10, 7))
                plt.plot(fpr, tpr, marker='o', label=f'{name} - {pipeline.classes_[i]}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.grid()
            plt.show()
    
    print(f"\nBest Model: {best_model}")
    
    # Test the prediction function with the best model
    def predict_emotion(text):
        prediction = best_model.predict([text])
        return prediction[0]
    
    new_text = "I am feeling incredibly joyful today!"
    print(f'Predicted Emotion for "{new_text}": {predict_emotion(new_text)}')

if __name__ == '__main__':
    main()
