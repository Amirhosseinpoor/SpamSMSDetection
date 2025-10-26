# ======================================================
# SPAM SMS CLASSIFICATION (TXT DATASET VERSION)
# CPU FRIENDLY — COMPLETE PIPELINE
# ======================================================

# Import Libraries
# ------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from wordcloud import WordCloud

# NLP + ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# ======================================================
# Load the Dataset (from TXT file)
# ======================================================

# CHANGE this path to your .txt file
data_path = "Spam SMS Collection.txt"

# The dataset is tab-separated: [label \t message]
df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')

print("✅ Dataset loaded successfully!")
print("\nData Sample:")
print(df.head())

# ======================================================
# Data Cleaning
# ======================================================
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
print("\nDataset shape:", df.shape)
print("Number of spam messages:", df['label_num'].sum())
print("Number of ham messages:", len(df) - df['label_num'].sum())

# ======================================================
# Exploratory Data Analysis (EDA)
# ======================================================

plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df, palette='Set2')
plt.title('Distribution of Spam vs Ham Messages')
plt.show()

df['message_len'] = df['message'].apply(len)
plt.figure(figsize=(8,5))
sns.histplot(df[df['label']=='ham']['message_len'], bins=40, color='green', label='Ham')
sns.histplot(df[df['label']=='spam']['message_len'], bins=40, color='red', label='Spam')
plt.legend()
plt.title('Message Length Distribution')
plt.show()

spam_words = ' '.join(df[df['label']=='spam']['message'])
ham_words = ' '.join(df[df['label']=='ham']['message'])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(WordCloud(width=500, height=300, background_color='white').generate(spam_words))
plt.title('Spam WordCloud')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(WordCloud(width=500, height=300, background_color='white').generate(ham_words))
plt.title('Ham WordCloud')
plt.axis('off')

plt.show()

# ======================================================
# Text Preprocessing
# ======================================================
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['clean_msg'] = df['message'].apply(clean_text)
print("\n✅ Text cleaning complete!")
print(df[['message', 'clean_msg']].head())

# ======================================================
# Split Dataset
# ======================================================
X = df['clean_msg']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ======================================================
# Feature Extraction (TF-IDF)
# ======================================================
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF Vocabulary Size:", len(tfidf.vocabulary_))

# ======================================================
# Model Training (Naive Bayes)
# ======================================================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("\n✅ Model trained successfully!")

# ======================================================
# Model Evaluation
# ======================================================
y_pred = model.predict(X_test_tfidf)

print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ======================================================
# Custom Message Prediction
# ======================================================
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_tfidf = tfidf.transform([msg_clean])
    pred = model.predict(msg_tfidf)[0]
    return "🚨 SPAM" if pred == 1 else "✅ HAM"

print("\nType your own messages to test the model!")
while True:
    msg = input("\nEnter a message (or type 'exit'): ")
    if msg.lower() == 'exit':
        print("👋 Exiting program.")
        break
    print("Prediction:", predict_message(msg))
