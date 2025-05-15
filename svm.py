import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import nltk
import re
from nltk.stem import WordNetLemmatizer

# Download hanya wordnet (tidak perlu punkt)
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Tokenisasi + lemmatization manual tanpa word_tokenize
def to_lemmas(text):
    # Ambil kata-kata huruf kecil (a-z) saja
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    return [lemmatizer.lemmatize(word) for word in tokens]

# === Load Dataset ===
df = pd.read_csv("Training.txt", sep='\t', header=None, names=['liked', 'text'])
print("Jumlah data:", len(df))
print("Contoh data:")
print(df.head())

# === Preprocessing ===
df['lemmatized'] = df['text'].apply(lambda x: ' '.join(to_lemmas(x)))

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['lemmatized'])
y = df['liked']

# === Split train-test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Training SVM ===
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# === Evaluasi ===
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Dislike', 'Like'],
            yticklabels=['Dislike', 'Like'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# === Contoh Prediksi ===
samples = [
    "the da vinci code is awesome",
    "the movie was boring and bad"
]
for s in samples:
    s_lem = ' '.join(to_lemmas(s))
    s_vec = vectorizer.transform([s_lem])
    pred = model.predict(s_vec)[0]
    print(f"'{s}' => {'Liked' if pred == 1 else 'Disliked'}")
