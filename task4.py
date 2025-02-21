import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

local_file = "spam.csv"
try:
    data = pd.read_csv(local_file, encoding='latin-1')
except FileNotFoundError:
    print(f"Error: The file {local_file} was not found. Please download it manually.")
    sys.exit(1) 

data = data.iloc[:, [0, 1]]  
data.columns = ["label", "message"]

data['label'] = data['label'].map({'ham': 0, 'spam': 1})  


if data.empty:
    print("Error: Loaded dataset is empty. Please check the file content.")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

if X_train.empty or X_test.empty:
    print("Error: Train-test split resulted in empty datasets. Check input data.")
    sys.exit(1)


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
