import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv")

# Rename columns
df.columns = ["label", "message"]

# Convert labels into numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Convert text into vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom message
message = input("Enter a message: ")
message_vector = vectorizer.transform([message])

if model.predict(message_vector)[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")