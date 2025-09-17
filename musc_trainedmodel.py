import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv("music_dataset.csv")

# Features & target
X = data[['danceability','energy','loudness','speechiness',
          'acousticness','instrumentalness','valence','tempo']]
y = data['genre']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model & scaler
pickle.dump(model, open("music_genre_model.sav", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\n🎵 Model & scaler saved successfully!")
