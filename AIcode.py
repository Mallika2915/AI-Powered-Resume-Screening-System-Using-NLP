# Install required libraries if not already installed
# !pip install transformers torch tensorflow scikit-learn pandas numpy nltk PyPDF2

import os
import re
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# ----------------------------
# 1. Data Loading (Sample CSV)
# ----------------------------
# Assuming a CSV file with columns: 'Resume_Text' and 'Job_Category'
df = pd.read_csv("resumes_dataset.csv")  # Replace with your dataset path

# ----------------------------
# 2. Text Preprocessing
# ----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Cleaned_Text'] = df['Resume_Text'].apply(clean_text)

# ----------------------------
# 3. Encode Labels
# ----------------------------
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Job_Category'])
num_classes = len(le.classes_)

# ----------------------------
# 4. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['Cleaned_Text'], df['Label'], test_size=0.2, random_state=42, stratify=df['Label']
)

# ----------------------------
# 5. BERT Tokenization & Embedding
# ----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

MAX_LEN = 512  # Maximum tokens per resume

def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []

    for txt in texts:
        encoded = tokenizer.encode_plus(
            txt,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    return np.array(input_ids), np.array(attention_masks)

X_train_ids, X_train_mask = bert_encode(X_train.tolist(), tokenizer)
X_test_ids, X_test_mask = bert_encode(X_test.tolist(), tokenizer)

# ----------------------------
# 6. Generate BERT Embeddings
# ----------------------------
def get_bert_embeddings(input_ids, attention_mask):
    outputs = bert_model(input_ids, attention_mask=attention_mask)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    return cls_embeddings.numpy()

X_train_emb = get_bert_embeddings(X_train_ids, X_train_mask)
X_test_emb = get_bert_embeddings(X_test_ids, X_test_mask)

# ----------------------------
# 7. Convert Labels to One-Hot
# ----------------------------
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# ----------------------------
# 8. Build Classification Model
# ----------------------------
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(768,)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------
# 9. Train Model
# ----------------------------
history = model.fit(
    X_train_emb, y_train_cat,
    validation_data=(X_test_emb, y_test_cat),
    epochs=10,
    batch_size=16
)

# ----------------------------
# 10. Evaluate Model
# ----------------------------
loss, accuracy = model.evaluate(X_test_emb, y_test_cat)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ----------------------------
# 11. Predict New Resume
# ----------------------------
def predict_resume(resume_text):
    cleaned = clean_text(resume_text)
    ids, mask = bert_encode([cleaned], tokenizer)
    emb = get_bert_embeddings(ids, mask)
    pred = model.predict(emb)
    label = le.inverse_transform([np.argmax(pred)])
    return label[0]

# Example:
new_resume = "Experienced data scientist with Python, Machine Learning, and SQL skills."
predicted_category = predict_resume(new_resume)
print(f"Predicted Job Category: {predicted_category}")
