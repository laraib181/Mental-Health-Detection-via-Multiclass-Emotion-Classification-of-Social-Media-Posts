# Mental-Health-Detection-via-Multiclass-Emotion-Classification-of-Social-Media-Posts
# 🧠 Emotion Detection from Social Media Posts using NLP & Machine Learning

## 📄 Abstract

In today’s digital age, the increasing use of social media platforms has resulted in a vast stream of emotionally rich textual content. This presents a significant opportunity to monitor mental well-being using artificial intelligence. This project aims to detect and classify human emotions from social media posts (primarily tweets) using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.

The end-to-end system includes data preprocessing, TF-IDF-based feature extraction, multi-model training (Logistic Regression, Naive Bayes, and Support Vector Machine), performance evaluation, and deployment using a user-friendly **Gradio interface**. It classifies tweets into one of six emotions: **joy, sadness, anger, fear, surprise, and love**.

The solution serves as a proof-of-concept for mental health monitoring, demonstrating how AI can help identify emotional distress patterns from online text. The SVM model was found to outperform others in overall accuracy, and the application is accessible for real-time emotion prediction.

---

## 📌 Introduction

As concerns for emotional well-being grow globally, integrating **AI into mental health monitoring** has become increasingly important. Social media posts are often a direct expression of users' emotional states. This project leverages NLP and ML to interpret and classify these expressions into specific emotion categories.

### 🎯 Objectives

- Build a **Multiclass Emotion Classification System** using NLP and ML.
- Identify six emotions from tweet text: **joy, sadness, anger, fear, love, surprise**.
- Deploy the model for **real-time interaction** through a simple web interface.
- Promote early detection and emotional awareness through AI.

---

## ❓ Problem Statement

- Mental health issues often go undiagnosed due to lack of direct evaluation.
- Social media posts reflect emotions, but **manual analysis is impractical**.
- There’s a need for an **automated, scalable** solution to classify emotions from text.
- This project creates a **tweet-based emotion classifier** to address this gap.

---

## 🔍 Importance of the Study

- Provides an **AI-based tool** for detecting emotions from real-world social data.
- Assists researchers, counselors, and digital platforms in understanding emotional trends.
- Demonstrates practical use of NLP and ML in **psychological health contexts**.
- Lays groundwork for future **emotion-aware applications** in mental health tech.

---

## 📚 Methodology

### 1. Data Collection

- **Dataset**: Publicly available tweet dataset labeled with emotions.
- **Samples**: 10,000+ tweets
- **Labels**: `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`
- **Source**: Twitter (anonymized, preprocessed)
- **Format**: `.txt` files with tweet;label format

---

### 2. Data Preprocessing

Performed comprehensive cleaning to ensure quality input:

- Lowercasing
- Punctuation and emoji removal
- Stopword removal
- Tokenization and lemmatization
- Label encoding for model compatibility

---

### 3. Feature Extraction

Used **TF-IDF Vectorization** to convert text to numerical format:

- Prioritizes words that are unique and informative.
- Ensures dimensional consistency for ML models.
- Results in sparse matrix input for model training.

---

### 4. Model Training & Comparison

Trained and evaluated three classic ML models:

| Model                | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.84     | 0.84      | 0.84   | 0.83     |
| Naive Bayes         | 0.71     | 0.75      | 0.72   | 0.69     |
| SVM (LinearSVC)     | 0.86     | 0.86      | 0.86   | 0.86     |

- **SVM (LinearSVC)** emerged as the top performer.
- All models were trained using the same TF-IDF features for fair comparison.

---

### 5. Exploratory Data Analysis (EDA)

To understand data distribution and patterns:

- Visualized emotion label distribution using bar charts.
- Analyzed most frequent words per emotion category.
- Used pie charts for emotional trend summaries.

---

### 6. Model Evaluation

Evaluated models using standard metrics:

- **Accuracy**: Overall correctness
- **Precision**: How many predicted positives were correct
- **Recall**: How many actual positives were found
- **F1-score**: Balance of precision and recall

Also tested on:

- **Held-out official test set** — Models maintained consistency
- **Unseen real-world tweets** — Demonstrated robust generalization

---

### 7. Deployment via Gradio

To make the project accessible:

- Developed an **interactive Gradio app**.
- Allows real-time prediction from all three models.
- Users enter a tweet and receive predicted emotion labels instantly.
- Deployed with public access for demonstrations and usability testing.

---

## 🎉 Example Predictions

**Tweet**: “I’m feeling so low and empty today.”  
- Logistic Regression: Sadness  
- Naive Bayes: Sadness  
- SVM: Sadness ✅

**Tweet**: “This surprise party made me feel so loved!”  
- Logistic Regression: Surprise  
- Naive Bayes: Joy  
- SVM: Surprise ✅

---

## 🧪 Testing & Results

The models were tested on:

1. **Official test set**  
   - Predictions aligned with validation results.
2. **Custom unseen tweets**  
   - High agreement among classifiers.
   - No significant performance drop on non-training data.

---

## 🚀 Conclusion

This project successfully built and deployed a complete NLP-based **Emotion Detection System** using traditional ML models. From data preprocessing to deployment, the system demonstrates how **AI can support mental health awareness** by interpreting emotional cues from text.

**SVM** proved to be the most effective model, and the final **Gradio interface** makes it easily accessible for non-technical users.

---

## 🔮 Future Work

- **Use Deep Learning**: Incorporate models like BERT or LSTM for improved context understanding.
- **Scale the Dataset**: Use larger, more diverse datasets for better generalization.
- **Multilingual Support**: Extend support to non-English tweets.
- **User Feedback Loop**: Allow users to verify or correct predictions.
- **Mobile/Web Integration**: Deploy on mobile or web apps for real-time emotion monitoring.

---

## 🤝 Contributors

Each team member contributed to specific modules including:

- Data preprocessing
- Feature extraction
- Model development
- EDA and visualization
- Deployment and UI design

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Gradio
- TQDM, Regex, NLTK

---

## 📫 Contact

For queries, feedback, or contributions:  
**Email:** laraib.saleha12@gmail.com  


