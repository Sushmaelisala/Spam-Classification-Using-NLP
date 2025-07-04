# Spam-Classification-Using-NLP

📌 Project Overview
This project focuses on building a Spam Classifier using Natural Language Processing (NLP) techniques and Machine Learning. It processes and analyzes a dataset of SMS messages labeled as "spam" or "ham" (not spam), and builds a model that can automatically classify new messages as spam or not.


📂 Dataset
- The dataset consists of SMS text messages and their corresponding labels:
- ham for non-spam messages
- spam for unwanted/spam messages
- It is loaded and cleaned before feature extraction.


⚙️ Workflow
1. Data Loading

- Read dataset using pandas.

2. Data Preprocessing

- Convert text to lowercase

- Remove punctuation, special characters, and stopwords

- Tokenization and stemming/lemmatization

3. Feature Extraction

- Using TF-IDF Vectorizer or CountVectorizer for transforming text into numerical features.

4. Model Building

- Train-Test Split

* ML models used:

- Naive Bayes

- Support Vector Machines (SVM)

- Logistic Regression (etc.)

- Evaluation

- Accuracy

- Confusion Matrix

- Precision, Recall, F1-score

- Prediction

- Test the classifier on sample messages

🛠️ Technologies Used
- Python

- Jupyter Notebook

- Pandas, Numpy

- Scikit-learn

- NLTK / spaCy (for NLP tasks)

- Matplotlib, Seaborn (for visualization)

📈 Model Performance:

The models are evaluated using:

- Accuracy

- Confusion Matrix

- Precision, Recall, and F1 Score

The best-performing model can be saved and deployed as needed.

