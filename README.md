# Fake News Detection using Machine Learning

## 📌 Overview

This project builds a machine learning system to classify news articles as **REAL** or **FAKE** using Natural Language Processing (NLP).

It uses **TF-IDF vectorization** and a **Support Vector Machine (LinearSVC)** model for text classification, along with a simple **Streamlit web interface** for real-time predictions.

---

## 🚀 Project Structure

```ascii
fake-news-detector/
│── fake_news_detector.ipynb  
│── app.py                   
│── model.pkl                 
│── tfidf.pkl                 
│── sample_news.csv 
│── README.md
````

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* LinearSVC (SVM)
* Streamlit

## Dataset

Dataset link:
[https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view](https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)

The dataset contains:

* **title**
* **text**
* **label (REAL / FAKE)**

### Preprocessing:

* Removed unnecessary columns (e.g., `Unnamed: 0`)
* Combined **title + text → content**
* Applied lowercasing and regex-based cleaning

## Methodology

1. **Data Preprocessing**

   * Cleaning text
   * Feature preparation

2. **Feature Extraction**

   * TF-IDF vectorization

3. **Model Training**

   * LinearSVC (Support Vector Machine)

4. **Evaluation**

   * Accuracy Score
   * Precision, Recall, F1-score

5. **Prediction**

   * Custom function for classifying new input

6. **Deployment**

   * Streamlit-based UI for real-time usage


## Results

* Accuracy: **~93%**
* Balanced performance on both REAL and FAKE classes


## Example Usage

Input:

```
<news-article>
```

Output:

```
REAL/FAKE
```

## Running the App

```bash
streamlit run app.py
```

## Limitations

* Performs best on structured news articles
* Struggles with short or informal text
* Cannot detect sarcasm or nuanced misinformation
* Depends heavily on dataset quality


## Future Improvements

* Use advanced NLP models (e.g., BERT)
* Improve handling of short text
* Add confidence-based filtering
* Deploy as a full web application

## Note

This project was developed as part of a Data Science mini project for academic coursework.




