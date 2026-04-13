# Fake News Detection using Machine Learning

## Overview

This project builds a machine learning model to classify news articles as **REAL** or **FAKE** using Natural Language Processing (NLP).
It uses TF-IDF vectorization and a Support Vector Machine (SVM) model to analyze textual data.


## Project Structure

```ascii
fake-news-detector/
│── fake_news_detector.ipynb
│── sample_news.csv (optional)
│── README.md
```

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* LinearSVC (SVM)


## Dataset

Link to dataset : https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view

The dataset contains news articles with the following columns:

* **Unnamed: 0**
* **title**
* **text**
* **label (REAL / FAKE)**

For processing, title and text are combined into a single column: **content** and unnecessary columns like Unnamed: 0 is removed.


## Methodology

1. Data Cleaning

   * Removed unnecessary columns
   * Combined title and text
   * Applied lowercasing and regex cleaning

2. Feature Extraction

   * Converted text into numerical features using TF-IDF

3. Model Training

   * Trained using Support Vector Machine (LinearSVC)

4. Evaluation

   * Accuracy score
   * Classification report (precision, recall, F1-score)

5. Prediction

   * Custom function to classify new/unseen news articles


## Results

* Accuracy: **~93%**
* Balanced performance on both REAL and FAKE classes


## Example Usage

Input:
"""<_news-article_>"""

Output:
REAL/FAKE

## Limitations

* Performs best on structured news articles
* Struggles with short or informal text
* Cannot detect sarcasm or nuanced misinformation
* Depends on dataset quality


## Future Improvements

* Use advanced NLP models like BERT
* Improve handling of short text inputs
* Deploy as a web application using Flask
* Add confidence scoring for predictions


## Note

This project was developed as part of a Data Science mini project for academic coursework.

