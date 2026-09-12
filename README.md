# 🍕 Zomato Rating Prediction [2025]

Predict restaurant ratings using ensemble machine learning techniques on Zomato Bengaluru data to help restaurant owners and business analysts make data-driven decisions.
---

## 📌 Problem Statement

Restaurant ratings on discovery platforms like Zomato are heavily influenced by a combination of key attributes, including location, cuisine options, pricing structure, restaurant type, and customer interaction metrics. 

Understanding the underlying factors that drive positive ratings allows restaurant owners to optimize their operational strategies and helps prospective business owners choose the right parameters for success. This project builds a regression pipeline to predict restaurant ratings based on feature engineering of the Zomato Bengaluru dataset.

---

## 🛠️ Tech Stack & Tools

* **Programming Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning Framework:** Scikit-Learn (*ExtraTreesRegressor*)
* **Web Deployment:** Flask

---

## ⚙️ Project Architecture & Approach

The regression pipeline employs an **Extra Trees Regressor** (Extremely Randomized Trees) — an ensemble learning technique that reduces variance and captures complex, non-linear relationships across restaurant parameters better than standard decision trees or linear models.

### Key Features Used:
* **Location:** Neighborhood and geographic area within Bengaluru.
* **Cuisine Type:** Single or multi-cuisine offerings.
* **Cost / Price Range:** Average cost for two people.
* **Restaurant Category/Type:** Casual Dining, Quick Bytes, Cafe, Fine Dining, etc.
* **Online Order & Table Booking:** Availability of online ordering and table reservation features.
* **Customer Feedback & Votes:** Count of user reviews and total engagement metrics.

### Machine Learning Pipeline:
1. **Data Cleaning & Preprocessing:** Handling null values, parsing rating strings (e.g., converting `"4.1/5"` to numeric floats), and removing duplicate entries.
2. **Feature Encoding & Scaling:** One-Hot and Target Encoding for high-cardinality categorical variables like location and cuisine.
3. **Train-Test Split:** Partitioning dataset into training and evaluation sets.
4. **Model Training:** Fitting the **Extra Trees Regressor** on engineered features.
5. **Hyperparameter Tuning:** Fine-tuning tree depth, number of estimators, and min-sample splits.
6. **Evaluation:** Scoring performance via continuous regression metrics ($R^2$, MAE, RMSE).

---


   ```bash
   git clone [https://github.com/Deepshikha1308/ml-project.git](https://github.com/Deepshikha1308/ml-project.git)
   cd ml-project
