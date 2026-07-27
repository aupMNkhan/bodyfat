# Body Fat Percentage Predictor

A simple machine learning web application that predicts a person's body fat percentage based on physical measurements, using a **Linear Regression** model built with scikit-learn and deployed with Streamlit.

**Live Demo:** https://body-fat-prediction001.streamlit.app

---

## 1. Project Overview

This project takes body measurement data (age, weight, height, chest, abdomen, hip, etc.) and trains a Linear Regression model to predict **Body Fat Percentage**. It includes:

- A data preprocessing pipeline (cleaning, handling missing/invalid values, encoding categorical data)
- A trained regression model with performance evaluation (R² Score, Mean Absolute Error)
- An interactive web interface where a user can input their own measurements and get an instant prediction

---

## 2. Dataset

- **Source:** [Body Fat Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)
- **File used:** `bodyfat.csv`
- **Target column:** `BodyFat`
- **Features used:** Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist (and Sex, if present in the dataset)

Note: The `Density` column (if present) is intentionally excluded, since it is used to mathematically derive the `BodyFat` value itself (via the Siri Equation) in the original data collection — including it would let the model "cheat" instead of genuinely learning from physical measurements.

---

## 3. Requirements

**Python Version:** 3.9 or higher recommended

**Libraries used:**

| Library | Purpose |
|---|---|
| `streamlit` | Builds the interactive web application/UI |
| `pandas` | Loads and processes the dataset |
| `scikit-learn` | Provides the Linear Regression model and evaluation metrics |
| `matplotlib` | Used for optional data visualization (in the standalone script) |

All dependencies are listed in `requirements.txt`:
```
streamlit
pandas
scikit-learn
matplotlib
```

---

## 4. Project Structure

```
bodyfat/
│
├── bodyfat_app.py        # Main Streamlit web application
├── bodyfat.csv            # Dataset (downloaded from Kaggle)
├── requirements.txt        # List of required Python libraries
└── README.md               # Project documentation (this file)
```

---

## 5. Setup Instructions (Step-by-Step)

### Step 1: Install Python
Download and install Python (3.9+) from [python.org](https://www.python.org/downloads/) if not already installed. During installation, make sure to check **"Add Python to PATH"**.

### Step 2: Get the project files
Clone or download this repository, or simply place the following files in one folder:
- `bodyfat_app.py`
- `bodyfat.csv`
- `requirements.txt`

### Step 3: Open the folder in VS Code
Open Visual Studio Code → File → Open Folder → select the project folder.

### Step 4: Open the terminal in VS Code
Go to **Terminal → New Terminal** (or press `` Ctrl + ` ``).

### Step 5: Install the required libraries
Run this command in the terminal:
```bash
pip install -r requirements.txt
```
This reads the `requirements.txt` file and installs all needed libraries automatically.

### Step 6: Run the application
```bash
streamlit run bodyfat_app.py
```

### Step 7: View the app
The terminal will show a local URL, typically:
```
Local URL: http://localhost:8501
```
It should open automatically in your browser. If not, click the link or paste it manually into your browser.

### Step 8: Stop the application
Press `Ctrl + C` in the terminal to stop the app when done.

---

## 6. How the Code Works (Explanation)

1. **Load Data:** The dataset (`bodyfat.csv`) is read using `pandas.read_csv()`.
2. **Clean Data:**
   - Non-informative columns (like `Original`) are removed.
   - Categorical text columns (like `Sex`) are encoded into numbers (M → 0, F → 1) so the model can process them.
   - Any invalid or missing values are converted to `NaN` and removed using `dropna()`.
3. **Split Data:** The dataset is split into **80% training data** and **20% testing data** using `train_test_split()`. The model learns from the training data and is evaluated on the unseen testing data.
4. **Train Model:** A `LinearRegression` model is trained using `.fit()` on the training data — it learns the mathematical relationship between the input features and body fat percentage.
5. **Evaluate Model:** Predictions are made on the test set, and accuracy is measured using:
   - **R² Score** – how well the model explains the variation in the data (closer to 1 is better)
   - **Mean Absolute Error (MAE)** – the average size of prediction errors
6. **Interactive Prediction:** The Streamlit interface lets a user input their own measurements. The app builds a single-row dataset from the input and passes it to `model.predict()` to generate a live prediction.

---

## 7. Model Performance

| Metric | Value (approximate) |
|---|---|
| R² Score | ~0.61 |
| Mean Absolute Error | ~3.33% |

*(Values may vary slightly depending on the random train/test split.)*

---

## 8. Deployment

This app is deployed on **Streamlit Community Cloud** (free hosting for Streamlit apps):

1. Push the project files to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"Create app"**, select the repository, branch, and set the main file path to `bodyfat_app.py`.
4. Click **Deploy**. Streamlit automatically installs dependencies from `requirements.txt` and hosts the app with a public link.

---

## 9. Limitations & Future Improvements

- The model is trained on a small dataset (252 samples), so predictions are approximate and not medically precise.
- Future improvements could include: trying more advanced models (Random Forest, Gradient Boosting), adding more visualizations (correlation heatmap), and cross-validation for more robust accuracy estimates.

---

## 10. Author

Developed as a beginner-level Machine Learning / Regression project to demonstrate the end-to-end workflow: data cleaning → model training → evaluation → deployment.
