# =========================================================
# Body Fat Percentage Prediction using Linear Regression
# Dataset: Kaggle "Body Fat Prediction Dataset" (bodyfat.csv)
# =========================================================

# ---- STEP 1: Import libraries ----
# pandas -> to read and handle our data (like an Excel sheet in Python)
import pandas as pd

# train_test_split -> splits our data into "training" and "testing" parts
from sklearn.model_selection import train_test_split

# LinearRegression -> this is our actual AI/ML model
from sklearn.linear_model import LinearRegression

# these 3 are used to check how good our model's predictions are
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# matplotlib -> just to draw a simple graph at the end (optional but nice)
import matplotlib.pyplot as plt


# ---- STEP 2: Load the dataset ----
# Download "bodyfat.csv" from Kaggle and keep it in the SAME folder as this .py file
data = pd.read_csv("bodyfat.csv")


# ---- STEP 3: Explore the data first ----
# Ye sirf isliye print kar rahe hain taake pata chale data kaisa dikhta hai
print("First 5 rows of the dataset:")
print(data.head())          # head() shows top 5 rows by default

print("\nColumn names:")
print(data.columns)         # ye batayega dataset me kaunse columns (features) hain

print("\nAny missing values in each column?")
print(data.isnull().sum())  # isnull().sum() batata hai kis column me kitni values missing hain


# ---- STEP 3.5: Clean the data ----
# Kabhi kabhi CSV file me kisi jagah 'N', 'NA', ya khaali jagah reh jati hai
# jise Python number nahi samajh pata. Isliye har value ko number banane ki
# koshish karte hain, aur jo value convert na ho paye use "NaN" bana dete hain
data = data.apply(pd.to_numeric, errors="coerce")

# Ab jin rows me bhi NaN (missing) value hai, unhe hata dete hain
data = data.dropna()


# ---- STEP 4: Choose Input (X) and Output (y) ----
# 'BodyFat' column wo hai jo hum PREDICT karna chahte hain -> isliye ye our "y" (output/target)
# Baaki saare columns (Age, Weight, Height, Abdomen, etc.) hamare "X" (input/features) hain
# jinke basis par model prediction karega

X = data.drop("BodyFat", axis=1)   # axis=1 means "column ko drop karo", axis=0 hota to row drop hoti
y = data["BodyFat"]


# ---- STEP 5: Split data into Training and Testing sets ----
# Training data -> model isse "seekhta" (learn) hai
# Testing data  -> model ne ye data pehle nahi dekha, isse hum check karte hain model sahi seekha ya nahi

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% data testing ke liye, 80% training ke liye
    random_state=42      # ye number fix rakhne se har baar run karne par same split milega
)


# ---- STEP 6: Create the model ----
# LinearRegression ek line/equation fit karne ki koshish karta hai
# jo input features (X) ko output (y = BodyFat %) se best relate kare
model = LinearRegression()


# ---- STEP 7: Train the model ----
# .fit() function model ko training data dikha kar "seekhne" ka mauka deta hai
model.fit(X_train, y_train)


# ---- STEP 8: Make predictions on test data ----
# Ab model se poochte hain: "is naye/unseen data par BodyFat % kitna hoga?"
y_pred = model.predict(X_test)


# ---- STEP 9: Check how accurate the model is ----
# MAE -> average me prediction kitni units se galat hai (chota better hai)
mae = mean_absolute_error(y_test, y_pred)

# MSE -> jaisa MAE, lekin bade errors ko zyada punish karta hai
mse = mean_squared_error(y_test, y_pred)

# R2 Score -> 0 se 1 ke beech hota hai, 1 ke jitna qareeb utna acha model
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)


# ---- STEP 10: Compare Actual vs Predicted values ----
# Sirf pehli 10 rows dikha rahe hain taake farq clearly samajh aaye
comparison = pd.DataFrame({
    "Actual BodyFat": y_test.values[:10],
    "Predicted BodyFat": y_pred[:10]
})
print("\nActual vs Predicted (first 10 rows):")
print(comparison)


# ---- STEP 11: Plot a simple graph (optional) ----
# Agar predictions perfect hote to sab points ek seedhi line par hote
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Body Fat %")
plt.ylabel("Predicted Body Fat %")
plt.title("Actual vs Predicted Body Fat Percentage")
plt.show()
