# =========================================================
# Body Fat Percentage Prediction - Interactive Web App
# Built using Streamlit (simple way to turn Python into a web app)
# Model: Linear Regression
# =========================================================

# ---- STEP 1: Import libraries ----
import streamlit as st                # streamlit banata hai humari web app ka UI
import pandas as pd                    # data handle karne ke liye
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ---- STEP 2: Page settings (sirf UI ko thoda acha dikhane ke liye) ----
st.set_page_config(page_title="Body Fat Predictor", page_icon="💪")
st.title("💪 Body Fat Percentage Predictor")
st.write("This app uses a Linear Regression model to predict your Body Fat Percentage.")


# ---- STEP 3: Load and train the model ----
# @st.cache_data ka matlab: ye function baar baar mat chalao,
# ek dafa data load/train ho jaye to yaad rakho (app fast chalti hai)
@st.cache_data
def load_data_and_train_model():
    data = pd.read_csv("bodyfat.csv")

    # 'Density' column ko hata rahe hain kyunke wo BodyFat nikalne ke formula
    # (Siri Equation) me already use hoti hai -> isse rakhna "cheating" jaisa hoga
    if "Density" in data.columns:
        data = data.drop("Density", axis=1)

    # 'Original' column sirf Y/N (ek label) hai, prediction ke liye useful nahi -> hata dete hain
    if "Original" in data.columns:
        data = data.drop("Original", axis=1)

    # 'Sex' column me text values (M/F) hoti hain. Model sirf numbers samajhta hai,
    # isliye hum M ko 0 aur F ko 1 me convert kar dete hain (isko "encoding" kehte hain)
    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].map({"M": 0, "F": 1})

    # Ab baaki bachi columns ko number banane ki koshish karte hain (agar kahin
    # 'N'/'NA' jaisi ajeeb value reh gayi ho to wo NaN ban jayegi)
    data = data.apply(pd.to_numeric, errors="coerce")

    # Jin rows me bhi NaN (missing) value hai, unhe hata dete hain
    data = data.dropna()

    X = data.drop("BodyFat", axis=1)   # input features
    y = data["BodyFat"]                # output/target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)        # model ko training data se seekhna

    # Test data par accuracy check kar rahe hain, taake user ko dikha sakein
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, X.columns, mae, r2   # sab kuch wapas bhej rahe hain jo hume baad me chahiye


# Function ko call karke model aur uski info hasil kar rahe hain
model, feature_names, mae, r2 = load_data_and_train_model()


# ---- STEP 4: Show model accuracy on the page ----
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)             # page ko 2 columns me divide kiya
col1.metric("R2 Score", f"{r2:.2f}")   # kitna acha model fit hua (1 ke qareeb = behtar)
col2.metric("Average Error", f"{mae:.2f} %")  # average kitna galat predict karta hai


# ---- STEP 5: Take user input for prediction ----
st.subheader("🔢 Enter Your Details")
st.write("Enter your body measurements below (in inches) to see your predicted Body Fat Percentage:")

# Har feature ke liye ek number input box bana rahe hain
# Hum ek "dictionary" (user_input) me sab values store kar rahe hain
user_input = {}

# Do columns me inputs dikha rahe hain taake UI zyada saaf lage
left_col, right_col = st.columns(2)

for i, feature in enumerate(feature_names):
    # Half features left column me, half right column me daal rahe hain
    if i % 2 == 0:
        user_input[feature] = left_col.number_input(f"{feature}", value=0.0, step=1.0)
    else:
        user_input[feature] = right_col.number_input(f"{feature}", value=0.0, step=1.0)


# ---- STEP 6: Predict button ----
if st.button("Predict Body Fat %"):
    # user_input dictionary ko ek row wale DataFrame me convert kar rahe hain
    # kyunke model.predict() ko DataFrame/array chahiye hota hai, single value nahi
    input_df = pd.DataFrame([user_input])

    # model se prediction le rahe hain
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Body Fat: **{prediction:.2f}%**")

    # Thoda context dene ke liye simple categories bhi dikha dete hain
    if prediction < 14:
        st.info("Category: Athletic / Fitness range")
    elif prediction < 25:
        st.info("Category: Average range")
    else:
        st.info("Category: Above average range")
