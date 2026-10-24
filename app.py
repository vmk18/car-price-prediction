import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Prediction")
st.write(
    "Predict the estimated selling price of a used car "
    "using Machine Learning."
)

@st.cache_data
def load_data():
    data = pd.read_csv("car data.csv")

    required = [
        "Car_Name", "Year", "Selling_Price",
        "Present_Price", "Kms_Driven", "Fuel_Type",
        "Seller_Type", "Transmission", "Owner"
    ]

    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    data = data[required].dropna().copy()

    data["Fuel_Type"] = data["Fuel_Type"].map({
        "Petrol": 0, "Diesel": 1, "CNG": 2
    })
    data["Seller_Type"] = data["Seller_Type"].map({
        "Dealer": 0, "Individual": 1
    })
    data["Transmission"] = data["Transmission"].map({
        "Manual": 0, "Automatic": 1
    })

    data = data.dropna()
    return data

@st.cache_resource
def train_model():
    data = load_data()

    X = data.drop(["Car_Name", "Selling_Price"], axis=1)
    y = data["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=2
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    score = r2_score(y_test, test_predictions)

    return model, list(X.columns), score

try:
    model, features, test_score = train_model()
except Exception as error:
    st.error(f"Could not load or train the model: {error}")
    st.stop()

st.caption(
    f"Model test R² score on the held-out split: {test_score:.3f}"
)

with st.form("car_details"):
    year = st.number_input(
        "Year of manufacture", 2000, 2026, 2018
    )
    present_price = st.number_input(
        "Present price (₹ lakh)",
        min_value=0.1,
        value=8.0,
        step=0.1
    )
    kms_driven = st.number_input(
        "Kilometres driven",
        min_value=0,
        value=30000,
        step=1000
    )
    fuel = st.selectbox(
        "Fuel type", ["Petrol", "Diesel", "CNG"]
    )
    seller = st.selectbox(
        "Seller type", ["Dealer", "Individual"]
    )
    transmission = st.selectbox(
        "Transmission", ["Manual", "Automatic"]
    )
    owner = st.selectbox(
        "Previous owners", [0, 1, 2, 3]
    )

    submitted = st.form_submit_button("Predict Price")

if submitted:
    values = {
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": {
            "Petrol": 0, "Diesel": 1, "CNG": 2
        }[fuel],
        "Seller_Type": {
            "Dealer": 0, "Individual": 1
        }[seller],
        "Transmission": {
            "Manual": 0, "Automatic": 1
        }[transmission],
        "Owner": owner
    }

    input_data = pd.DataFrame(
        [[values[col] for col in features]],
        columns=features
    )

    prediction = float(model.predict(input_data)[0])

    if prediction < 0:
        st.warning(
            f"The model returned ₹{prediction:.2f} lakh. "
            "A negative estimate is not a realistic selling price; "
            "please treat this prediction as unreliable."
        )
    else:
        st.success(
            f"Estimated selling price: ₹{prediction:.2f} lakh"
        )

    st.caption(
        "This is a machine-learning estimate, not a guaranteed market price."
    )
