import math
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bike Demand Predictor", page_icon="🚲", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/bike_model.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    return model, feature_cols

model, feature_cols = load_artifacts()

def build_input_df(season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, hum, windspeed):
    hr_sin = math.sin(2 * math.pi * hr / 24)
    hr_cos = math.cos(2 * math.pi * hr / 24)

    input_data = {
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp,
        "hum": hum,
        "windspeed": windspeed,
        "hr_sin": hr_sin,
        "hr_cos": hr_cos,
    }

    input_df = pd.DataFrame([input_data])
    return input_df[feature_cols]

def get_prediction(input_df):
    pred = model.predict(input_df)[0]
    return float(pred)

st.title("Bike Demand Prediction")
st.write("Enter weather and time conditions to predict hourly bike rental demand.")

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

days = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday"
]

weather_type = [
    "Clear / Good weather",
    "Misty / Cloudy",
    "Light rain / Snow",
    "Heavy rain / Snow"
]

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        yr = st.selectbox("Year", ["2011", "2012"])
        mnth = st.selectbox("Month", months)
        hr = st.slider("Hour", 0, 23, 8)
        holiday = st.selectbox("Holiday", ["No", "Yes"])
        weekday = st.selectbox("Weekday", days)

    with col2:
        workingday = st.selectbox("Working day", ["No", "Yes"])
        weathersit = st.selectbox("Weather Condition", weather_type)
        temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5, 0.01)
        hum = st.slider("Humidity (0 to 1)", 0.0, 1.0, 0.5, 0.01)
        windspeed = st.slider("Windspeed (0 to 1)", 0.0, 1.0, 0.2, 0.01)

    submitted = st.form_submit_button("Predict demand")

bool_value = {
    "Yes" : 1,
    "No" :0
}

season_to_value = {
    "Spring": 1,
    "Summer": 2,
    "Fall" : 3,
    "Winter": 4
}

year_to_value = {
    "2011": 0,
    "2012": 1
}

month_to_value = {
    "January" : 1,
    "February": 2,
    "March" : 3,
    "April" : 4,
    "May" : 5,
    "June" : 6,
    "July" : 7,
    "August" : 8,
    "September" : 9,
    "October" : 10,
    "November" : 11,
    "December" : 12 
}

day_to_value = {
    "Sunday" : 0,
    "Monday" : 1,
    "Tuesday" : 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}

weather_to_value = {
    "Clear / Good weather" : 1,
    "Misty / Cloudy" : 2,
    "Light rain / Snow" : 3,
    "Heavy rain / Snow" : 4
}


def make_hourly_chart(model, feature_cols, season, yr, mnth, holiday, weekday, workingday, weathersit, temp, hum, windspeed):
    rows = []

    for hr in range(24):
        hr_sin = math.sin(2 * math.pi * hr / 24)
        hr_cos = math.cos(2 * math.pi * hr / 24)

        row = {
            "season": season,
            "yr": yr,
            "mnth": mnth,
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weathersit,
            "temp": temp,
            "hum": hum,
            "windspeed": windspeed,
            "hr_sin": hr_sin,
            "hr_cos": hr_cos
        }

        input_df = pd.DataFrame([row])
        input_df = input_df[feature_cols]
        pred = model.predict(input_df)[0]

        rows.append({
            "hour": hr,
            "predicted_demand": float(pred)
        })

    chart_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chart_df["hour"], chart_df["predicted_demand"], marker="o")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Bike Demand")
    ax.set_title("Predicted Bike Demand Across the Day")
    ax.set_xticks(range(24))

    return fig, chart_df

if submitted:
    season_num = season_to_value[season]
    yr_num = year_to_value[yr]
    mnth_num = month_to_value[mnth]
    holiday_num = bool_value[holiday]
    weekday_num = day_to_value[weekday]
    workingday_num = bool_value[workingday]
    weather_num = weather_to_value[weathersit]

    input_df = build_input_df(
        season=season_num,
        yr=yr_num,
        mnth=mnth_num,
        hr=hr,
        holiday=holiday_num,
        weekday=weekday_num,
        workingday=workingday_num,
        weathersit=weather_num,
        temp=temp,
        hum=hum,
        windspeed=windspeed,
    )

    prediction = get_prediction(input_df)

    st.subheader("Prediction")
    st.metric("Predicted bike rentals", f"{prediction:.0f}")
    st.write(f"Scenario: {season}, {mnth}, {weekday} at {hr}:00 with {weathersit.lower()}.")

    if prediction < 100:
        st.info("Expected demand is relatively low.")
    elif prediction < 300:
        st.success("Expected demand is moderate.")
    else:
        st.success("Expected demand is high.")

    fig, chart_df = make_hourly_chart(
        model=model,
        feature_cols=feature_cols,
        season=season_num,
        yr=yr_num,
        mnth=mnth_num,
        holiday=holiday_num,
        weekday=weekday_num,
        workingday=workingday_num,
        weathersit=weather_num,
        temp=temp,
        hum=hum,
        windspeed=windspeed,
    )

    st.pyplot(fig)

    with st.expander("See model input"):
        st.dataframe(input_df)


st.sidebar.header("About")
st.sidebar.write(
    "This app predicts hourly bike rental demand using a Random Forest model trained on historical bike-sharing data."
)

st.sidebar.write("Built with Streamlit and scikit-learn.")

st.sidebar.markdown("[View GitHub Repo](https://github.com/MichaelMancini9/Bike-Sharing)")

with st.expander("Model details"):
    st.write("Model: Random Forest Regressor")
    st.write("Features: weather, season, working day, cyclical hour encoding")