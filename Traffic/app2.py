import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pymongo

models = {
    "Decision Tree": joblib.load("traffic_model2.pkl"),
    "Logistic Regression": joblib.load("traffic_model3.pkl"),
    "KNN": joblib.load("traffic_model4.pkl"),
    "Gradient Boosting": joblib.load("traffic_model5.pkl"),
    "Random Forest": joblib.load("traffic_model.pkl"),
    "SVC": joblib.load("traffic_model6.pkl"),
}
scaler = joblib.load("scaler.pkl")
scaled_model_names = ["Logistic Regression", "KNN", "SVC"]

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["traffic_database"]
collection = db["traffic_data"]

def get_traffic_data():
    data = {
        'Time': pd.date_range('2023-01-01', periods=100, freq='H'),
        'Traffic Volume': np.random.randint(100, 1000, size=100),
        'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=100),
        'Day Of Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=100),
        'Hour Of Day': np.random.randint(0, 24, size=100),
        'Is Peak Hour': np.random.choice([0, 1], size=100),
        'Speed': np.random.randint(20, 120, size=100)
    }
    return pd.DataFrame(data)

def get_traffic_light_color(traffic_density, speed):
    return 'Green' if speed > 50 else 'Red'

traffic_density_labels = {0: 'Low', 1: 'Medium', 2: 'High'}

st.title('🚦 Traffic Management Dashboard')
st.sidebar.header('Simulation Controls')

# Model selection
model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
selected_model = models[model_name]

# Sidebar inputs
weather = st.sidebar.selectbox('Weather Condition', ['Sunny', 'Rainy', 'Cloudy'])
day_of_week = st.sidebar.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
hour_of_day = st.sidebar.slider('Hour Of Day', 0, 23, 12)
is_peak_hour = st.sidebar.radio('Is Peak Hour', ['No', 'Yes'])
speed = st.sidebar.slider('Speed', 20, 120, 60)
vehicle_count = st.sidebar.slider('Vehicle Count', 1, 50, np.random.randint(1, 50), key='vehicle_count_slider')

# Simulate and store traffic data
if st.sidebar.button('Simulate Traffic'):
    traffic_data = {
        'Weather': weather,
        'Day Of Week': day_of_week,
        'Hour Of Day': hour_of_day,
        'Is Peak Hour': 1 if is_peak_hour == 'Yes' else 0,
        'Speed': speed,
        'Vehicle Count': vehicle_count
    }
    collection.insert_one(traffic_data)
    st.sidebar.success('✅ Added new simulated traffic data!')

# Show traffic data graph
st.header('📊 Real-Time Traffic Data')
df_traffic = get_traffic_data()
st.write(df_traffic.tail(10))

fig, ax = plt.subplots()
ax.plot(df_traffic['Time'], df_traffic['Traffic Volume'], marker='o', linestyle='-')
ax.set_xlabel('Time')
ax.set_ylabel('Traffic Volume')
ax.set_title('Traffic Volume Over Time')
st.pyplot(fig)

# Check Prediction Distribution
if st.button('Check Prediction Distribution'):
    latest_data_subset = pd.DataFrame(list(collection.find().sort("_id", -1).limit(100)))
    latest_data_subset = latest_data_subset[['Hour Of Day', 'Speed', 'Is Peak Hour']]
    latest_data_array = latest_data_subset.to_numpy()

    if model_name in scaled_model_names:
        latest_data_array = scaler.transform(latest_data_array)

    predictions = selected_model.predict(latest_data_array)
    unique, counts = np.unique(predictions, return_counts=True)
    prediction_distribution = {int(u): int(c) for u, c in zip(unique, counts)}

    st.write(f"📈 Prediction Distribution using **{model_name}**:")
    st.write(prediction_distribution)

# Predict Traffic Density & Show Light
if st.button('Predict Traffic Density'):
    latest_data_subset = pd.DataFrame(list(collection.find().sort("_id", -1).limit(10)))
    latest_data_subset = latest_data_subset[['Hour Of Day', 'Speed', 'Is Peak Hour']]
    latest_data_array = latest_data_subset.to_numpy()

    if model_name in scaled_model_names:
        latest_data_array = scaler.transform(latest_data_array)

    predictions = selected_model.predict(latest_data_array)
    predicted_traffic_density = [traffic_density_labels[pred] for pred in predictions]

    avg_traffic_density = int(round(np.mean(predictions)))
    traffic_light_color = get_traffic_light_color(traffic_density_labels[avg_traffic_density], speed)

    st.write(f"🚦 Predicted Traffic Light using **{model_name}**: **{traffic_light_color}**")

    if traffic_light_color == 'Green':
        st.write('⚫🟢')
    else:
        st.write('⚫🔴')
