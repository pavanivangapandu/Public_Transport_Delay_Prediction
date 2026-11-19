
import streamlit as st
import pandas as pd
import joblib

from datetime import time

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Public Transport Delay Prediction",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 Public Transport Delay Prediction")
st.write("Predict whether a bus or train will be delayed using a Decision Tree algorithm.")

# --------------------------------
# 2. LOAD TRAINED MODEL FILES
# --------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("decision_tree_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    category_options = joblib.load("category_options.pkl")

    return model, feature_columns, category_options


try:
    model, feature_columns, category_options = load_files()
except Exception as e:
    st.error("Unable to load model files. Check whether all three PKL files are in the project folder.")
    st.stop()


# --------------------------------
# 3. USER INPUT FORM
# --------------------------------

st.subheader("Enter Transport Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        transport_type = st.selectbox(
            "Transport Type",
            category_options["transport_type"]
        )

        route_id = st.selectbox(
            "Route ID",
            category_options["route_id"]
        )

        origin_station = st.selectbox(
            "Origin Station",
            category_options["origin_station"]
        )

        destination_station = st.selectbox(
            "Destination Station",
            category_options["destination_station"]
        )

        weather_condition = st.selectbox(
            "Weather Condition",
            category_options["weather_condition"]
        )

        temperature_C = st.number_input(
            "Temperature (°C)",
            value=25.0
        )

        humidity_percent = st.number_input(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=50
        )

        wind_speed_kmh = st.number_input(
            "Wind Speed (km/h)",
            min_value=0,
            value=10
        )

        precipitation_mm = st.number_input(
            "Precipitation (mm)",
            min_value=0.0,
            value=0.0
        )

    with col2:

        event_type = st.selectbox(
            "Event Type",
            category_options["event_type"]
        )

        season = st.selectbox(
            "Season",
            category_options["season"]
        )

        event_attendance_est = st.number_input(
            "Event Attendance Estimate",
            min_value=0,
            value=0
        )

        traffic_congestion_index = st.number_input(
            "Traffic Congestion Index",
            min_value=0,
            max_value=100,
            value=50
        )

        holiday = st.selectbox(
            "Holiday",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        peak_hour = st.selectbox(
            "Peak Hour",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        weekday = st.selectbox(
            "Weekday (use dataset encoding)",
            list(range(7))
        )

        selected_date = st.date_input("Journey Date")

        current_time = st.time_input(
            "Current Time",
            value=time(8, 0)
        )

        scheduled_departure = st.time_input(
            "Scheduled Departure",
            value=time(8, 30)
        )

        scheduled_arrival = st.time_input(
            "Scheduled Arrival",
            value=time(9, 30)
        )

    submit = st.form_submit_button(
        "Predict Delay",
        use_container_width=True
    )


# --------------------------------
# 4. DATA PREPROCESSING
# --------------------------------

if submit:

    input_data = {
        "date": str(selected_date),
        "time": current_time.strftime("%H:%M:%S"),

        "transport_type": transport_type,
        "route_id": route_id,
        "origin_station": origin_station,
        "destination_station": destination_station,

        "scheduled_departure": scheduled_departure.strftime("%H:%M:%S"),
        "scheduled_arrival": scheduled_arrival.strftime("%H:%M:%S"),

        "weather_condition": weather_condition,

        "temperature_C": temperature_C,
        "humidity_percent": humidity_percent,
        "wind_speed_kmh": wind_speed_kmh,
        "precipitation_mm": precipitation_mm,

        "event_type": event_type,
        "event_attendance_est": event_attendance_est,
        "traffic_congestion_index": traffic_congestion_index,

        "holiday": holiday,
        "peak_hour": peak_hour,
        "weekday": weekday,
        "season": season
    }

    new_data = pd.DataFrame([input_data])

    # Convert date into numerical features
    new_data["date"] = pd.to_datetime(new_data["date"])

    new_data["month"] = new_data["date"].dt.month
    new_data["day"] = new_data["date"].dt.day

    # Convert time columns into hour and minute
    time_columns = [
        "time",
        "scheduled_departure",
        "scheduled_arrival"
    ]

    for col in time_columns:

        new_data[col] = pd.to_datetime(
            new_data[col],
            format="%H:%M:%S",
            errors="coerce"
        )

        new_data[col + "_hour"] = new_data[col].dt.hour
        new_data[col + "_minute"] = new_data[col].dt.minute

    # Remove original columns
    new_data = new_data.drop(
        columns=[
            "date",
            "time",
            "scheduled_departure",
            "scheduled_arrival"
        ]
    )

    # Convert categorical columns into numerical values
    new_data = pd.get_dummies(
        new_data,
        drop_first=True,
        dtype=int
    )

    # Match training feature columns and their exact order
    new_data = new_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # --------------------------------
    # 5. PREDICTION
    # --------------------------------

    prediction = model.predict(new_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Public Transport is Predicted to be DELAYED")

    else:

        st.success("✅ Public Transport is Predicted to be NOT DELAYED")

    st.write("Prediction generated using the Decision Tree Classifier.")

    st.caption(
        "Note: This is an educational machine-learning prediction "
        "based on the provided dataset. It is not a live transport tracking system."
    )




# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #101827, #182538);
    color: #f1f5f9;
}

/* Main Heading */
h1 {
    color: #38d9f5 !important;
    font-size: 36px !important;
    font-weight: 750 !important;
    padding-bottom: 8px;
}

/* Subheadings */
h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 650 !important;
}

/* Normal Text */
p, label {
    color: #dbeafe !important;
}

/* Input Fields */
.stSelectbox div[data-baseweb="select"],
.stNumberInput input,
.stTextInput input,
.stDateInput input,
.stTimeInput input {
    background-color: #263449 !important;
    color: white !important;
    border-radius: 9px !important;
    border: 1px solid #40516a !important;
}

/* Input Focus */
.stNumberInput input:focus,
.stTextInput input:focus {
    border: 1px solid #22d3ee !important;
    box-shadow: 0 0 5px #22d3ee !important;
}

/* Predict Button */
.stButton > button {
    background: linear-gradient(90deg, #0891b2, #06b6d4);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #06b6d4, #0ea5e9);
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.35);
    transform: translateY(-2px);
}

/* Form Container */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #334b63;
    border-radius: 12px;
    padding: 12px;
}

/* Success and Error Messages */
.stAlert {
    border-radius: 10px;
    font-weight: 600;
}

/* Divider */
hr {
    border-color: #334155;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111c2d;
}

/* Footer Text */
.stCaption {
    color: #94a3b8 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------







# -------- DECISION TREE EXPLANATION --------

st.divider()

left_col, right_col = st.columns([1, 1.3], gap="large")

decision_path = model.decision_path(new_data)
tree = model.tree_

node_indicator = decision_path.indices[
    decision_path.indptr[0]:decision_path.indptr[1]
]

# LEFT SIDE - DECISION EXPLANATION
with left_col:

    st.subheader("🌳 How Did the Decision Tree Decide?")

    st.write("The Decision Tree checked these conditions:")

    for node_id in node_indicator:

        if tree.feature[node_id] != -2:

            feature_index = tree.feature[node_id]
            feature_name = feature_columns[feature_index]
            threshold = tree.threshold[node_id]
            value = new_data.iloc[0, feature_index]

            if value <= threshold:
                st.write(
                    f"✅ **{feature_name}**: {value:.2f} "
                    f"≤ {threshold:.2f} → Left branch"
                )
            else:
                st.write(
                    f"➡️ **{feature_name}**: {value:.2f} "
                    f"> {threshold:.2f} → Right branch"
                )

    st.success("🌿 Reached the final leaf node.")

    st.info(
        "The model followed these conditions "
        "to generate the prediction."
    )


# RIGHT SIDE - ACTUAL DECISION TREE DIAGRAM
with right_col:

    st.subheader("🌳 Decision Tree Diagram")

    fig, ax = plt.subplots(figsize=(14, 9))

    plot_tree(
        model,
        feature_names=list(feature_columns),
        class_names=["Not Delayed", "Delayed"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
        ax=ax
    )

    ax.set_title(
        "Public Transport Delay Decision Tree",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    st.caption(
        "Showing the first 3 levels of the trained Decision Tree. "
        "The left panel shows the complete decision path for this prediction."
    )
