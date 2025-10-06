import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import random
import time

# -------------------- Load model & scaler --------------------
model = pickle.load(open("calories_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------- History CSV --------------------
HISTORY_FILE = "history.csv"
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["Date","Calories Burned","Duration (min)","Exercise Type"]).to_csv(HISTORY_FILE, index=False)

# -------------------- Helper functions --------------------
def save_prediction(prediction, duration, exercise_type="Workout"):
    new_data = pd.DataFrame({
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Calories Burned": [prediction],
        "Duration (min)": [duration],
        "Exercise Type": [exercise_type]
    })
    new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

def load_history():
    df = pd.read_csv(HISTORY_FILE)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by='Date')
    return df

# -------------------- Session state --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users" not in st.session_state:
    st.session_state.users = {"test":"1234"}  # default user

# -------------------- Login page --------------------
def login_page():
    st.markdown("<h2 style='text-align:center;color:#FF4B4B;'>Welcome to Calorie Tracker 🔥</h2>", unsafe_allow_html=True)
    choice = st.radio("Login or Sign Up", ["Login", "Sign Up"], horizontal=True)

    if choice == "Sign Up":
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")
        if st.button("Sign Up"):
            if new_user in st.session_state.users:
                st.error("Username already exists!")
            elif new_user.strip() == "" or new_pass.strip() == "":
                st.warning("Username and password cannot be empty!")
            else:
                st.session_state.users[new_user] = new_pass
                st.success("Account created! Please login.")
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.success("Login successful! Use sidebar to navigate.")
            else:
                st.error("Invalid username or password!")

# -------------------- Health Messages --------------------
def health_message(calories):
    if calories < 50:
        return "Light workout. Remember, consistency is key! 💪"
    elif calories < 200:
        return "Moderate workout. Great job staying active! 🏃‍♂️"
    else:
        return "High intensity workout. Excellent effort! 🔥 Stay hydrated!"

# -------------------- Quotes --------------------
quotes = [
    "“The body achieves what the mind believes.” 🧠💪",
    "“Take care of your body. It's the only place you have to live.” 🏠",
    "“Fitness is not about being better than someone else. It’s about being better than you used to be.” 🌟",
    "“Sweat is just fat crying...”",
    "“Push yourself because no one else is going to do it for you.” 🔥"
]

def display_quotes():
    quote_placeholder = st.empty()
    quote = random.choice(quotes)
    quote_placeholder.markdown(f"<h3 style='text-align:center;color:#FF5733;'>{quote}</h3>", unsafe_allow_html=True)

# -------------------- Main App --------------------
def main_app():
    display_quotes()  # Show a random quote on top
    page = st.sidebar.selectbox("Navigation", ["🏠 Home", "🏋️ Predict Calories", "📊 History & Stats", "🔒 Logout"])

    # -------------------- Logout --------------------
    if page == "🔒 Logout":
        st.session_state.logged_in = False
        st.success("Logged out! Please login again.")
        st.stop()
    

    # -------------------- DASHBOARD / HOME PAGE --------------------
    elif page == "🏠 Home":
        st.markdown("<h1 style='text-align:center;color:#FF4B4B;'>🔥 Fitness Dashboard</h1>", unsafe_allow_html=True)
        st.image(
            "https://imgs.search.brave.com/Pla0ZoF43UoE7TQwmVX6bhkwssF-ojoyTM3HqkSLtxo/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/YmV0dGVybWUud29y/bGQvYXJ0aWNsZXMv/d3AtY29udGVudC91/cGxvYWRzLzIwMjAv/MDMvRnJhbWUtNDU2/OS0xMDI0eDU3Ni5w/bmc",
            use_container_width=True
        )
        st.markdown("---")

        history = load_history()
        if not history.empty:
            st.subheader("📊 Weekly Summary")
            last_week = datetime.now() - timedelta(days=7)
            weekly_data = history[history['Date'] >= last_week]

            col1, col2, col3 = st.columns(3)
            col1.metric("Calories Burned (7 days)", f"{weekly_data['Calories Burned'].sum():.0f} kcal")
            col2.metric("Average Duration", f"{weekly_data['Duration (min)'].mean():.1f} min")
            col3.metric("Workouts Completed", len(weekly_data))

            st.subheader("📈 Calories by Exercise Type (All Time)")
            pie_data = history.groupby("Exercise Type")["Calories Burned"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(5,5))
            ax.pie(pie_data["Calories Burned"], labels=pie_data["Exercise Type"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

            st.subheader("📝 Latest Workouts")
            st.dataframe(weekly_data.sort_values(by="Date", ascending=False).head(5))

        else:
            st.info("No workouts yet! Predict calories to start tracking.")

    # -------------------- PREDICTION PAGE --------------------
    elif page == "🏋️ Predict Calories":
        st.subheader("Enter your details to predict calories burned:")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 10, 100, 25)
            height = st.slider("Height (cm)", 100, 250, 170)
            exercise_type = st.selectbox("Exercise Type", ["Running", "Cycling", "Weight Training", "Yoga", "Other"])
        with col2:
            weight = st.slider("Weight (kg)", 30, 200, 70)
            duration = st.slider("Exercise Duration (minutes)", 1, 300, 30)
            heart_rate = st.slider("Heart Rate (bpm)", 40, 220, 100)
            body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
        goal = st.number_input("Set your calorie goal", min_value=50, max_value=1000, value=500)

        if st.button("Predict Calories Burned 🔥"):
            gender_val = 1 if gender == "Male" else 0
            input_data = np.array([[gender_val, age, height, weight, duration, heart_rate, body_temp]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            st.success(f"🔥 Estimated Calories Burned: {prediction:.2f} kcal")
            
            # Health message
            st.info(health_message(prediction))

            save_prediction(prediction, duration, exercise_type)

            # Pie chart vs goal
            pie_data = pd.DataFrame({
                "Status": ["Calories Burned", "Remaining to Goal"],
                "Value": [prediction, max(0, goal-prediction)]
            })
            fig, ax = plt.subplots(figsize=(5,5))
            ax.pie(pie_data["Value"], labels=pie_data["Status"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
            st.balloons()

    # -------------------- HISTORY PAGE --------------------
    elif page == "📊 History & Stats":
        st.subheader("Your Workout History")
        history = load_history()
        if not history.empty:
            history_chart = history.set_index("Date")
            st.line_chart(history_chart["Calories Burned"])
            st.bar_chart(history_chart["Duration (min)"])
            
            st.markdown("### Summary Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Calories", f"{history['Calories Burned'].sum():.2f} kcal")
            col2.metric("Average Duration", f"{history['Duration (min)'].mean():.1f} min")
            col3.metric("Number of Workouts", len(history))
        else:
            st.info("No history yet! Predict some calories first.")

# -------------------- Run --------------------
if st.session_state.logged_in:
    main_app()
else:
    login_page()
