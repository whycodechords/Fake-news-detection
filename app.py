import streamlit as st
import joblib

# --- Configuration (Includes Setting the Tab Icon) ---
# Use the 'FND' image you generated (or link to it online/use an emoji)
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="/static/fnd_icon.png", # <--- NEW LINE using the static path
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --- Load Model and Vectorizer (Keep this the same) ---
try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
except FileNotFoundError:
    st.error("Error: Model files (vectorizer.jb or lr_model.jb) not found. Please ensure they are in the same directory.")
    st.stop() # Stop the app if files are missing

# --- UI Layout ---

# Center the content and add space
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-bottom: 30px; color: #555555;'>
    Enter a News Article below to check whether it is Fake or Real using our trained model.
    </div>
""", unsafe_allow_html=True)

# Use a container for the input area to make it look professional
with st.container(border=True):
    news_input = st.text_area(
        label="**News Article:**",
        height=200,
        placeholder="Paste your news article text here..."
    )

# Use columns to center the button
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("🚨 ANALYZE ARTICLE 🚀", use_container_width=True):
        if news_input.strip():
            with st.spinner('Analyzing article...'):
                # Step 1: Transform the input text
                transform_input = vectorizer.transform([news_input])
                
                # Step 2: Make a prediction
                prediction = model.predict(transform_input)
                
                # Step 3: Display the result
                if prediction[0] == 1: # Assuming '1' means Real/True
                    st.success("✅ The News is Real! (TRUSTWORTHY)")
                else: # Assuming '0' means Fake/False
                    st.error("❌ The News is Fake! (UNRELIABLE)")
        else:
            st.warning("🧐 Please enter some text to analyze.")

# Add a simple footer/info section
st.markdown("---")
st.info("💡 **How it works:** This is a Machine Learning classification model (likely Logistic Regression) that analyzes text features to predict reliability.")