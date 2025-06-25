import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# Load model and vectorizer
model = joblib.load("fake_job_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

# Fake keyword list (same as training)
fake_keywords = [
    'earn money', 'daily payout', 'easy money', 'registration fee',
    'no experience required', 'instant payout', 'send your bank details',
    'apply now', 'hurry', 'immediate joining', 'guaranteed income',
    'limited slots', 'pay to apply', 'work from home', 'get rich quick',
    'gift cards', 'free travel', 'part-time job', 'flexible hours',
    'no interview required', 'guaranteed job', 'background check not required',
    'win prizes', 'daily cash', 'simple online work', 'secret shopper',
    'survey job', 'no skills needed', 'high payout', 'unlimited earnings',
    'sign up fee', 'deposit required', 'pay to start', 'sms verification',
    'advance fee', 'processing fee', 'urgent hiring', 'sms job',
    'captcha work', 'click ads', 'get paid to click', 'get rich',
    'credit card required', 'training fee', 'starter kit required',
    'easy data entry', 'home typing job', 'daily work payment',
    'work 1 hour per day', 'no resume needed', 'get paid daily',
    'foreign job offer', 'instant approval', 'easy registration',
    'fake visa', 'placement fee', 'HR fee', 'admin fee', 'data entry scam',
    'part time work', 'limited time offer', 'send a copy of ID',
    'training not needed', 'earn upto', 'lifetime income', 'weekly cash',
    'membership fee', 'money back guarantee', 'trusted work from home',
    'online assignment work', 'limited seats', 'you are selected',
    'instant hire', 'click here to apply', 'fill the form', 'submit fee',
    'send us your documents', 'you are hired', '100% earning', 'apply today'
]

# Fake word count
def count_fake_words(text):
    count = 0
    text_lower = text.lower()
    for word in fake_keywords:
        if word in text_lower:
            count += 1
    return count

# Streamlit app
st.set_page_config(page_title="Fake Job Detector", page_icon="🛡️", layout="centered")

st.title(" Fake Job Post Detector")
st.markdown("###  Paste a job description below to check if it's likely a fake or real job posting.")

# Input area
with st.form("job_form"):
    job_input = st.text_area(" Enter Job Description", height=200, placeholder="Paste the job description here...")
    submit = st.form_submit_button("Analyze")

# Prediction logic
if submit and job_input.strip() != "":
    # Vectorize input
    input_tfidf = tfidf.transform([job_input])
    fake_word_count = count_fake_words(job_input)
    fake_word_count_arr = np.array([[fake_word_count]])
    input_final = hstack([input_tfidf, fake_word_count_arr])

    prediction = model.predict(input_final)[0]

    st.markdown("---")
    if prediction == 'Fake':
        st.error(" This looks like a **FAKE JOB POST**. Be cautious!")
    else:
        st.success(" This appears to be a **REAL job post**.")

# Sidebar
with st.sidebar:
    st.header(" About")
    with st.expander("ℹ What this app does"):
        st.write("""
            This app uses a Machine Learning model trained on real and fake job descriptions.
            It detects scammy language and keyword patterns often used in fraudulent job ads.

            **Tech Used:** Python, Scikit-Learn, TF-IDF, RandomForest, Streamlit
        """)
    st.write("Developed by Kanishka | B.Tech IT | IGDTUW")

