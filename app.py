import streamlit as st
import joblib
import re
import csv
import os
from datetime import datetime

tfidf = joblib.load('tfidf_vectorizer.pkl')
category_model = joblib.load('category_model.pkl')
urgency_model = joblib.load('urgency_model.pkl')

LOG_FILE = 'flagged_for_review.csv'

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def log_for_review(ticket_text, result):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'ticket_text', 'predicted_category',
                              'category_confidence', 'predicted_urgency', 'urgency_confidence'])
        writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            ticket_text,
            result['predicted_category'], result['category_confidence'],
            result['predicted_urgency'], result['urgency_confidence']
        ])

def triage_ticket(text, confidence_threshold=0.6):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])

    cat_pred = category_model.predict(vec)[0]
    cat_conf = category_model.predict_proba(vec).max()
    urg_pred = urgency_model.predict(vec)[0]
    urg_conf = urgency_model.predict_proba(vec).max()

    result = {
        "predicted_category": cat_pred, "category_confidence": round(cat_conf, 3),
        "predicted_urgency": urg_pred, "urgency_confidence": round(urg_conf, 3),
        "needs_human_review": (cat_conf < confidence_threshold) or (urg_conf < confidence_threshold)
    }

    if result["needs_human_review"]:
        log_for_review(text, result)   # <-- the actual fix: persist it, don't just display it

    return result

st.title("Support Ticket Triage")
st.write("Paste a customer support ticket below to see its predicted category and urgency.")

ticket_text = st.text_area("Ticket text:", height=120)

if st.button("Classify"):
    if not ticket_text.strip():
        st.error("Please enter some ticket text.")
    else:
        result = triage_ticket(ticket_text)
        col1, col2 = st.columns(2)
        col1.metric("Category", result['predicted_category'], f"{result['category_confidence']:.0%} confidence")
        col2.metric("Urgency", result['predicted_urgency'], f"{result['urgency_confidence']:.0%} confidence")

        if result['needs_human_review']:
            st.warning("⚠️ Low confidence — flagged and logged for human review")
        else:
            st.success("✅ Auto-routed with high confidence")

# sidebar: shows the review queue is real, not just a UI message
if os.path.isfile(LOG_FILE):
    with open(LOG_FILE, encoding='utf-8') as f:
        flagged_count = sum(1 for _ in f) - 1
    st.sidebar.metric("Tickets flagged for review", max(flagged_count, 0))
else:
    st.sidebar.metric("Tickets flagged for review", 0)