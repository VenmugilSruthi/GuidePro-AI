import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from utils import render_chat_bubble
from rag import RAGStore
from booking_flow import start_booking_flow, handle_booking_turn
from email_utils import send_confirmation_email
from db import init_db, add_booking, get_bookings, delete_booking, export_bookings_csv
from hotel_data import hotels
from llm_utils import get_llm_client, transcribe_audio, generate_answer

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="GuidePro AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# PREMIUM UI
# ----------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main {
    background: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #c5d5c5;
    border-right: 1px solid #e2e8f0;
    padding-top: 25px;
}
.sidebar-title {
    font-size: 28px;
    font-weight: 800;
    color: #3A7AFE;
    text-align: center;
}
.sidebar-section {
    margin-top: 20px;
    font-weight: 700;
    color: #1F3B7F;
}
.hero-bg {
    width: 100%;
    height: 430px;
    background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
    background-size: cover;
    background-position: center;
    border-radius: 0px;
    position: relative;
    margin: -60px 0 0 0;
}
.hero-card {
    position: absolute;
    top: 62%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 70%;
    background: rgba(255, 255, 255, 0.45);
    backdrop-filter: blur(8px);
    padding: 30px 50px;
    border-radius: 20px;
    text-align: center;
}
.hero-title {
    font-size: 48px;
    font-weight: 900;
    color: #1f2e4b;
}
.hero-sub {
    font-size: 20px;
    color: #2c3e50;
}
.chat-user {
    background: #d6e5ff;
    padding: 12px 18px;
    border-radius: 14px 14px 4px 14px;
    margin-bottom: 12px;
    max-width: 75%;
}
.chat-bot {
    background: white;
    padding: 12px 18px;
    border: 1px solid #ececec;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    border-radius: 14px 14px 14px 4px;
    margin-bottom: 12px;
    max-width: 75%;
}
.fixed-input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 10px;
    background: white;
    border-top: 1px solid #e1e1e1;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# INITIALIZE STATES
# ----------------------------------------------------------
init_db()

if "chat" not in st.session_state:
    st.session_state.chat = [{
        "role": "assistant",
        "content": "Hello! 👋 I’m GuidePro AI. How can I assist your travel today?"
    }]

if "rag" not in st.session_state:
    st.session_state.rag = RAGStore()

if "llm_client" not in st.session_state:
    st.session_state.llm_client = get_llm_client()

if "booking_in_progress" not in st.session_state:
    st.session_state.booking_in_progress = False

if "current_booking_data" not in st.session_state:
    st.session_state.current_booking_data = {}

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>GuidePro AI</div>", unsafe_allow_html=True)

    page = st.radio("Navigate",
                    ["Chat Assistant", "Trip Planner", "Hotels Browser", "Admin", "About"])

    st.markdown("<div class='sidebar-section'>Upload PDFs for RAG</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        st.session_state.rag.add_documents(uploaded)
        st.success("PDF(s) indexed successfully!")

# ----------------------------------------------------------
# LLM HELPER
# ----------------------------------------------------------
def call_llm_system(messages):
    client = st.session_state.llm_client
    if client:
        try:
            return generate_answer(client, messages)
        except:
            pass

    last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
    return st.session_state.rag.query(last_user["content"])

# ----------------------------------------------------------
# CHAT PAGE
# ----------------------------------------------------------
if page == "Chat Assistant":

    st.markdown("""
        <div class="hero-bg">
            <div class="hero-card">
                <h1 class="hero-title">GuidePro AI</h1>
                <p class="hero-sub">Your personal AI for trips, hotels & smart planning.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.chat:
        render_chat_bubble(msg)

    st.markdown("---")

    new_input = st.chat_input("Type your message…")

    if new_input:
        user_msg = new_input
        st.session_state.chat.append({"role": "user", "content": user_msg})

        user_lower = user_msg.lower()

        # ------------------------------------------------------
        # FIXED RAG LOGIC: RUN ONLY IF PDFs EXIST
        # ------------------------------------------------------
        rag_ans = None

        if st.session_state.rag.docs and user_lower.endswith("?"):
            rag_ans = st.session_state.rag.query(user_msg)

        if rag_ans and isinstance(rag_ans, str) and rag_ans.strip():
            st.session_state.chat.append({"role": "assistant", "content": rag_ans})
            st.rerun()

        # Booking Flow
        if start_booking_flow(user_msg) or st.session_state.booking_in_progress:
            reply = handle_booking_turn(user_msg)
            st.session_state.chat.append({"role": "assistant", "content": reply})
            st.rerun()

        # Normal Chat
        reply = call_llm_system(st.session_state.chat)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()

# ----------------------------------------------------------
# TRIP PLANNER
# ----------------------------------------------------------
elif page == "Trip Planner":
    st.header("Trip Planner")
    trip_type = st.selectbox("Trip Type", ["Beach", "City", "Mountain", "International"])
    guests = st.number_input("Guests", 1, 20, 2)
    destination = st.text_input("Destination")

    if st.button("Generate Itinerary"):
        query = f"Create a detailed 3-day {trip_type} trip itinerary for {guests} guests to {destination}."
        reply = call_llm_system([{"role": "user", "content": query}])
        st.write(reply)

# ----------------------------------------------------------
# HOTELS BROWSER
# ----------------------------------------------------------
elif page == "Hotels Browser":
    st.header("Hotels Browser")

    for h in hotels:
        st.markdown("---")
        st.subheader(h["name"])
        st.write(h["location"])
        st.image(h["images"][0], width=250)

# ----------------------------------------------------------
# ADMIN
# ----------------------------------------------------------
elif page == "Admin":
    st.header("Admin Panel")

    data = get_bookings()
    st.table(data)

    if st.button("Export as CSV"):
        export_bookings_csv()
        st.success("Exported bookings.csv!")

    del_id = st.text_input("Delete booking by ID")
    if st.button("Delete Booking"):
        delete_booking(del_id)
        st.success("Booking Deleted")

# ----------------------------------------------------------
# ABOUT
# ----------------------------------------------------------
elif page == "About":
    st.header("About GuidePro AI")
    st.write("Your smart AI travelling assistant.")
