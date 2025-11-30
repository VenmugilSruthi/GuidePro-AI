import streamlit as st
from datetime import datetime
import re

# Words that indicate the user wants to start a booking
BOOKING_KEYWORDS = ["book", "booking", "reserve", "reservation", "hotel", "trip", "room"]


# -------------------------------
# Helpers
# -------------------------------
def is_valid_date(date_str):
    """Check if string is a valid YYYY-MM-DD date."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_missing_slot(required_slots, data):
    """Return the next slot that has not been filled yet."""
    # required_slots expected as a list of tuples (key, prompt)
    for key, prompt in required_slots:
        if key not in data or data[key] in [None, "", []]:
            return key, prompt
    return None, None


def start_booking_flow(user_input: str):
    """
    Detect user intent to start booking. If detected and no booking in progress,
    initialize required slots and booking state in session_state and return True.
    If booking already in progress, just return True.
    """
    if not user_input:
        return False
    text = user_input.lower()

    # If already in a booking flow, keep it going
    if st.session_state.get("booking_in_progress", False):
        return True

    if any(k in text for k in BOOKING_KEYWORDS):
        # Initialize booking state
        # Define the required slots as a list of (key, human prompt)
        required_slots = [
            ("name", "full name"),
            ("email", "email address"),
            ("phone", "phone number"),
            ("destination", "destination"),
            ("checkin", "check-in date (YYYY-MM-DD)"),
            ("checkout", "check-out date (YYYY-MM-DD)"),
            ("guests", "number of guests"),
        ]

        st.session_state.required_slots = required_slots
        st.session_state.filled_slots = {}
        st.session_state.current_booking_data = {}
        st.session_state.booking_in_progress = True

        # This flag indicates we just started — the next call to handle_booking_turn
        # should *ask* the first slot rather than taking the trigger utterance as the answer.
        st.session_state.booking_just_started = True

        return True

    return False


# -------------------------------
# MAIN STATE MACHINE
# -------------------------------
def handle_booking_turn(user_input: str):
    """
    Manages the booking conversation. Assumes start_booking_flow() was called
    and st.session_state.booking_in_progress is True.
    """

    # Ensure session variables exist (defensive)
    if "required_slots" not in st.session_state:
        st.session_state.required_slots = []
    if "current_booking_data" not in st.session_state:
        st.session_state.current_booking_data = {}
    if "booking_in_progress" not in st.session_state:
        st.session_state.booking_in_progress = False

    data = st.session_state.current_booking_data
    required_slots = st.session_state.required_slots

    # If booking was just started, ask the first question (don't interpret the trigger as an answer)
    if st.session_state.get("booking_just_started", False):
        st.session_state.booking_just_started = False
        first_key, first_prompt = get_missing_slot(required_slots, data)
        if first_key:
            return f"Sure — let's book your hotel. What is your **{first_prompt}**?"
        # If somehow there are no slots, fall through

    # ---------------------------
    # STEP 1 — CONFIRMATION PHASE
    # ---------------------------
    if data.get("_AWAITING_CONFIRMATION", False):

        choice = user_input.lower().strip()

        if choice in ["yes", "y", "ok", "confirm"]:
            # Save booking
            from db import add_booking
            from email_utils import send_confirmation_email

            data["phone"] = data.get("phone", "")
            data["hotel"] = data.get("hotel", data.get("destination"))
            data["notes"] = data.get("notes", "")

            add_booking(data)

            # send email
            try:
                send_confirmation_email(data)
                email_msg = "and a confirmation email has been sent."
            except Exception:
                email_msg = "but the confirmation email could not be sent."

            # Reset flow (clean up session state)
            data.pop("_AWAITING_CONFIRMATION", None)
            st.session_state.booking_in_progress = False
            st.session_state.booking_just_started = False
            st.session_state.required_slots = []
            st.session_state.filled_slots = {}
            st.session_state.current_booking_data = {}

            return f"🎉 **Your booking is confirmed!** The details have been saved {email_msg}"

        elif choice in ["no", "n", "cancel"]:
            # Cancel and cleanup
            st.session_state.current_booking_data = {}
            st.session_state.booking_in_progress = False
            st.session_state.booking_just_started = False
            st.session_state.required_slots = []
            st.session_state.filled_slots = {}
            return "❌ Booking cancelled. How else may I assist you?"

        else:
            return "Please respond with **Yes** or **No** to confirm your booking."

    # ---------------------------
    # STEP 2 — SLOT FILLING PHASE
    # ---------------------------
    missing_key, missing_prompt = get_missing_slot(required_slots, data)

    if missing_key:
        # User input ALWAYS fills the CURRENT missing slot
        slot_to_fill = missing_key
        input_value = user_input.strip()

        # ---------------------------
        # Validation for each slot
        # ---------------------------
        is_valid = True
        error_msg = ""

        if slot_to_fill == "email":
            if not re.match(r"[^@]+@[^@]+\.[^@]+", input_value):
                is_valid = False
                error_msg = "That doesn't look like a valid email address. "

        elif slot_to_fill in ["checkin", "checkout"]:
            if not is_valid_date(input_value):
                is_valid = False
                error_msg = "Please enter a valid date in the format **YYYY-MM-DD**. "

        elif slot_to_fill == "guests":
            try:
                num = int(input_value)
                if num <= 0:
                    is_valid = False
                    error_msg = "Number of guests must be greater than zero. "
                input_value = num
            except ValueError:
                is_valid = False
                error_msg = "Please enter the number of guests as a number. "

        # If invalid → re-ask same question
        if not is_valid:
            return f"{error_msg}What is your **{missing_prompt}**?"

        # If valid → store it
        data[slot_to_fill] = input_value
        st.session_state.current_booking_data = data  # persist

        # Ask next missing slot
        next_key, next_prompt = get_missing_slot(required_slots, data)

        if next_key:
            return f"Got it. And what is your **{next_prompt}**?"

        # If all slots collected → ask for final confirmation
        data["_AWAITING_CONFIRMATION"] = True
        st.session_state.current_booking_data = data

        summary = f"""
### 📄 Booking Summary

- **Name:** {data.get('name')}
- **Email:** {data.get('email')}
- **Phone:** {data.get('phone', 'N/A')}
- **Destination:** {data.get('destination')}
- **Check-in:** {data.get('checkin')}
- **Check-out:** {data.get('checkout')}
- **Guests:** {data.get('guests')}

Does everything look correct?  
Please reply **Yes** or **No**.
"""
        return summary

    # ---------------------------
    # BACKUP — lost flow
    # ---------------------------
    # If this happens, try to reset booking flags defensively
    st.session_state.booking_in_progress = False
    st.session_state.booking_just_started = False
    return "I seem to have lost the booking flow. Please say **Book Hotel** to start again."
