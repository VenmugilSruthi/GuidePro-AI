# email_utils.py -- SendGrid API version (works WITHOUT domain verification)
import requests
import streamlit as st

SENDGRID_API_KEY = st.secrets["SENDGRID_API_KEY"]
FROM_EMAIL = st.secrets["FROM_EMAIL"]

SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"


def send_confirmation_email(booking):
    to_email = booking.get("email")

    booking_id = f"NS-{hash(booking.get('email')) % 10000}"

    subject = f"Your Booking Confirmation — ID {booking_id}"

    body = f"""
Hello {booking.get('name')},

Your booking is confirmed with GuidePro AI!

--- Booking Details ---
Booking ID: {booking_id}
Destination: {booking.get('destination')}
Check-in: {booking.get('checkin')}
Check-out: {booking.get('checkout')}
Guests: {booking.get('guests')}
Phone: {booking.get('phone')}
-----------------------

Thank you for choosing GuidePro AI.
"""

    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "personalizations": [{
            "to": [{"email": to_email}],
            "subject": subject
        }],
        "from": {"email": FROM_EMAIL},
        "content": [{
            "type": "text/plain",
            "value": body
        }]
    }

    response = requests.post(SENDGRID_URL, json=data, headers=headers)

    if response.status_code == 202:
        print("✅ SendGrid: Email sent successfully!")
        return True
    else:
        print("❌ SendGrid Error:", response.text)
        return False
