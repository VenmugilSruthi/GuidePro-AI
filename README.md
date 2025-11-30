*GuidePro AI — AI-Powered Travel & Hotel Booking Assistant*

A fully functional AI Booking Assistant built with Streamlit, OpenAI, RAG, multi-turn slot-filling dialogs, SQLite, and email confirmation, deployed publicly.
Supports: hotel booking, PDF-based retrieval (RAG), admin dashboard, memory, error handling, tool-based routing, and beautiful UI.

1. *Project Overview*

GuidePro AI is an AI-driven travel & hotel booking assistant that allows users to:
✔️ Chat naturally
✔️ Ask travel questions using RAG (PDF-based)
✔️ Book hotels via multi-turn conversational flow
✔️ Confirm all details
✔️ Store bookings in SQLite
✔️ Receive email confirmation
✔️ View & manage bookings in the Admin Dashboard
✔️ Enjoy a beautifully designed chat UI

This project fully satisfies all mandatory requirements in the assignment 

AI Booking Assistant
2. *Features*
RAG (Retrieval-Augmented Generation)
Users upload PDFs
App extracts text, chunks, embeds, and stores vectors
All travel-related questions are answered using RAG + LLM

*Conversational Hotel Booking*
The assistant can collect:
Name
Email
Phone
Destination
Check-in date
Check-out date
Number of guests

Includes:

✔ Multi-turn dialog
✔ Missing field detection
✔ Validation (email/date/phone)
✔ Booking summary
✔ "Yes"/"No" confirmation
✔ Error recovery

*Database Storage (SQLite)*

Tables:
customers
bookings
Automatically stores confirmed bookings.

*Email Confirmation*

After booking:
✔ Sends confirmation email (SMTP / Gmail App Password)
✔ Includes booking ID, dates, hotel, customer name

*Tools Used*:
RAG Tool
Booking Persistence Tool
Email Tool