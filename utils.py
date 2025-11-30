import streamlit as st
import markdown

def render_chat_bubble(msg):
    role = msg.get("role", "assistant")
    content = msg.get("content", "")

    # Convert Markdown to HTML
    html_content = markdown.markdown(content)

    # Avatar emoji
    if role == "assistant":
        avatar = "🤖"
        bubble_class = "chat-bot"
    else:
        avatar = "🧑"
        bubble_class = "chat-user"

    st.markdown(f"""
        <div style="display:flex; align-items:flex-start; margin-bottom:12px;">
            <div style="font-size:28px; margin-right:12px;">{avatar}</div>
            <div class="{bubble_class}">
                {html_content}
            </div>
        </div>
    """, unsafe_allow_html=True)
