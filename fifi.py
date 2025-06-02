import streamlit as st
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException
import traceback # Kept for potential local debugging if needed, but not active in deployed app
import datetime
from fpdf import FPDF

# --- Configuration from Streamlit Secrets ---
try:
    API_KEY = st.secrets["PINECONE_API_KEY"]
    # Using .get for assistant_name and region to provide defaults if not set in secrets
    # This can be useful if these values are fairly static or for easier local testing without secrets.
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifi-co-pilot")
    REGION = st.secrets.get("PINECONE_REGION", "us")
except KeyError as e:
    st.error(f"Missing critical secret: {e}. Please ensure this secret is configured in your Streamlit Cloud app settings or local secrets.toml.")
    st.stop() # Stop the app if essential secrets like API_KEY are missing
except Exception as e: # Catch other potential issues with st.secrets if it's not available
    st.error(f"Error loading secrets: {e}. This app requires secrets to be configured.")
    st.stop()

# --- Theme Configuration ---
st.set_page_config(
    page_title="FiFi Co-Pilot",
    page_icon="",
    layout="wide"
)
# For .streamlit/config.toml theming:
# [theme]
# primaryColor="#f37021"
# backgroundColor="#FFFFFF"
# secondaryBackgroundColor="#F0F2F6"
# textColor="#31333F"
# font="sans serif"

# --- Initialize Pinecone client and assistant (cached for efficiency) ---
@st.cache_resource
def initialize_pinecone_assistant():
    if not API_KEY: # Should have been caught by the secrets loading, but defensive check
        st.error("Pinecone API Key is not available. Cannot initialize assistant.")
        return None
    try:
        pc = Pinecone(api_key=API_KEY)
        # If region is important for Assistant instantiation (depends on SDK version/setup)
        # assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME, region=REGION)
        assistant = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME) # Assuming region is handled by client or not needed here
        return assistant
    except PineconeApiException as e:
        st.error(f"Pinecone API Error during initialization: {e}. Ensure '{ASSISTANT_NAME}' exists and API key is valid.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during assistant initialization: {e}")
        return None

assistant = initialize_pinecone_assistant()

# --- PDF Generation Function ---
def generate_pdf_from_chat(chat_messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.set_fill_color(243, 112, 33) # 1-2-Taste Orange
    pdf.set_text_color(255, 255, 255) # White text
    pdf.cell(0, 10, "FiFi Co-Pilot Chat Transcript", 1, 1, 'C', True)
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0) # Black text
    line_height = 5
    for message in chat_messages:
        role = message["role"].capitalize()
        content = message["content"]
        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, line_height, f"{role}:", 0, 'L', False)
        pdf.set_font("Arial", '', 10)
        try:
            # Attempt to encode to 'latin-1', replacing unsupported characters
            encoded_content = content.encode('latin-1', 'replace').decode('latin-1')
        except Exception:
            # Fallback for content that even 'replace' can't handle well with latin-1
            encoded_content = "[Content with characters not supported by PDF font]"
        pdf.multi_cell(0, line_height, encoded_content, 0, 'L', False)
        pdf.ln(3)
    return pdf.output(dest='S').encode('latin-1')

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Function to handle message sending and processing ---
def handle_user_query(user_query: str):
    if not user_query: # Do nothing if query is empty
        return
    if not assistant:
        st.error("Assistant is not available. Please check configuration or try again later.")
        return

    st.session_state.messages.append({"role": "user", "content": user_query})

    messages_for_sdk_conversion = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
    pinecone_messages_for_sdk = []
    conversion_error_message = None
    try:
        for msg_dict in messages_for_sdk_conversion:
            pinecone_messages_for_sdk.append(Message(role=msg_dict.get("role"), content=msg_dict.get("content")))
    except Exception as e:
        conversion_error_message = f"Error preparing messages for assistant: {e}"

    assistant_reply_content = ""
    if conversion_error_message:
        assistant_reply_content = conversion_error_message
    else:
        try:
            with st.spinner("FiFi is thinking..."):
                response_from_sdk = assistant.chat(messages=pinecone_messages_for_sdk, model="gpt-4o") # Ensure model is correct
            
            if isinstance(response_from_sdk, dict):
                message_data = response_from_sdk.get("message")
                if isinstance(message_data, dict):
                    assistant_reply_content = message_data.get("content", "")
            elif hasattr(response_from_sdk, "message") and hasattr(response_from_sdk.message, "content"):
                assistant_reply_content = response_from_sdk.message.content
            elif hasattr(response_from_sdk, "content"):
                assistant_reply_content = response_from_sdk.content
            
            if not assistant_reply_content: # If empty or not extracted
                 assistant_reply_content = "(FiFi returned an empty or unreadable reply.)"

        except Exception as e:
            assistant_reply_content = f"Sorry, an error occurred while FiFi was thinking: {e}"
            # For server-side logs during development if needed:
            # print(f"Error during assistant.chat(): {e}")
            # traceback.print_exc()
            
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply_content})

# --- Streamlit App UI ---
st.title("1-2-Taste FiFi Co-Pilot")

if not assistant:
    st.warning("FiFi Co-Pilot is currently unavailable. This may be due to configuration issues or service downtime.")
    # Allow sidebar interaction even if assistant is down (e.g. to see previous messages if loaded)
else:
    # --- Sidebar for Quick Questions ---
    st.sidebar.markdown("## Quick Questions")
    preview_questions = [
        "Help me with my recipe for a new juice drink",
        "Suggest me some strawberry flavours for beverage",
        "I need vanilla flavours for ice-cream"
    ]
    for question in preview_questions:
        if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
            handle_user_query(question)
            # Streamlit automatically reruns on button click

    # --- Main Chat Input ---
    user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=(not assistant))
    if user_prompt:
        handle_user_query(user_prompt)
        # Streamlit automatically reruns on chat_input submission

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Sidebar Controls (Downloads, Clear Chat) ---
st.sidebar.markdown("---")
if st.session_state.messages:
    chat_export_data_txt = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        label="📥 Download Chat (TXT)",
        data=chat_export_data_txt,
        file_name=f"fifi_chat_{current_time}.txt",
        mime="text/plain",
        use_container_width=True
    )
    try:
        pdf_data = generate_pdf_from_chat(st.session_state.messages)
        st.sidebar.download_button(
            label="📄 Download Chat (PDF)",
            data=pdf_data,
            file_name=f"fifi_chat_{current_time}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.sidebar.error(f"PDF generation failed: {e}")
        # For server-side logs during development if needed:
        # print(f"PDF generation error: {e}")
        # traceback.print_exc()

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    # Streamlit automatically reruns on button click

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Ask about specific ingredients, applications, or technical details!")
