import streamlit as st
from pinecone import Pinecone # This will be used by the plugins
from pinecone_plugins.assistant.models.chat import Message
from pinecone_plugins.assistant.control.core.client.exceptions import PineconeApiException
import traceback # Kept for potential local debugging if needed
import datetime
# from fpdf import FPDF # Removed FPDF import

# --- Configuration from Streamlit Secrets ---
try:
    API_KEY = st.secrets["PINECONE_API_KEY"]
    ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "fifiv1")
    REGION = st.secrets.get("PINECONE_REGION", "us") # This might be used by Pinecone() or Assistant()
except KeyError as e:
    st.error(f"Missing critical secret: {e}. Please ensure this secret is configured in your Streamlit Cloud app settings or local secrets.toml.")
    st.stop()
except Exception as e:
    st.error(f"Error loading secrets: {e}. This app requires secrets to be configured.")
    st.stop()

# --- Theme Configuration ---
st.set_page_config(
    page_title="FiFi Co-Pilot",
    page_icon="", # Using an emoji for the icon
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
    if not API_KEY:
        st.error("Pinecone API Key is not available. Cannot initialize assistant.")
        return None
    try:
        pc = Pinecone(api_key=API_KEY)
        assistant_instance = pc.assistant.Assistant(assistant_name=ASSISTANT_NAME)
        return assistant_instance
    except PineconeApiException as e:
        st.error(f"Pinecone API Error during initialization: {e}. Ensure '{ASSISTANT_NAME}' exists and API key/region are correct.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during assistant initialization: {e}")
        return None

assistant = initialize_pinecone_assistant()

# --- PDF Generation Function (REMOVED) ---
# def generate_pdf_from_chat(chat_messages):
#     ... (function content removed) ...

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Function to handle message sending and processing ---
def handle_user_query(user_query: str):
    if not user_query:
        return
    if not assistant:
        st.error("Assistant is not available. Please check configuration or try again later.")
        return

    st.session_state.messages.append({"role": "user", "content": user_query})

    sdk_messages = []
    conversion_error_message = None
    try:
        for msg_dict in st.session_state.messages:
            sdk_messages.append(Message(role=str(msg_dict.get("role")), content=str(msg_dict.get("content"))))
    except Exception as e:
        conversion_error_message = f"Error preparing messages for assistant: {e}"

    assistant_reply_content = ""
    if conversion_error_message:
        assistant_reply_content = conversion_error_message
    else:
        try:
            with st.spinner("FiFi is thinking..."):
                response_from_sdk = assistant.chat(messages=sdk_messages, model="gpt-4.0")
            
            if isinstance(response_from_sdk, dict):
                message_data = response_from_sdk.get("message")
                if isinstance(message_data, dict):
                    assistant_reply_content = message_data.get("content", "")
                elif hasattr(message_data, 'content'):
                    assistant_reply_content = message_data.content
            elif hasattr(response_from_sdk, 'message') and hasattr(response_from_sdk.message, 'content'):
                 assistant_reply_content = response_from_sdk.message.content
            elif hasattr(response_from_sdk, 'content'):
                assistant_reply_content = response_from_sdk.content
            
            if not assistant_reply_content:
                 assistant_reply_content = "(FiFi returned an empty or unreadable reply.)"
        except PineconeApiException as e:
            assistant_reply_content = f"Sorry, a Pinecone API error occurred: {e}"
        except Exception as e:
            assistant_reply_content = f"Sorry, an unexpected error occurred while FiFi was thinking: {e}"
            
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply_content})

# --- Streamlit App UI ---
st.title("1-2-Taste FiFi Co-Pilot")

if not assistant:
    st.warning("FiFi Co-Pilot is currently unavailable. This may be due to configuration issues or service downtime.")
else:
    st.sidebar.markdown("## Quick Questions")
    preview_questions = [
        "Help me with my recipe for a new juice drink",
        "Suggest me some strawberry flavours for beverage",
        "I need vanilla flavours for ice-cream"
    ]
    for question in preview_questions:
        if st.sidebar.button(question, key=f"preview_{question}", use_container_width=True):
            handle_user_query(question)

    user_prompt = st.chat_input("Ask FiFi Co-Pilot...", key="main_chat_input", disabled=(not assistant))
    if user_prompt:
        handle_user_query(user_prompt)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(str(message.get("content","")))

st.sidebar.markdown("---")
if st.session_state.messages:
    chat_export_data_txt = "\n\n".join([f"{str(msg.get('role','Unknown')).capitalize()}: {str(msg.get('content',''))}" for msg in st.session_state.messages])
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        label="📥 Download Chat (TXT)",
        data=chat_export_data_txt,
        file_name=f"fifi_chat_{current_time}.txt",
        mime="text/plain",
        use_container_width=True
    )
    # --- PDF Download Button and Logic REMOVED ---
    # try:
    #     pdf_data = generate_pdf_from_chat(st.session_state.messages)
    #     st.sidebar.download_button(
    #         label="📄 Download Chat (PDF)",
    #         data=pdf_data, 
    #         file_name=f"fifi_chat_{current_time}.pdf",
    #         mime="application/pdf",
    #         use_container_width=True
    #     )
    # except Exception as e:
    #     st.sidebar.error(f"PDF generation failed: {e}")

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Ask about specific ingredients, applications, or technical details!")
