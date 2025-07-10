import streamlit as st
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_tavily import TavilySearch
from datetime import datetime

try:
    from pinecone import Pinecone
    from pinecone_plugins.assistant.models.chat import Message as PineconeMessage
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    st.error("Pinecone packages not found. Please install: pip install pinecone pinecone-plugin-assistant")

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="FiFi AI Chat Assistant", page_icon="🤖", layout="wide")

class PineconeAssistantTool:
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available.")
        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = self._initialize_assistant()

    def _initialize_assistant(self):
        try:
            # A more direct prompt to encourage citing
            instructions = (
                "You are a helpful AI assistant. Answer questions based on the provided documents. "
                "When you use information from a document, you MUST generate an inline citation marker, like [1]. "
                "This is not optional."
            )
            assistants_list = self.pc.assistant.list_assistants()
            if self.assistant_name not in [a.name for a in assistants_list]:
                st.warning(f"Assistant '{self.assistant_name}' not found. Creating...")
                return self.pc.assistant.create_assistant(assistant_name=self.assistant_name, instructions=instructions)
            else:
                st.info(f"Connected to assistant: '{self.assistant_name}'")
                return self.pc.assistant.Assistant(assistant_name=self.assistant_name)
        except Exception as e:
            st.error(f"Failed to initialize Pinecone Assistant: {e}")
            return None

    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """
        Queries Pinecone and correctly parses the `response.citations` field.
        It uses the file's `name` as the default citation and enhances it with `metadata.source_url` if available.
        """
        if not self.assistant:
            return None
        try:
            pinecone_messages = [PineconeMessage(role="user" if isinstance(msg, HumanMessage) else "assistant", content=msg.content) for msg in chat_history]
            response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o")
            content = response.message.content

            # Correctly parse the `response.citations` object as per Pinecone documentation
            if hasattr(response, 'citations') and response.citations:
                citations_header = "\n\n---\n**Sources:**\n"
                citations_list = []
                seen_sources = set()

                # Iterate over each citation group returned by the API
                for citation in response.citations:
                    # Iterate over each reference within that citation
                    for reference in citation.references:
                        if hasattr(reference, 'file') and reference.file:
                            source_display_name = None
                            
                            # Default fallback: Use the file's name, as per your request.
                            if hasattr(reference.file, 'name'):
                                source_display_name = reference.file.name
                            
                            # Enhancement: If metadata with a source_url exists, use it instead.
                            if hasattr(reference.file, 'metadata') and reference.file.metadata and 'source_url' in reference.file.metadata:
                                source_display_name = reference.file.metadata['source_url']
                            
                            # Add the determined source to our list if it's valid and new
                            if source_display_name and source_display_name not in seen_sources:
                                citations_list.append(f"[{len(seen_sources) + 1}] {source_display_name}")
                                seen_sources.add(source_display_name)
                
                if citations_list:
                    content += citations_header + "\n".join(citations_list)

            return {
                "content": content,
                "success": True,
                "source": "FiFi"
            }
        except Exception as e:
            st.error(f"Pinecone Assistant error: {str(e)}")
            return None

# The TavilyFallbackAgent and ChatApp classes remain unchanged.
class TavilyFallbackAgent:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.7)
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful AI assistant with web search. Today's date is {today}."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.llm, [self.tavily_tool], prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=[self.tavily_tool], verbose=True, handle_parsing_errors=True)
    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            response = self.agent_executor.invoke({"input": message, "chat_history": chat_history})
            return {"content": response["output"], "success": True, "source": "tavily_fallback"}
        except Exception as e:
            return {"content": f"I apologize, but an error occurred: {e}", "success": False, "source": "error"}

class ChatApp:
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
    def initialize_tools(self, pinecone_api_key: str, assistant_name: str,
                        openai_api_key: str, tavily_api_key: str):
        if PINECONE_AVAILABLE and pinecone_api_key and assistant_name:
            self.pinecone_tool = PineconeAssistantTool(pinecone_api_key, assistant_name)
        if openai_api_key and tavily_api_key:
            self.tavily_agent = TavilyFallbackAgent(openai_api_key, tavily_api_key)
    def get_response(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if self.pinecone_tool:
            with st.spinner("🔍 Querying FiFi Assistant..."):
                pinecone_response = self.pinecone_tool.query(chat_history)
                if pinecone_response and pinecone_response.get("success"):
                    return pinecone_response
        if self.tavily_agent:
            st.warning("FiFi Assistant failed or is unavailable. Switching to web search fallback.")
            with st.spinner("🌐 Searching web..."):
                last_message = chat_history[-1].content if chat_history else ""
                return self.tavily_agent.query(last_message, chat_history[:-1])
        return {"content": "Systems unavailable. Check API keys.", "success": False, "source": "error"}

def main():
    st.title("🤖 AI Chat Assistant")
    st.markdown("**Powered by FiFi**")
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("API Keys are loaded from environment variables.", icon="🔐")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        assistant_name = st.text_input("Pinecone Assistant Name", value=os.getenv("PINECONE_ASSISTANT_NAME", "my-chat-assistant"))
        st.subheader("🔧 Tool Status")
        pinecone_status = "✅ Ready" if PINECONE_AVAILABLE and pinecone_api_key and assistant_name else "❌ Not configured"
        tavily_status = "✅ Ready" if openai_api_key and tavily_api_key else "❌ Not configured"
        st.write(f"**FiFi Assistant:** {pinecone_status}")
        st.write(f"**Tavily Fallback:** {tavily_status}")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_app" not in st.session_state:
        st.session_state.chat_app = ChatApp()
        st.session_state.chat_app.initialize_tools(pinecone_api_key, assistant_name, openai_api_key, tavily_api_key)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if "source" in message:
                st.caption(f"Source: {message['source']}")
    if prompt := st.chat_input("Ask me anything..."):
        if not (pinecone_api_key and assistant_name) and not (openai_api_key and tavily_api_key):
            st.error("Please configure API keys in your environment variables.")
            return
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.session_state.chat_app.get_response(st.session_state.chat_history)
            st.markdown(response["content"], unsafe_allow_html=True)
            st.caption(f"Source: {response['source']}")
            st.session_state.messages.append({"role": "assistant", "content": response["content"], "source": response["source"]})
            st.session_state.chat_history.append(AIMessage(content=response["content"]))

if __name__ == "__main__":
    main()
