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
            instructions = ("You are a helpful AI assistant. Answer questions based on the provided documents. "
                            "When you use information from a document, you MUST generate an inline citation marker, like [1]. "
                            "This is not optional.")
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
        if not self.assistant: return None
        try:
            pinecone_messages = [PineconeMessage(role="user" if isinstance(msg, HumanMessage) else "assistant", content=msg.content) for msg in chat_history]
            response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o")
            content = response.message.content
            if hasattr(response, 'citations') and response.citations:
                citations_header = "\n\n---\n**Sources:**\n"
                citations_list = []
                seen_items = set()
                for citation in response.citations:
                    for reference in citation.references:
                        if hasattr(reference, 'file') and reference.file:
                            display_text = None
                            link_url = None
                            if hasattr(reference.file, 'metadata') and reference.file.metadata:
                                link_url = reference.file.metadata.get('source_url')
                            if not link_url and hasattr(reference.file, 'signed_url') and reference.file.signed_url:
                                link_url = reference.file.signed_url
                            
                            if link_url:
                                display_text = link_url
                                if display_text not in seen_items:
                                    link = f"[{len(seen_items) + 1}] [{display_text}]({link_url})"
                                    citations_list.append(link)
                                    seen_items.add(display_text)
                            else:
                                display_text = getattr(reference.file, 'name', 'Unknown Source')
                                if display_text not in seen_items:
                                    link = f"[{len(seen_items) + 1}] {display_text}"
                                    citations_list.append(link)
                                    seen_items.add(display_text)
                if citations_list:
                    content += citations_header + "\n".join(citations_list)
            return {"content": content, "success": True, "source": "FiFi"}
        except Exception as e:
            st.error(f"Pinecone Assistant error: {str(e)}")
            return None

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
            return {"content": response["output"], "success": True, "source": "FiFi Web Search"}
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
    def _should_use_web_fallback(self, fifi_response_content: str) -> bool:
        content = fifi_response_content.lower()
        insufficient_keywords = [
            "no specific information", "cannot find specific information",
            "i don't have access", "do not have access",
            "information is not available", "not contain specific information",
            "search results do not provide", "search results do not contain",
            "insufficient information", "limited information",
            "additional documents", "please provide more context"
        ]
        if any(keyword in content for keyword in insufficient_keywords):
            return True
        return False
    def get_response(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if self.pinecone_tool:
            with st.spinner("🔍 Querying FiFi (Internal Specialist)..."):
                pinecone_response = self.pinecone_tool.query(chat_history)
                if pinecone_response and pinecone_response.get("success"):
                    content = pinecone_response.get("content", "")
                    if not self._should_use_web_fallback(content):
                        return pinecone_response
                    else:
                        st.info("FiFi has limited information. Switching to web search for a better answer.")
        if self.tavily_agent:
            with st.spinner("🌐 Searching the web with FiFi Web Search..."):
                last_message = chat_history[-1].content if chat_history else ""
                return self.tavily_agent.query(last_message, chat_history[:-1])
        return {"content": "I apologize, but all systems are currently unavailable.", "success": False, "source": "error"}

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
        st.write(f"**FiFi Web Search:** {tavily_status}")
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
