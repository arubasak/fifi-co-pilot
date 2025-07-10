import streamlit as st
import os
from typing import List, Dict, Any

# Core dependencies
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_tavily import TavilySearch
from datetime import datetime

# Pinecone Assistant (using correct package name)
try:
    from pinecone import Pinecone
    # Correctly import the Message object for the assistant
    from pinecone_plugins.assistant.models.chat import Message as PineconeMessage
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    # More specific installation instruction
    st.error("Pinecone packages not found. Please install them: pip install pinecone pinecone-plugin-assistant")

# Configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PineconeAssistantTool:
    """Wrapper for Pinecone Assistant API with conversational memory"""

    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available")

        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        # FIX: Implement the robust "get or create" logic
        self.assistant = self._initialize_assistant()

    def _initialize_assistant(self):
        """Initialize assistant by getting an existing one or creating a new one."""
        try:
            assistants_list = self.pc.assistant.list_assistants()
            # FIX: Use attribute access (a.name) as per documentation
            assistant_names = [a.name for a in assistants_list]

            if self.assistant_name not in assistant_names:
                st.warning(f"Assistant '{self.assistant_name}' not found. Creating a new one...")
                # FIX: Programmatically create the assistant if it doesn't exist
                return self.pc.assistant.create_assistant(
                    assistant_name=self.assistant_name,
                    instructions="You are a helpful assistant. Use American English for spelling and grammar.",
                    timeout=30
                )
            else:
                st.info(f"Connected to existing assistant: '{self.assistant_name}'")
                return self.pc.assistant.Assistant(assistant_name=self.assistant_name)
        except Exception as e:
            st.error(f"Failed to initialize Pinecone Assistant: {e}")
            return None

    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """
        Query Pinecone Assistant using the full conversation history.
        FIX: This method now accepts the chat history to provide context.
        """
        if not self.assistant:
            return None

        try:
            # FIX: Convert LangChain messages to Pinecone's expected format
            pinecone_messages = []
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    # FIX: Add the 'role' parameter as required by the API
                    pinecone_messages.append(PineconeMessage(role="user", content=msg.content))
                elif isinstance(msg, AIMessage):
                    pinecone_messages.append(PineconeMessage(role="assistant", content=msg.content))
            
            # FIX: Call the chat method correctly, passing the list of messages
            # The 'conversation_id' parameter is not used; context is managed via the message list.
            response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o")

            # FIX: Parse the response correctly based on documentation (response.message.content)
            content = response.message.content
            return {
                "content": content,
                "success": True,
                "source": "pinecone"
            }
        except Exception as e:
            st.error(f"Pinecone Assistant error: {str(e)}")
            return None

class TavilyFallbackAgent:
    """LangChain agent with Tavily as fallback tool (Unchanged)"""
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
    """Main chat application"""
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
        """Get response with fallback logic"""
        # FIX: Pass the entire chat history to the Pinecone tool
        if self.pinecone_tool:
            with st.spinner("🔍 Querying Pinecone Assistant with history..."):
                pinecone_response = self.pinecone_tool.query(chat_history)
                if pinecone_response and pinecone_response.get("success"):
                    return pinecone_response

        if self.tavily_agent:
            with st.spinner("🌐 Pinecone failed. Searching web for additional information..."):
                # Tavily agent needs the last message and the history
                last_message = chat_history[-1].content if chat_history else ""
                return self.tavily_agent.query(last_message, chat_history[:-1])
        
        return {"content": "Apologies, both systems are unavailable. Check API keys.", "success": False, "source": "error"}

def main():
    st.title("🤖 AI Chat Assistant")
    st.markdown("**Powered by Pinecone Assistant with Tavily Web Search Fallback**")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))
        assistant_name = st.text_input("Pinecone Assistant Name", value=os.getenv("PINECONE_ASSISTANT_NAME", "my-default-assistant"))
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        tavily_api_key = st.text_input("Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY", ""))
        
        pinecone_status = "✅ Ready" if PINECONE_AVAILABLE and pinecone_api_key and assistant_name else "❌ Not configured"
        tavily_status = "✅ Ready" if openai_api_key and tavily_api_key else "❌ Not configured"
        st.write(f"**Pinecone Assistant:** {pinecone_status}")
        st.write(f"**Tavily Fallback:** {tavily_status}")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_app" not in st.session_state:
        st.session_state.chat_app = ChatApp()
        st.session_state.chat_app.initialize_tools(pinecone_api_key, assistant_name, openai_api_key, tavily_api_key)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message:
                st.caption(f"Source: {message['source']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        if not (pinecone_api_key and assistant_name) and not (openai_api_key and tavily_api_key):
            st.error("Please configure at least one tool in the sidebar.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # The get_response method now implicitly uses the history
            response = st.session_state.chat_app.get_response(st.session_state.chat_history)
            
            st.markdown(response["content"])
            st.caption(f"Source: {response['source']}")
            
            st.session_state.messages.append({"role": "assistant", "content": response["content"], "source": response["source"]})
            st.session_state.chat_history.append(AIMessage(content=response["content"]))

if __name__ == "__main__":
    main()
