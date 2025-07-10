import streamlit as st
import os
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime
import json

# Core dependencies
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_tavily import TavilySearch
from langchain.schema import BaseMessage

# Pinecone Assistant (assuming you have pinecone-client installed)
try:
    from pinecone import Pinecone
    from pinecone_plugins.assistant.models.chat import Message
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    st.error("Pinecone client not installed. Please install: pip install pinecone-client")

# Configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PineconeAssistantTool:
    """Wrapper for Pinecone Assistant API"""
    
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available")
        
        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = None
        self._initialize_assistant()
    
    def _initialize_assistant(self):
        """Initialize or get existing assistant"""
        try:
            # Try to get existing assistant
            assistants = self.pc.assistant.list_assistants()
            for assistant in assistants:
                if assistant['name'] == self.assistant_name:
                    self.assistant = self.pc.assistant.Assistant(assistant_name=self.assistant_name)
                    return
            
            # Create new assistant if not found
            st.warning(f"Assistant '{self.assistant_name}' not found. Please create it in Pinecone console first.")
            
        except Exception as e:
            st.error(f"Failed to initialize Pinecone Assistant: {str(e)}")
    
    def query(self, message: str, conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Query Pinecone Assistant"""
        if not self.assistant:
            return None
        
        try:
            # Create message
            msg = Message(content=message)
            
            # Send message to assistant
            response = self.assistant.chat(
                messages=[msg],
                conversation_id=conversation_id
            )
            
            if response and response.choices:
                return {
                    "content": response.choices[0].message.content,
                    "conversation_id": response.conversation_id,
                    "success": True,
                    "source": "pinecone"
                }
            
            return None
            
        except Exception as e:
            st.error(f"Pinecone Assistant error: {str(e)}")
            return None

class TavilyFallbackAgent:
    """LangChain agent with Tavily as fallback tool"""
    
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.7
        )
        
        # Initialize Tavily Search Tool
        self.tavily_tool = TavilySearch(
            max_results=5,
            topic="general",
            api_key=tavily_api_key,
            search_depth="basic"
        )
        
        # Create agent
        self._create_agent()
    
    def _create_agent(self):
        """Create LangChain agent with Tavily tool"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful AI assistant with access to web search capabilities.
            Today's date is {today}.
            
            When the primary knowledge source fails or doesn't have sufficient information:
            1. Use the tavily_search tool to find current, relevant information
            2. Provide comprehensive answers based on the search results
            3. Always cite your sources when using web search results
            4. Be transparent about when you're using web search vs. your base knowledge
            
            Guidelines:
            - Use web search for recent events, current data, or when you need more specific information
            - Provide accurate, helpful responses
            - If search results are insufficient, acknowledge limitations
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=[self.tavily_tool],
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=[self.tavily_tool],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def query(self, message: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        """Query using Tavily fallback agent"""
        try:
            if chat_history is None:
                chat_history = []
            
            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": chat_history
            })
            
            return {
                "content": response["output"],
                "success": True,
                "source": "tavily_fallback"
            }
            
        except Exception as e:
            return {
                "content": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "success": False,
                "source": "error"
            }

class ChatApp:
    """Main chat application"""
    
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
        self.conversation_id = None
        
    def initialize_tools(self, pinecone_api_key: str, assistant_name: str, 
                        openai_api_key: str, tavily_api_key: str):
        """Initialize both Pinecone and Tavily tools"""
        try:
            # Initialize Pinecone Assistant
            if PINECONE_AVAILABLE and pinecone_api_key and assistant_name:
                self.pinecone_tool = PineconeAssistantTool(pinecone_api_key, assistant_name)
            
            # Initialize Tavily Fallback Agent
            if openai_api_key and tavily_api_key:
                self.tavily_agent = TavilyFallbackAgent(openai_api_key, tavily_api_key)
                
        except Exception as e:
            st.error(f"Failed to initialize tools: {str(e)}")
    
    def get_response(self, message: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
        """Get response with fallback logic"""
        
        # Try Pinecone Assistant first
        if self.pinecone_tool:
            with st.spinner("🔍 Querying Pinecone Assistant..."):
                pinecone_response = self.pinecone_tool.query(message, self.conversation_id)
                
                if pinecone_response and pinecone_response.get("success"):
                    # Update conversation ID for context
                    self.conversation_id = pinecone_response.get("conversation_id")
                    return pinecone_response
        
        # Fallback to Tavily if Pinecone fails or unavailable
        if self.tavily_agent:
            with st.spinner("🌐 Searching web for additional information..."):
                return self.tavily_agent.query(message, chat_history)
        
        # Final fallback
        return {
            "content": "I apologize, but both primary and backup systems are currently unavailable. Please check your API keys and try again.",
            "success": False,
            "source": "error"
        }

def main():
    st.title("🤖 AI Chat Assistant")
    st.markdown("**Powered by Pinecone Assistant with Tavily Web Search Fallback**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("API Keys")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", 
                                       value=os.getenv("PINECONE_API_KEY", ""))
        assistant_name = st.text_input("Pinecone Assistant Name", 
                                     value=os.getenv("PINECONE_ASSISTANT_NAME", ""))
        openai_api_key = st.text_input("OpenAI API Key", type="password",
                                     value=os.getenv("OPENAI_API_KEY", ""))
        tavily_api_key = st.text_input("Tavily API Key", type="password",
                                     value=os.getenv("TAVILY_API_KEY", ""))
        
        # Tool Status
        st.subheader("🔧 Tool Status")
        pinecone_status = "✅ Ready" if (PINECONE_AVAILABLE and pinecone_api_key and assistant_name) else "❌ Not configured"
        tavily_status = "✅ Ready" if (openai_api_key and tavily_api_key) else "❌ Not configured"
        
        st.write(f"**Pinecone Assistant:** {pinecone_status}")
        st.write(f"**Tavily Fallback:** {tavily_status}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        # Instructions
        st.subheader("📋 How it works")
        st.markdown("""
        1. **Primary**: Queries Pinecone Assistant first
        2. **Fallback**: If Pinecone fails, uses Tavily web search
        3. **Smart Routing**: Automatically switches between tools
        4. **Source Attribution**: Shows which tool provided the answer
        """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_app" not in st.session_state:
        st.session_state.chat_app = ChatApp()
    
    # Initialize tools
    if pinecone_api_key or (openai_api_key and tavily_api_key):
        st.session_state.chat_app.initialize_tools(
            pinecone_api_key, assistant_name, openai_api_key, tavily_api_key
        )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message:
                source_emoji = {
                    "pinecone": "🔍",
                    "tavily_fallback": "🌐",
                    "error": "⚠️"
                }
                st.caption(f"{source_emoji.get(message['source'], '🤖')} Source: {message['source']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Check if at least one tool is configured
        if not (pinecone_api_key and assistant_name) and not (openai_api_key and tavily_api_key):
            st.error("Please configure at least one set of API keys in the sidebar.")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            response = st.session_state.chat_app.get_response(
                prompt, st.session_state.chat_history
            )
            
            st.markdown(response["content"])
            
            # Show source
            source_emoji = {
                "pinecone": "🔍",
                "tavily_fallback": "🌐", 
                "error": "⚠️"
            }
            st.caption(f"{source_emoji.get(response['source'], '🤖')} Source: {response['source']}")
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["content"],
                "source": response["source"]
            })
            st.session_state.chat_history.append(AIMessage(content=response["content"]))

if __name__ == "__main__":
    main()
