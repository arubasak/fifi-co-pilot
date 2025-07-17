import streamlit as st
import os
import re
import json
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

st.set_page_config(page_title="FiFi AI Chat Assistant - Enhanced", page_icon="🤖", layout="wide")

class PineconeAssistantTool:
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE: 
            raise ImportError("Pinecone client not available.")
        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = self._initialize_assistant()

    def _initialize_assistant(self):
        try:
            # Enhanced instructions for better fallback detection
            instructions = (
                "You are a helpful AI assistant. Answer questions based on the provided documents. "
                "When you use information from a document, you MUST generate an inline citation marker, like [1]. "
                "IMPORTANT: If you cannot find specific information in the documents to answer a question, "
                "you MUST respond with 'I don't have specific information about this topic in my knowledge base.' "
                "Do not provide generic or speculative answers for topics not covered in your documents. "
                "Be honest when you don't know something rather than guessing."
            )
            
            assistants_list = self.pc.assistant.list_assistants()
            if self.assistant_name not in [a.name for a in assistants_list]:
                st.warning(f"Assistant '{self.assistant_name}' not found. Creating...")
                return self.pc.assistant.create_assistant(
                    assistant_name=self.assistant_name, 
                    instructions=instructions
                )
            else:
                st.info(f"Connected to assistant: '{self.assistant_name}'")
                return self.pc.assistant.Assistant(assistant_name=self.assistant_name)
        except Exception as e:
            st.error(f"Failed to initialize Pinecone Assistant: {e}")
            return None

    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if not self.assistant: 
            return None
        try:
            pinecone_messages = [
                PineconeMessage(
                    role="user" if isinstance(msg, HumanMessage) else "assistant", 
                    content=msg.content
                ) for msg in chat_history
            ]
            
            response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o")
            content = response.message.content
            has_citations = False
            
            if hasattr(response, 'citations') and response.citations:
                has_citations = True
                citations_header = "\n\n---\n**Sources:**\n"
                citations_list = []
                seen_items = set()
                
                for citation in response.citations:
                    for reference in citation.references:
                        if hasattr(reference, 'file') and reference.file:
                            link_url = None
                            if hasattr(reference.file, 'metadata') and reference.file.metadata:
                                link_url = reference.file.metadata.get('source_url')
                            if not link_url and hasattr(reference.file, 'signed_url') and reference.file.signed_url:
                                link_url = reference.file.signed_url
                            
                            if link_url:
                                if '?' in link_url:
                                    link_url += '&utm_source=fifi-in'
                                else:
                                    link_url += '?utm_source=fifi-in'
                                
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
            
            return {
                "content": content, 
                "success": True, 
                "source": "FiFi",
                "has_citations": has_citations,
                "response_length": len(content),
                "raw_content": content  # For debugging
            }
        except Exception as e:
            st.error(f"Pinecone Assistant error: {str(e)}")
            return None

class TavilyFallbackAgent:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.7)
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)
        today = datetime.now().strftime("%Y-%m-%d")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful AI assistant with web search capabilities. Today's date is {today}."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, [self.tavily_tool], prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=[self.tavily_tool], 
            verbose=True, 
            handle_parsing_errors=True
        )

    def _add_utm_to_links(self, content: str) -> str:
        """Finds all Markdown links in a string and appends UTM parameters."""
        def replacer(match):
            url = match.group(1)
            utm_params = "utm_source=12taste.com&utm_medium=fifi-chat"
            if '?' in url:
                new_url = f"{url}&{utm_params}"
            else:
                new_url = f"{url}?{utm_params}"
            return f"({new_url})"
        return re.sub(r'(?<=\])\(([^)]+)\)', replacer, content)

    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            response = self.agent_executor.invoke({
                "input": message, 
                "chat_history": chat_history
            })
            original_content = response["output"]
            modified_content = self._add_utm_to_links(original_content)
            return {
                "content": modified_content, 
                "success": True, 
                "source": "FiFi Web Search"
            }
        except Exception as e:
            return {
                "content": f"I apologize, but an error occurred: {e}", 
                "success": False, 
                "source": "error"
            }

class ChatApp:
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
        self.debug_mode = False
        
    def set_debug_mode(self, debug: bool):
        """Enable or disable debug mode for detailed fallback analysis."""
        self.debug_mode = debug
        
    def initialize_tools(self, pinecone_api_key: str, assistant_name: str,
                        openai_api_key: str, tavily_api_key: str):
        if PINECONE_AVAILABLE and pinecone_api_key and assistant_name:
            self.pinecone_tool = PineconeAssistantTool(pinecone_api_key, assistant_name)
        if openai_api_key and tavily_api_key:
            self.tavily_agent = TavilyFallbackAgent(openai_api_key, tavily_api_key)

    def _should_use_web_fallback(self, pinecone_response: Dict[str, Any]) -> tuple[bool, str]:
        """Enhanced fallback detection with detailed reasoning."""
        content = pinecone_response.get("content", "").lower()
        reasons = []
        
        # Expanded insufficient information keywords
        insufficient_keywords = [
            # Original keywords
            "no specific information", "cannot find specific information",
            "i don't have access", "do not have access",
            "information is not available", "not contain specific information",
            "search results do not provide", "search results do not contain",
            "insufficient information", "limited information",
            "additional documents", "please provide more context",
            
            # Additional keywords
            "i don't know", "i'm not sure", "i cannot help",
            "outside my knowledge", "not in my database",
            "no information about", "cannot provide information",
            "don't have information", "unable to find",
            "no data available", "no relevant information",
            "i don't have specific information", "not available in my knowledge",
            "cannot answer", "no details about", "not familiar with",
            "beyond my knowledge", "no records of", "not documented"
        ]
        
        # Check for insufficient information keywords
        for keyword in insufficient_keywords:
            if keyword in content:
                reasons.append(f"Found insufficient info keyword: '{keyword}'")
                break
        
        # Check if response lacks citations (might indicate generic response)
        if not pinecone_response.get("has_citations", False):
            if "[1]" not in pinecone_response.get("content", "") and "**Sources:**" not in pinecone_response.get("content", ""):
                reasons.append("No citations found - possibly generic response")
        
        # Check for very short responses
        response_length = pinecone_response.get("response_length", 0)
        if response_length < 100:
            reasons.append(f"Response too short ({response_length} chars)")
        
        # Check for generic phrases that might indicate uncertainty
        generic_phrases = [
            "based on my knowledge", "generally speaking", "typically",
            "in general", "it's possible that", "might be", "could be"
        ]
        for phrase in generic_phrases:
            if phrase in content:
                reasons.append(f"Found generic phrase: '{phrase}'")
                break
        
        should_fallback = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else "Response seems adequate"
        
        return should_fallback, reason_text

    def get_response(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        debug_info = {
            "pinecone_attempted": False,
            "pinecone_response": None,
            "fallback_triggered": False,
            "fallback_reason": "",
            "tavily_used": False
        }
        
        if self.pinecone_tool:
            debug_info["pinecone_attempted"] = True
            with st.spinner("🔍 Querying FiFi (Internal Specialist)..."):
                pinecone_response = self.pinecone_tool.query(chat_history)
                debug_info["pinecone_response"] = pinecone_response
                
                if pinecone_response and pinecone_response.get("success"):
                    should_fallback, fallback_reason = self._should_use_web_fallback(pinecone_response)
                    debug_info["fallback_triggered"] = should_fallback
                    debug_info["fallback_reason"] = fallback_reason
                    
                    if self.debug_mode:
                        self._display_debug_info(pinecone_response, should_fallback, fallback_reason)
                    
                    if not should_fallback:
                        return {**pinecone_response, "debug_info": debug_info}
                    else:
                        st.info(f"🔄 FiFi has limited information ({fallback_reason}). Switching to web search for a better answer.")
        
        if self.tavily_agent:
            debug_info["tavily_used"] = True
            with st.spinner("🌐 Searching the web with FiFi Web Search..."):
                last_message = chat_history[-1].content if chat_history else ""
                tavily_response = self.tavily_agent.query(last_message, chat_history[:-1])
                return {**tavily_response, "debug_info": debug_info}
        
        return {
            "content": "I apologize, but all systems are currently unavailable.", 
            "success": False, 
            "source": "error",
            "debug_info": debug_info
        }
    
    def _display_debug_info(self, pinecone_response: Dict, should_fallback: bool, fallback_reason: str):
        """Display detailed debug information."""
        with st.expander("🐛 Debug Information", expanded=True):
            st.write("**Pinecone Response Analysis:**")
            st.write(f"- Content length: {pinecone_response.get('response_length', 0)} characters")
            st.write(f"- Has citations: {pinecone_response.get('has_citations', False)}")
            st.write(f"- Should fallback: {should_fallback}")
            st.write(f"- Fallback reason: {fallback_reason}")
            
            st.write("**Raw Response Preview:**")
            raw_content = pinecone_response.get('raw_content', '')
            st.code(raw_content[:500] + "..." if len(raw_content) > 500 else raw_content)

def create_test_scenarios():
    """Create test scenarios to verify fallback functionality."""
    return [
        {
            "name": "Generic Knowledge Test",
            "query": "What is the capital of Mars?",
            "expected_fallback": True,
            "description": "Should trigger fallback for unknown information"
        },
        {
            "name": "Current Events Test", 
            "query": "What happened in the news today?",
            "expected_fallback": True,
            "description": "Should trigger fallback for current events"
        },
        {
            "name": "Recent Technology Test",
            "query": "What are the latest AI model releases in 2025?",
            "expected_fallback": True,
            "description": "Should trigger fallback for recent tech developments"
        },
        {
            "name": "Specific Product Test",
            "query": "Tell me about the iPhone 16 Pro Max specifications",
            "expected_fallback": True,
            "description": "Should trigger fallback if not in knowledge base"
        }
    ]

def main():
    st.title("🤖 Enhanced FiFi Chat Assistant")
    st.markdown("**Powered by FiFi with Advanced Fallback Testing**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("API Keys are loaded from environment variables.", icon="🔐")
        
        # API Keys
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        assistant_name = st.text_input(
            "Pinecone Assistant Name", 
            value=os.getenv("PINECONE_ASSISTANT_NAME", "my-chat-assistant")
        )
        
        # Debug mode toggle
        debug_mode = st.checkbox("🐛 Enable Debug Mode", value=False)
        
        # Tool status
        st.subheader("🔧 Tool Status")
        pinecone_status = "✅ Ready" if PINECONE_AVAILABLE and pinecone_api_key and assistant_name else "❌ Not configured"
        tavily_status = "✅ Ready" if openai_api_key and tavily_api_key else "❌ Not configured"
        st.write(f"**FiFi Assistant:** {pinecone_status}")
        st.write(f"**FiFi Web Search:** {tavily_status}")
        
        # Test scenarios
        st.subheader("🧪 Test Fallback")
        test_scenarios = create_test_scenarios()
        
        if st.button("🚀 Run All Tests"):
            st.session_state.run_tests = True
            
        st.write("**Individual Tests:**")
        for i, scenario in enumerate(test_scenarios):
            if st.button(f"Test: {scenario['name']}", key=f"test_{i}"):
                st.session_state.messages.append({"role": "user", "content": scenario['query']})
                st.session_state.chat_history.append(HumanMessage(content=scenario['query']))
                st.session_state.test_mode = scenario
                st.rerun()
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.test_mode = None
            st.rerun()
            
        # Force refresh app (useful if class definition changed)
        if st.button("🔄 Refresh App"):
            if 'chat_app' in st.session_state:
                del st.session_state.chat_app
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_app" not in st.session_state or not hasattr(st.session_state.chat_app, 'set_debug_mode'):
        # Create new ChatApp instance if it doesn't exist or is missing methods
        st.session_state.chat_app = ChatApp()
        st.session_state.chat_app.initialize_tools(
            pinecone_api_key, assistant_name, openai_api_key, tavily_api_key
        )
    if "test_mode" not in st.session_state:
        st.session_state.test_mode = None
    if "run_tests" not in st.session_state:
        st.session_state.run_tests = False

    # Set debug mode safely
    if hasattr(st.session_state.chat_app, 'set_debug_mode'):
        st.session_state.chat_app.set_debug_mode(debug_mode)
    else:
        # Fallback: set debug_mode directly
        st.session_state.chat_app.debug_mode = debug_mode

    # Run all tests if requested
    if st.session_state.run_tests:
        st.header("🧪 Running Fallback Tests")
        test_scenarios = create_test_scenarios()
        
        for scenario in test_scenarios:
            with st.expander(f"Test: {scenario['name']}", expanded=True):
                st.write(f"**Query:** {scenario['query']}")
                st.write(f"**Expected:** {'Fallback to Tavily' if scenario['expected_fallback'] else 'Use Pinecone'}")
                st.write(f"**Description:** {scenario['description']}")
                
                # Create test message
                test_history = [HumanMessage(content=scenario['query'])]
                
                # Get response
                response = st.session_state.chat_app.get_response(test_history)
                
                # Analyze results
                debug_info = response.get('debug_info', {})
                fallback_triggered = debug_info.get('fallback_triggered', False)
                tavily_used = debug_info.get('tavily_used', False)
                
                # Display results
                st.write("**Actual Results:**")
                if tavily_used:
                    st.success("✅ Used Tavily (Web Search)")
                elif not fallback_triggered:
                    st.info("ℹ️ Used Pinecone (No fallback)")
                else:
                    st.warning("⚠️ Fallback triggered but Tavily not available")
                
                if debug_info.get('fallback_reason'):
                    st.write(f"**Fallback Reason:** {debug_info['fallback_reason']}")
                
                # Test result
                test_passed = (scenario['expected_fallback'] and tavily_used) or (not scenario['expected_fallback'] and not tavily_used)
                if test_passed:
                    st.success("✅ Test PASSED")
                else:
                    st.error("❌ Test FAILED")
                
                st.write("**Response:**")
                st.write(response['content'][:200] + "..." if len(response['content']) > 200 else response['content'])
                st.divider()
        
        st.session_state.run_tests = False

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if "source" in message:
                st.caption(f"Source: {message['source']}")

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Check if tools are configured
        if not (pinecone_api_key and assistant_name) and not (openai_api_key and tavily_api_key):
            st.error("Please configure API keys in your environment variables.")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = st.session_state.chat_app.get_response(st.session_state.chat_history)
            
            # Display response
            st.markdown(response["content"], unsafe_allow_html=True)
            st.caption(f"Source: {response['source']}")
            
            # Display test analysis if in test mode
            if st.session_state.test_mode:
                with st.expander("🧪 Test Analysis", expanded=True):
                    scenario = st.session_state.test_mode
                    debug_info = response.get('debug_info', {})
                    
                    st.write(f"**Test:** {scenario['name']}")
                    st.write(f"**Expected Fallback:** {scenario['expected_fallback']}")
                    st.write(f"**Actual Fallback:** {debug_info.get('fallback_triggered', False)}")
                    st.write(f"**Tavily Used:** {debug_info.get('tavily_used', False)}")
                    
                    if debug_info.get('fallback_reason'):
                        st.write(f"**Fallback Reason:** {debug_info['fallback_reason']}")
                    
                    # Test result
                    tavily_used = debug_info.get('tavily_used', False)
                    test_passed = (scenario['expected_fallback'] and tavily_used) or (not scenario['expected_fallback'] and not tavily_used)
                    
                    if test_passed:
                        st.success("✅ Test PASSED")
                    else:
                        st.error("❌ Test FAILED - Fallback logic may need adjustment")
                
                st.session_state.test_mode = None
            
            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["content"], 
                "source": response["source"]
            })
            st.session_state.chat_history.append(AIMessage(content=response["content"]))

    # Display helpful information
    if not st.session_state.messages:
        st.info("""
        👋 **Welcome to the Enhanced FiFi Chat Assistant!**
        
        **New Features:**
        - 🐛 **Debug Mode**: See detailed information about why fallback decisions are made
        - 🧪 **Test Scenarios**: Built-in tests to verify fallback functionality
        - 📊 **Response Analysis**: Understand what triggers the Pinecone → Tavily switch
        
        **How to test the fallback:**
        1. Enable Debug Mode in the sidebar
        2. Use the test scenarios or ask questions your Pinecone assistant doesn't know
        3. Watch the debug information to see the decision process
        
        **Common fallback triggers:**
        - Responses without citations
        - Short responses (< 100 characters)
        - Phrases like "I don't know", "not available", etc.
        - Generic responses without specific information
        """)

if __name__ == "__main__":
    main()
