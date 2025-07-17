import streamlit as st
import os
import re
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
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
                "response_length": len(content)
            }
        except Exception as e:
            st.error(f"Pinecone Assistant error: {str(e)}")
            return None

class TavilyFallbackAgent:
    def __init__(self, tavily_api_key: str):
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)

    def _add_utm_to_links(self, content: str) -> str:
        """
        Finds all Markdown links in a string and appends the UTM parameters.
        """
        def replacer(match):
            url = match.group(1)
            utm_params = "utm_source=12taste.com&utm_medium=fifi-chat"
            if '?' in url:
                new_url = f"{url}&{utm_params}"
            else:
                new_url = f"{url}?{utm_params}"
            return f"({new_url})"
        return re.sub(r'(?<=\])\(([^)]+)\)', replacer, content)

    def _synthesize_search_results(self, results, query: str) -> str:
        """
        Synthesize search results into a coherent response similar to LLM output.
        """
        if isinstance(results, str):
            # If Tavily returns a string summary, use it directly
            return results
        
        if not results or not isinstance(results, list):
            return "I couldn't find any relevant information for your query."
        
        # Extract key information from search results
        relevant_info = []
        sources = []
        
        for i, result in enumerate(results[:3], 1):  # Use top 3 results
            if isinstance(result, dict):
                title = result.get('title', '')
                content = result.get('content', result.get('snippet', ''))
                url = result.get('url', '')
                
                if content:
                    relevant_info.append(content)
                    if url:
                        sources.append(f"[{title}]({url})")
        
        if not relevant_info:
            return "I couldn't find relevant information for your query."
        
        # Create a synthesized response
        response_parts = []
        
        # Combine the information in a natural way
        if len(relevant_info) == 1:
            response_parts.append(f"Based on my search, {relevant_info[0]}")
        else:
            response_parts.append("Based on my search:")
            for info in relevant_info:
                if len(info) > 300:
                    info = info[:300] + "..."
                response_parts.append(f"\n{info}")
        
        # Add sources section
        if sources:
            response_parts.append(f"\n\n**Sources:**")
            for i, source in enumerate(sources, 1):
                response_parts.append(f"{i}. {source}")
        
        return "".join(response_parts)

    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            # Get search results from Tavily
            search_results = self.tavily_tool.invoke({"query": message})
            
            # Synthesize results into a coherent response
            synthesized_content = self._synthesize_search_results(search_results, message)
            
            # Apply UTM tracking to any links
            final_content = self._add_utm_to_links(synthesized_content)
            
            return {
                "content": final_content,
                "success": True,
                "source": "FiFi Web Search"
            }
        except Exception as e:
            return {
                "content": f"I apologize, but an error occurred while searching: {str(e)}",
                "success": False,
                "source": "error"
            }

class ChatApp:
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
        
    def initialize_tools(self, pinecone_api_key: str, assistant_name: str, tavily_api_key: str):
        if PINECONE_AVAILABLE and pinecone_api_key and assistant_name:
            self.pinecone_tool = PineconeAssistantTool(pinecone_api_key, assistant_name)
        if tavily_api_key:
            self.tavily_agent = TavilyFallbackAgent(tavily_api_key)

    def _should_use_web_fallback(self, pinecone_response: Dict[str, Any]) -> bool:
        """Enhanced fallback detection logic."""
        content = pinecone_response.get("content", "").lower()
        
        # Comprehensive list of insufficient information indicators
        insufficient_keywords = [
            # Direct statements of not knowing
            "i don't have specific information", "i don't know", "i'm not sure", 
            "i cannot help", "i cannot provide", "cannot find specific information",
            "no specific information", "no information about", "don't have information",
            
            # Access and availability issues
            "i don't have access", "do not have access", "information is not available",
            "not available in my knowledge", "not contain specific information",
            "unable to find", "no data available", "no relevant information",
            
            # Search and document related
            "search results do not provide", "search results do not contain",
            "not in my database", "no records of", "not documented",
            
            # Uncertainty indicators
            "insufficient information", "limited information", "outside my knowledge",
            "beyond my knowledge", "not familiar with", "cannot answer",
            "additional documents", "please provide more context"
        ]
        
        # Check for insufficient information keywords
        if any(keyword in content for keyword in insufficient_keywords):
            return True
        
        # Check if response lacks citations (might indicate generic response)
        if not pinecone_response.get("has_citations", False):
            if "[1]" not in pinecone_response.get("content", "") and "**Sources:**" not in pinecone_response.get("content", ""):
                return True
        
        # Check for very short responses (likely insufficient)
        if pinecone_response.get("response_length", 0) < 100:
            return True
        
        # Check for generic/uncertain language patterns
        generic_phrases = [
            "based on my knowledge", "generally speaking", "typically",
            "in general", "it's possible that", "might be", "could be",
            "often", "usually", "commonly"
        ]
        if any(phrase in content for phrase in generic_phrases):
            # Only trigger fallback if it's a very generic response (multiple indicators)
            generic_count = sum(1 for phrase in generic_phrases if phrase in content)
            if generic_count >= 2:
                return True
        
        return False

    def get_response(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if self.pinecone_tool:
            with st.spinner("🔍 Querying FiFi (Internal Specialist)..."):
                pinecone_response = self.pinecone_tool.query(chat_history)
                
                if pinecone_response and pinecone_response.get("success"):
                    should_fallback = self._should_use_web_fallback(pinecone_response)
                    
                    if not should_fallback:
                        return pinecone_response
                    else:
                        st.info("🔄 FiFi has limited information on this topic. Switching to web search for a more comprehensive answer.")
        
        if self.tavily_agent:
            with st.spinner("🌐 Searching the web with FiFi Web Search..."):
                last_message = chat_history[-1].content if chat_history else ""
                return self.tavily_agent.query(last_message, chat_history[:-1])
        
        return {
            "content": "I apologize, but all systems are currently unavailable.", 
            "success": False, 
            "source": "error"
        }

def main():
    st.title("🤖 FiFi AI Chat Assistant")
    st.markdown("**Powered by FiFi with Smart Fallback**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("API Keys are loaded from environment variables.", icon="🔐")
        
        # API Keys
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        assistant_name = st.text_input(
            "Pinecone Assistant Name", 
            value=os.getenv("PINECONE_ASSISTANT_NAME", "my-chat-assistant")
        )
        
        # Tool status
        st.subheader("🔧 Tool Status")
        pinecone_status = "✅ Ready" if PINECONE_AVAILABLE and pinecone_api_key and assistant_name else "❌ Not configured"
        tavily_status = "✅ Ready" if tavily_api_key else "❌ Not configured"
        st.write(f"**FiFi Assistant:** {pinecone_status}")
        st.write(f"**FiFi Web Search:** {tavily_status}")
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_app" not in st.session_state:
        st.session_state.chat_app = ChatApp()
        st.session_state.chat_app.initialize_tools(
            pinecone_api_key, assistant_name, tavily_api_key
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if "source" in message:
                st.caption(f"Source: {message['source']}")

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Check if tools are configured
        if not (pinecone_api_key and assistant_name) and not tavily_api_key:
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
            
            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["content"], 
                "source": response["source"]
            })
            st.session_state.chat_history.append(AIMessage(content=response["content"]))

    # Display helpful information for new users
    if not st.session_state.messages:
        st.info("""
        👋 **Welcome to FiFi AI Chat Assistant!**
        
        **How it works:**
        - 🔍 **First**: Searches your internal knowledge base via Pinecone
        - 🌐 **Fallback**: Automatically switches to web search when needed
        - 🎯 **Smart Detection**: Knows when to use which source for the best answers
        - 📰 **Raw Results**: Web search provides direct, unfiltered search results
        
        Just ask any question and FiFi will find the best way to answer it!
        """)

if __name__ == "__main__":
    main()
