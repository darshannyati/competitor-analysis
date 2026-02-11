import streamlit as st
import json
import os
import re
from typing import TypedDict, List, Dict, Literal
from pypdf import PdfReader

# Frameworks
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langgraph.graph import StateGraph, END, START

# --- 1. CONFIGURATION ---
GROQ_API_KEY = "removed as this project is uploaded on github"  # Set your Groq API key here
TAVILY_API_KEY = "removed as this project is uploaded on github"  # Set your Tavily API key here

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# --- 2. AGENT STATE ---
class AgentState(TypedDict):
    business_context: str
    competitor_data: str
    swot_analysis: str
    final_report: str
    next_action: str
    iterations: int
    logs: List[str]

# --- 3. ROBUST UTILITIES ---
def extract_json(text: str):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"action": "DONE", "reason": "Could not parse JSON, emergency stop."}
    except:
        return {"action": "DONE", "reason": "Error parsing LLM response."}

# --- 4. AGENT NODES ---
ORCHESTRATOR_PROMPT = """
Analyze the Current State and decide the EXACT next action.

CRITICAL: If 'has_report' is True, you MUST choose 'DONE'.

STATE SUMMARY:
- Research Data Exists: {has_research}
- SWOT Analysis Exists: {has_swot}
- Final Report Exists: {has_report}
- Search Attempts: {iterations}/2

DECISION RULES:
1. If no research data -> RESEARCH_AGENT
2. If research exists but no SWOT -> SWOT_AGENT
3. If SWOT exists but no Report -> REPORT_AGENT
4. If Report exists -> DONE

Output valid JSON: {{"action": "RESEARCH_AGENT | SWOT_AGENT | REPORT_AGENT | DONE", "reason": "why"}}
"""

def orchestrator(state: AgentState):
    status = {
        "has_research": len(state.get('competitor_data', '')) > 200,
        "has_swot": bool(state.get('swot_analysis')),
        "has_report": bool(state.get('final_report')),
        "iterations": state.get('iterations', 0)
    }
    
    if status["iterations"] >= 3 and not status["has_report"]:
        return {"next_action": "REPORT_AGENT", "logs": state['logs'] + ["⚠️ Max searches reached, forcing report."]}
    
    res = llm.invoke(ORCHESTRATOR_PROMPT.format(**status))
    decision = extract_json(res.content)
    return {"next_action": decision.get("action", "DONE"), "logs": state['logs'] + [f"🧠 {decision.get('reason')}"]}

def research_agent(state: AgentState):
    opt_res = llm.invoke(f"Write a 15-word search query for competitors of: {state['business_context'][:300]}")
    query = opt_res.content[:380]
    search_results = tavily.search(query=query, search_depth="advanced", max_results=3)
    new_data = "\n".join([f"Source: {r['url']}\nSnippet: {r['content']}" for r in search_results['results']])
    
    return {
        "competitor_data": (state.get('competitor_data', '') + "\n" + new_data).strip(),
        "iterations": state.get('iterations', 0) + 1,
        "logs": state['logs'] + [f"🔍 Search #{state.get('iterations', 0)+1} completed."]
    }

def swot_specialist(state: AgentState):
    prompt = f"Context: {state['business_context'][:500]}\nIntel: {state['competitor_data']}\nCreate SWOT Markdown table."
    res = llm.invoke(prompt)
    return {"swot_analysis": res.content, "logs": state['logs'] + ["📊 SWOT created."]}

def report_node(state: AgentState):
    report = f"# Intelligence Report\n\n{state['swot_analysis']}\n\n## Research\n{state['competitor_data'][:1500]}"
    return {"final_report": report, "logs": state['logs'] + ["📝 Final report compiled."]}

# --- 5. GRAPH CONSTRUCTION ---
builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator)
builder.add_node("research_agent", research_agent)
builder.add_node("swot_specialist", swot_specialist)
builder.add_node("report_node", report_node)
builder.set_entry_point("orchestrator")

def route_next(state):
    action = state["next_action"]
    mapping = {
        "RESEARCH_AGENT": "research_agent",
        "SWOT_AGENT": "swot_specialist",
        "REPORT_AGENT": "report_node",
        "DONE": END
    }
    return mapping.get(action, END)

builder.add_conditional_edges("orchestrator", route_next)
builder.add_edge("research_agent", "orchestrator")
builder.add_edge("swot_specialist", "orchestrator")
builder.add_edge("report_node", "orchestrator")

graph = builder.compile()

# --- 6. CHATBOT FUNCTIONALITY ---
def get_chat_response(user_question: str, context: dict) -> str:
    """Generate a response based on the report and user question."""
    chat_prompt = f"""You are an intelligent assistant helping analyze a market intelligence report.

REPORT CONTEXT:
Business Context: {context.get('business_context', '')[:800]}

SWOT Analysis:
{context.get('swot_analysis', '')}

Competitor Research:
{context.get('competitor_data', '')[:1200]}

USER QUESTION: {user_question}

Provide a helpful, specific answer based on the report data. If the question cannot be answered from the available data, say so and suggest what additional research might help."""

    response = llm.invoke(chat_prompt)
    return response.content

# --- 7. STREAMLIT UI ---
st.set_page_config(page_title="Market Intelligence Agent", page_icon="🌐", layout="wide")

st.title("🌐 Market Intelligence Agent")
st.markdown("Upload a PDF with business context to generate competitive intelligence and chat with the results.")

# Initialize session state
if "report" not in st.session_state:
    st.session_state.report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "final_state" not in st.session_state:
    st.session_state.final_state = None

# Sidebar for file upload and controls
with st.sidebar:
    st.header("📁 Upload & Settings")
    uploaded = st.file_uploader("Upload PDF", type=['pdf'])
    run_btn = st.button("🚀 Start Analysis", use_container_width=True)
    
    if st.session_state.report:
        st.success("✅ Report Generated")
        if st.button("🔄 Reset & Start New", use_container_width=True):
            st.session_state.report = None
            st.session_state.chat_history = []
            st.session_state.final_state = None
            st.rerun()
    
    st.divider()
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Upload a PDF with business info
    2. Click 'Start Analysis'
    3. Review the generated report
    4. Ask questions in the chat
    """)

# Main content area
if uploaded and run_btn and not st.session_state.report:
    reader = PdfReader(uploaded)
    text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
    
    with st.status("🤖 Agent is working...", expanded=True) as status:
        st.write("📄 Extracting business context...")
        st.write("🔍 Researching competitors...")
        st.write("📊 Generating SWOT analysis...")
        st.write("📝 Compiling final report...")
        
        config = {"recursion_limit": 50}
        final_state = graph.invoke({
            "business_context": text,
            "competitor_data": "",
            "swot_analysis": "",
            "final_report": "",
            "iterations": 0,
            "logs": []
        }, config=config)
        
        st.session_state.report = final_state.get("final_report", "No report generated.")
        st.session_state.final_state = final_state
        status.update(label="✅ Analysis Complete!", state="complete")
    
    st.rerun()

# Display report and chat interface
if st.session_state.report:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Intelligence Report")
        with st.container(height=600):
            st.markdown(st.session_state.report)
        
        with st.expander("🔍 View Agent Logs"):
            if st.session_state.final_state:
                for log in st.session_state.final_state.get('logs', []):
                    st.write(log)
    
    with col2:
        st.header("💬 Chat with Your Report")
        
        # Chat container
        chat_container = st.container(height=450)
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**🧑 You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message['content']}")
                st.divider()
        
        # Chat input
        with st.container():
            user_input = st.chat_input("Ask a question about the report...")
            
            if user_input:
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Generate response
                with st.spinner("Thinking..."):
                    response = get_chat_response(
                        user_input,
                        st.session_state.final_state
                    )
                
                # Add assistant message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.rerun()
            
            # Suggested questions
            st.markdown("**💡 Suggested Questions:**")
            suggestions = [
                "What are the key strengths of our business?",
                "Who are the main competitors?",
                "What threats should we be aware of?",
                "What opportunities can we leverage?"
            ]
            
            cols = st.columns(2)
            for idx, suggestion in enumerate(suggestions):
                with cols[idx % 2]:
                    if st.button(suggestion, key=f"suggest_{idx}", use_container_width=True):
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": suggestion
                        })
                        with st.spinner("Thinking..."):
                            response = get_chat_response(
                                suggestion,
                                st.session_state.final_state
                            )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        st.rerun()

else:
    # Welcome screen
    st.info("👈 Upload a PDF in the sidebar to get started!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Research")
        st.markdown("Automatically searches for competitor intelligence")
    with col2:
        st.markdown("### 📊 Analysis")
        st.markdown("Generates comprehensive SWOT analysis")
    with col3:
        st.markdown("### 💬 Chat")
        st.markdown("Interactive Q&A about your report")