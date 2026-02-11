# Competitor Analysis -- LLM-Orchestrated Competitive Analysis System

## Project Description

Market Intelligence Agent is a Streamlit-based multi-agent system
powered by Groq's LLaMA 3.3 70B model, Tavily Search API, and LangGraph.

The application ingests a business-context PDF, performs automated
competitor research, generates a structured SWOT analysis, and compiles
a final intelligence report.\
A graph-based orchestration loop dynamically routes execution between
specialized agents (Research, SWOT, Report) using LLM-driven decision
logic.

The system also provides an interactive chat interface that allows users
to query insights directly from the generated intelligence report.

This project demonstrates practical implementation of: - Graph-based
multi-agent orchestration (LangGraph) - LLM-driven conditional routing -
Autonomous web research using Tavily API - Structured SWOT generation -
Context-aware conversational Q&A over generated reports

------------------------------------------------------------------------

## Features

-   **LangGraph Multi-Agent Architecture**
    -   Orchestrator-driven control flow
    -   Conditional routing between:
        -   RESEARCH_AGENT
        -   SWOT_AGENT
        -   REPORT_AGENT
        -   DONE state
-   **PDF Business Context Ingestion**
    -   Extracts text using `pypdf`
    -   Uses document context to guide research and analysis
-   **Autonomous Competitor Research**
    -   Generates optimized search queries via LLM
    -   Performs advanced web search using Tavily API
    -   Aggregates structured competitor intelligence
-   **SWOT Analysis Generation**
    -   Produces structured Markdown SWOT table
    -   Integrates business context and competitor data
-   **Final Intelligence Report Compilation**
    -   Consolidates SWOT + research findings
    -   Generates structured, readable market intelligence output
-   **Context-Aware Chat Interface**
    -   Enables interactive Q&A over generated report
    -   Uses full research + SWOT context for responses
-   **Iteration Control & Safety**
    -   Maximum research iteration handling
    -   Robust JSON parsing for LLM decisions
    -   Fallback termination logic
-   **Execution Logging**
    -   Step-by-step agent reasoning logs
    -   Visible audit trail in UI
