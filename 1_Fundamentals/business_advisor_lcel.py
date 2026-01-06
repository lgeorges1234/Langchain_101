# Nom du fichier: business_advisor_lcel.py
# Description: Multi-step AI Business Advisor using LangChain Expression Language (LCEL).
# This script demonstrates state management, parallel logging, and structured output.

import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 1. ENVIRONMENT SETUP
# Load API keys from .env file for security (Best Practice: Avoid hardcoding keys)
load_dotenv()

# 2. MODEL INITIALIZATION
# We use Gemini 2.5 Flash for high-speed inference and native structured output support.
# temperature=0.0 ensures deterministic (consistent) outputs, crucial for trading logic.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0
)

# 3. OBSERVABILITY & LOGGING UTILITIES
# Audit trail list to store raw LLM responses for debugging/regulatory compliance.
logs = []
parser = StrOutputParser()

# Custom 'Tee' component: Splits the LLM stream into two parallel paths.
# Path 'output': Standardizes the response into a string for the next chain link.
# Path 'log': Side-effect function that appends the raw object to the logs list.
parse_and_log_output_chain = RunnableParallel(
    output=parser,
    log=RunnableLambda(lambda x: logs.append(x))
)

# 4. CHAIN 1: IDEA GENERATION
# Defines the persona and the creative task.
idea_prompt = PromptTemplate(
    template="You are a startup expert and consultant. Give an innovative business idea for the sector: {industry}. Give a name and a short concept."
)
idea_chain = idea_prompt | llm | parse_and_log_output_chain

# 5. CHAIN 2: CRITICAL ANALYSIS
# Takes the output of Chain 1 to perform context-aware reasoning.
analysis_prompt = PromptTemplate(
    template="Analyze this business idea. Give 3 strengths and 3 weaknesses: {idea_text}"
)
analysis_chain = analysis_prompt | llm | parse_and_log_output_chain

# 6. CHAIN 3: DATA STRUCTURED REPORTING
# Uses Pydantic to enforce a strict JSON-like schema for the final output.
class AnalysisReport(BaseModel):
    """Schema for structured business analysis reports."""
    strengths: List[str] = Field(default=[], description="List of the idea's core advantages")
    weaknesses: List[str] = Field(default=[], description="List of the idea's main risks/challenges")

report_prompt = PromptTemplate(
    template=(
        "Based on the following analysis: {analysed_output}, "
        "generate a formal structured report extracting only the key points."
    )
)

# .with_structured_output ensures the return type is an AnalysisReport object, not a string.
report_chain = report_prompt | llm.with_structured_output(AnalysisReport)

# 7. END-TO-END WORKFLOW ORCHESTRATION
# We use functional mappers (lambdas) to rename dict keys between links.
# This 'glue' ensures the output of one step matches the expected input key of the next prompt.
e2e_chain = (
    idea_chain
    | (lambda x: {"idea_text": x["output"]})          # Bridge: dict['output'] -> prompt['idea_text']
    | analysis_chain
    | (lambda x: {"analysed_output" : x["output"]})   # Bridge: dict['output'] -> prompt['analysed_output']
    | report_chain
)

# 8. EXECUTION BLOCK
if __name__ == "__main__":
    print("--- Starting AI Business Advisor Workflow ---")

    # Visualizes the computational graph in the terminal
    e2e_chain.get_graph().print_ascii()
    
    try:
        # Initializing the chain with the root industry parameter
        final_report = e2e_chain.invoke({"industry": "agro"})
        
        print("\n--- FINAL STRUCTURED REPORT ---")
        # final_report is now a Pydantic object with dot-notation access
        print(f"STRENGTHS: {final_report.strengths}")
        print(f"WEAKNESSES: {final_report.weaknesses}")
        
        print(f"\nAudit Trail: {len(logs)} LLM steps captured successfully.")
        
        # Verify raw responses from the audit trail
        for i, message in enumerate(logs):
            print(f"  > Log {i+1} Snippet: {message.content[:60]}...")

    except Exception as e:
        print(f"Workflow execution failed: {str(e)}")


#         1. Memory Improvement (State Management)
# In our current script, the model is "stateless." It forgets the business idea as soon as the script ends. In trading, your agent needs to remember its previous decisions (e.g., "I bought SPY at $500, so I shouldn't buy more right now").

# The Concept: We use BaseChatMemory or LangGraph states.

# Why: To maintain context across multiple turns without re-sending the entire history manually.

# In our project: The Agent will store the "Reasoning" behind a trade so it can learn from its past mistakes (Self-Correction).

# 2. Exploring Runnables (The Lego Bricks)
# You have already used RunnableParallel and RunnableLambda. But there are others that add extreme flexibility:

# RunnablePassthrough: It allows you to pass data to the next step unchanged. It’s like a bypass valve in a hydraulic system.

# RunnableBranch: This allows for routing.

# Example: If the ML signal is "High Volatility" -> Route to Conservative Agent. If "Low Volatility" -> Route to Aggressive Agent.

# RunnableConfig: This allows you to pass specific parameters (like a user ID or a trace ID) throughout the whole chain without modifying the function signatures.

# 3. Adding More Complexity (The Hybrid Logic)
# Complexity doesn't mean "messy code"; it means sophisticated reasoning. To move toward our hybrid trading system, we add complexity by:

# Tools/Functions: Giving the AI a calculator or access to a real-time stock API.

# Validation Loops: If the AnalysisReport shows a high risk, the chain automatically loops back to the "Idea Generation" to find a safer alternative.

# Multi-Model Chaining: Using Gemini 2.5 Pro for the heavy analysis and Gemini 2.5 Flash for the quick formatting to save on costs and latency.