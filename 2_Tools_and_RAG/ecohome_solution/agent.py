import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from tools import TOOL_KIT

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset


load_dotenv()


class Agent:
    def __init__(self, instructions:str, model:str="gpt-4o-mini"):

        # Initialize the LLM
        llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )

        # Create the Energy Advisor agent
        self.graph = create_react_agent(
            name="energy_advisor",
            prompt=SystemMessage(content=instructions),
            model=llm,
            tools=TOOL_KIT,
        )

    def invoke(self, question: str, context:str=None) -> str:
        """
        Ask the Energy Advisor a question about energy optimization.
        
        Args:
            question (str): The user's question about energy optimization
            location (str): Location for weather and pricing data
        
        Returns:
            str: The advisor's response with recommendations
        """
        
        messages = []
        if context:
            # Add some context to the question as a system message
            messages.append(
                ("system", context)
            )

        messages.append(
            ("user", question)
        )
        
        # Get response from the agent
        response = self.graph.invoke(
            input= {
                "messages": messages
            }
        )
        
        return response

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in TOOL_KIT]


class Judge:
    def __init__(self, model: str = "gpt-4o-mini"):
        # 1. Initialize the LLM for grading
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )
        # 2. Initialize Embeddings for Ragas (The missing piece!)
        self.embeddings = OpenAIEmbeddings(
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("VOCAREUM_API_KEY")
        )
        # Use wrapper for new version of Ragas
        # self.judge_llm = LangchainLLMWrapper(self.llm)
        self.judge_llm = self.llm

    def evaluate_response(self, question, answer, contexts, ground_truth):
        """
        Runs the Ragas evaluation for a single test case.
        """
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts], 
            "ground_truth": [ground_truth]
        })
        
        # We specify the metrics we want to measure
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=self.judge_llm,
            embeddings=self.embeddings
        )
        return result
    
    def evaluate_tool_usage(self, actual_tools: list, expected_tools: list) -> dict:
        """
        Mathematical evaluation of tool usage.
        Calculates precision (appropriateness) and recall (completeness).
        """
        # Remove duplicates just in case
        actual = set(actual_tools)
        expected = set(expected_tools)
        
        # Calculate Intersection (Tools that were both expected and called)
        correct_calls = actual.intersection(expected)
        
        # 1. TOOL_APPROPRIATENESS (Precision)
        # Of the tools I called, how many were actually useful/expected?
        appropriateness = len(correct_calls) / len(actual) if actual else (1.0 if not expected else 0.0)
        
        # 2. TOOL_COMPLETENESS (Recall)
        # Of the tools I should have called, how many did I actually call?
        completeness = len(correct_calls) / len(expected) if expected else 1.0

        return {
            "TOOL_APPROPRIATENESS": appropriateness,
            "TOOL_COMPLETENESS": completeness
        }