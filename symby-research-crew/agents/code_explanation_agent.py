"""
Code Explanation Agent for CrewAI
Runs code in isolated containers and explains each step in detail.
"""

from crewai import Agent
from typing import Optional

# Import the code execution tools
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.code_execution_tools import (
    SandboxCodeExecutionTool,
    CodeBreakdownTool,
    StepByStepExecutionTool,
)
from tools.code_tools import (
    CodeStaticAnalysisTool,
    CodeVerificationTool,
)


def create_code_explanation_agent(llm=None) -> Agent:
    """
    Creates the Code Explanation Agent.

    Responsibilities:
    - Execute code in isolated Docker containers
    - Break down code into understandable steps
    - Explain what each part of the code does
    - Trace execution flow and variable changes
    - Identify potential issues or improvements

    Args:
        llm: Optional LLM instance to use

    Returns:
        Agent configured for code explanation
    """
    tools = [
        SandboxCodeExecutionTool(),
        CodeBreakdownTool(),
        StepByStepExecutionTool(),
        CodeStaticAnalysisTool(),
    ]

    return Agent(
        role="Code Explanation Expert",
        goal="""Thoroughly analyze, execute, and explain code step by step.
        Run code in isolated containers and provide detailed explanations of:
        - What each line/block of code does
        - How data flows through the program
        - What variables are created and modified
        - What the output means
        - Any potential issues or improvements""",
        backstory="""You are an expert code educator and analyst with deep experience in
        teaching programming concepts. You excel at taking complex code and breaking it
        down into understandable steps. You can trace through code execution mentally,
        predict outputs, and explain why code behaves the way it does.

        You have access to secure Docker containers where you can safely execute code
        and observe its behavior. You use this to validate your explanations and show
        real execution results. You're skilled at identifying common patterns, potential
        bugs, and areas where code could be improved.

        Your explanations are clear, thorough, and accessible to both beginners and
        experienced programmers. You adjust your level of detail based on the complexity
        of the code and the needs of your audience.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,  # This agent works independently
        memory=True,
    )


def create_code_executor_agent(llm=None) -> Agent:
    """
    Creates a focused Code Executor Agent.

    This agent specializes in running code and reporting results,
    without the full explanation capabilities.

    Args:
        llm: Optional LLM instance to use

    Returns:
        Agent configured for code execution
    """
    tools = [
        SandboxCodeExecutionTool(),
        CodeStaticAnalysisTool(),
    ]

    return Agent(
        role="Code Executor",
        goal="""Execute code in isolated sandbox containers and report the results.
        Focus on:
        - Running code safely in Docker containers
        - Capturing all output (stdout, stderr)
        - Reporting execution status and timing
        - Identifying runtime errors""",
        backstory="""You are a reliable code execution specialist who runs code
        in secure, isolated Docker containers. You ensure that code is executed
        safely, capture all outputs, and report results accurately. You can
        handle various programming languages and understand execution environments.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
    )


def create_code_tutor_agent(llm=None) -> Agent:
    """
    Creates a Code Tutor Agent for educational purposes.

    This agent combines execution with detailed teaching-style explanations.

    Args:
        llm: Optional LLM instance to use

    Returns:
        Agent configured for code tutoring
    """
    tools = [
        SandboxCodeExecutionTool(),
        CodeBreakdownTool(),
        StepByStepExecutionTool(),
        CodeStaticAnalysisTool(),
    ]

    return Agent(
        role="Code Tutor",
        goal="""Teach programming concepts by executing code and explaining each step.
        Your teaching approach:
        - Start with the big picture (what does this code accomplish?)
        - Break down into digestible steps
        - Execute code to show real behavior
        - Explain any unexpected results
        - Suggest exercises or variations for learning
        - Answer follow-up questions""",
        backstory="""You are a patient and knowledgeable programming tutor who loves
        helping people understand code. You believe in learning by doing, so you
        frequently execute code to demonstrate concepts. You adapt your explanations
        to the learner's level - more detailed for beginners, more concise for
        experienced programmers.

        You have years of experience teaching programming and have developed
        effective methods for explaining complex concepts. You use analogies,
        visual descriptions, and step-by-step walkthroughs to make code
        understandable. You're encouraging and positive, celebrating progress
        while gently guiding learners past mistakes.""",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
    )


# Convenience function to get all code-related agents
def get_code_agents(llm=None) -> dict:
    """
    Create and return all code-related agents.

    Args:
        llm: Optional LLM instance to use for all agents

    Returns:
        Dictionary of agent name -> Agent instance
    """
    return {
        "code_explainer": create_code_explanation_agent(llm),
        "code_executor": create_code_executor_agent(llm),
        "code_tutor": create_code_tutor_agent(llm),
    }
