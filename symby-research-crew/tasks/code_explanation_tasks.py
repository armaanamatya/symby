"""
Code Explanation Tasks for CrewAI
Defines tasks for code execution and step-by-step explanation.
"""

from crewai import Task
from typing import Optional, List


def create_code_explanation_task(
    agent,
    code: str,
    context: Optional[list] = None,
    detail_level: str = "detailed"
) -> Task:
    """
    Create a task for explaining code step by step.

    Args:
        agent: The code explanation agent
        code: The code to explain
        context: Optional list of preceding tasks for context
        detail_level: Level of detail ('brief', 'detailed', 'comprehensive')

    Returns:
        Task for code explanation
    """
    return Task(
        description=f"""Analyze and explain the following code step by step:

```python
{code}
```

Your explanation should include:

1. **Overview**: What does this code do overall?

2. **Step-by-Step Breakdown**:
   - Break the code into logical sections
   - Explain what each section does
   - Describe data flow between sections

3. **Key Concepts**: Explain any important programming concepts used

4. **Execution**: Execute the code in a sandbox and show the actual output

5. **Variable Tracking**: Track how variables change throughout execution

6. **Potential Issues**: Note any bugs, edge cases, or improvements

Detail level requested: {detail_level}
""",
        expected_output="""A comprehensive code explanation including:
- High-level summary of the code's purpose
- Step-by-step breakdown with explanations
- Actual execution output from sandbox
- Variable state changes
- Any issues or suggestions""",
        agent=agent,
        context=context,
    )


def create_code_execution_task(
    agent,
    code: str,
    context: Optional[list] = None,
    timeout_seconds: int = 60
) -> Task:
    """
    Create a task for executing code in a sandbox.

    Args:
        agent: The code executor agent
        code: The code to execute
        context: Optional list of preceding tasks for context
        timeout_seconds: Maximum execution time

    Returns:
        Task for code execution
    """
    return Task(
        description=f"""Execute the following code in an isolated sandbox container:

```python
{code}
```

Requirements:
1. Execute the code safely in a Docker container
2. Capture all output (stdout and stderr)
3. Report execution time and resource usage
4. Note any errors or warnings
5. Maximum timeout: {timeout_seconds} seconds

Provide a complete execution report.""",
        expected_output="""An execution report containing:
- Execution status (success/error/timeout)
- Complete stdout output
- Complete stderr output
- Exit code
- Execution time
- Memory usage
- Any errors with explanations""",
        agent=agent,
        context=context,
    )


def create_step_by_step_execution_task(
    agent,
    code: str,
    context: Optional[list] = None
) -> Task:
    """
    Create a task for step-by-step code execution with explanations.

    Args:
        agent: The code explanation agent
        code: The code to execute step by step
        context: Optional list of preceding tasks for context

    Returns:
        Task for step-by-step execution
    """
    return Task(
        description=f"""Execute the following code step by step, explaining each step as you go:

```python
{code}
```

For each step:
1. Show the code being executed
2. Execute it and show the output
3. Explain what happened
4. Show how variables changed
5. Connect this step to the overall program flow

This should be like watching a debugger walk through the code, but with explanations.""",
        expected_output="""A step-by-step execution log including:
- Each code step with its output
- Explanation of what each step does
- Variable state after each step
- Overall execution summary
- Final program output""",
        agent=agent,
        context=context,
    )


def create_code_tutoring_task(
    agent,
    code: str,
    learner_level: str = "intermediate",
    focus_topics: Optional[List[str]] = None,
    context: Optional[list] = None
) -> Task:
    """
    Create a task for teaching code concepts through execution.

    Args:
        agent: The code tutor agent
        code: The code to teach
        learner_level: Skill level ('beginner', 'intermediate', 'advanced')
        focus_topics: Specific topics to emphasize
        context: Optional list of preceding tasks for context

    Returns:
        Task for code tutoring
    """
    topics_str = ", ".join(focus_topics) if focus_topics else "general programming concepts"

    return Task(
        description=f"""Teach the following code to a {learner_level} programmer:

```python
{code}
```

Focus on explaining: {topics_str}

Your teaching should include:

1. **Learning Objectives**: What will the learner understand after this?

2. **Prerequisites**: What should they already know?

3. **Concept Introduction**: Introduce key concepts before diving into code

4. **Code Walkthrough**:
   - Execute the code to show real behavior
   - Explain each part at the appropriate level
   - Use analogies where helpful

5. **Hands-on Exercises**: Suggest modifications they could try

6. **Common Mistakes**: What errors might they make?

7. **Summary**: Recap the key takeaways

Adjust your explanations for a {learner_level} programmer.""",
        expected_output="""A complete tutorial including:
- Learning objectives
- Prerequisites
- Concept explanations
- Step-by-step code walkthrough with execution
- Practice exercises
- Common pitfalls to avoid
- Summary of key points""",
        agent=agent,
        context=context,
    )


def create_code_debugging_task(
    agent,
    code: str,
    error_description: str = "",
    context: Optional[list] = None
) -> Task:
    """
    Create a task for debugging code with explanations.

    Args:
        agent: The code explanation agent
        code: The code to debug
        error_description: Description of the error or unexpected behavior
        context: Optional list of preceding tasks for context

    Returns:
        Task for code debugging
    """
    error_info = f"\n\nReported issue: {error_description}" if error_description else ""

    return Task(
        description=f"""Debug and explain the following code:{error_info}

```python
{code}
```

Your debugging process should:

1. **Execute the Code**: Run it and observe what happens

2. **Identify Issues**:
   - Syntax errors
   - Runtime errors
   - Logic errors
   - Unexpected behavior

3. **Root Cause Analysis**: Explain WHY the bug occurs

4. **Step Through**: Execute step by step to find where things go wrong

5. **Suggest Fixes**: Provide corrected code with explanations

6. **Prevention Tips**: How to avoid similar bugs in the future""",
        expected_output="""A debugging report including:
- Execution results (errors, output)
- List of identified issues
- Root cause explanation for each issue
- Step-by-step trace showing where bugs occur
- Fixed code with explanations
- Prevention tips""",
        agent=agent,
        context=context,
    )


def create_code_comparison_task(
    agent,
    code_v1: str,
    code_v2: str,
    context: Optional[list] = None
) -> Task:
    """
    Create a task for comparing two code implementations.

    Args:
        agent: The code explanation agent
        code_v1: First code version
        code_v2: Second code version
        context: Optional list of preceding tasks for context

    Returns:
        Task for code comparison
    """
    return Task(
        description=f"""Compare these two code implementations:

**Version 1:**
```python
{code_v1}
```

**Version 2:**
```python
{code_v2}
```

Your comparison should:

1. **Execute Both**: Run both versions and compare outputs

2. **Functionality Comparison**:
   - Do they produce the same results?
   - Are there edge cases where they differ?

3. **Approach Differences**:
   - How does the algorithm/approach differ?
   - What are the trade-offs?

4. **Performance**:
   - Compare execution times
   - Compare memory usage

5. **Code Quality**:
   - Readability
   - Maintainability
   - Best practices

6. **Recommendation**: Which version is better and why?""",
        expected_output="""A comparison report including:
- Execution results for both versions
- Functionality analysis (same/different output)
- Approach comparison
- Performance metrics
- Code quality assessment
- Recommendation with justification""",
        agent=agent,
        context=context,
    )


def create_paper_code_explanation_task(
    agent,
    code: str,
    paper_context: str,
    context: Optional[list] = None
) -> Task:
    """
    Create a task for explaining code from a research paper.

    Args:
        agent: The code explanation agent
        code: The code extracted from a paper
        paper_context: Description of the paper and its methods
        context: Optional list of preceding tasks for context

    Returns:
        Task for paper code explanation
    """
    return Task(
        description=f"""Explain the following code from a research paper:

**Paper Context:**
{paper_context}

**Code:**
```python
{code}
```

Your explanation should:

1. **Connect to Paper**: How does this code implement the paper's methods?

2. **Algorithm Breakdown**:
   - What algorithm does this implement?
   - How does it relate to equations/descriptions in the paper?

3. **Execute and Validate**:
   - Run the code in a sandbox
   - Show what outputs it produces
   - Verify it works as described

4. **Key Components**:
   - Identify the most important functions/classes
   - Explain what each does in the context of the paper

5. **Implementation Details**:
   - What choices did the authors make?
   - Are there any discrepancies with the paper?

6. **Usage Guide**: How would someone use this code?""",
        expected_output="""A research code explanation including:
- Connection to paper's methods
- Algorithm breakdown
- Execution results
- Key component explanations
- Implementation details analysis
- Practical usage guide""",
        agent=agent,
        context=context,
    )
