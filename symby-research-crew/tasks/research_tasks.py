"""
Research Tasks for CrewAI
Defines the workflow tasks for autonomous paper research.
"""

from crewai import Task
from typing import Optional


def create_paper_analysis_task(agent, paper_path: str, context: Optional[list] = None) -> Task:
    """
    Create task for analyzing an uploaded paper.
    """
    return Task(
        description=f"""Perform comprehensive analysis of the research paper at: {paper_path}

        === EXTRACTION REQUIREMENTS ===

        1. **METADATA** (exact values from paper):
           - Full title
           - All authors (first author highlighted)
           - Publication date/year
           - Venue (conference/journal name)
           - DOI or arXiv ID if available

        2. **RESEARCH CONTEXT**:
           - Problem Statement: What specific problem does this paper address?
           - Motivation: Why is this problem important?
           - Research Gap: What limitation in prior work does this address?

        3. **METHODOLOGY**:
           - Primary method/approach name
           - Key algorithmic innovations
           - Architecture details (for ML papers: model type, layers, attention mechanisms)
           - Training procedure (optimizer, learning rate, epochs, batch size)
           - Computational requirements mentioned

        4. **EVALUATION**:
           - Datasets used (with versions if specified)
           - Evaluation metrics reported
           - Baseline methods compared against
           - Key quantitative results (exact numbers)
           - Ablation studies performed

        5. **REPRODUCIBILITY INDICATORS**:
           - Code availability (repository URL if mentioned)
           - Data availability
           - Hyperparameters provided
           - Implementation details level

        6. **SEARCH KEYWORDS**:
           - Technical terms for finding implementations
           - Author names for GitHub search
           - Method names that might appear in repo names

        === ANALYSIS STANDARDS ===
        - Extract EXACT values, don't paraphrase metrics or results
        - Note what information is MISSING if not provided
        - Be critical about reproducibility - most papers have gaps
        - Prioritize information useful for finding/verifying code""",
        expected_output="""Structured analysis report with sections:

        ## Metadata
        - Title, Authors, Date, Venue, Identifiers

        ## Problem & Motivation
        - Clear problem statement
        - Research gap addressed

        ## Methodology
        - Approach name and description
        - Technical architecture details
        - Training specifications

        ## Results
        - Datasets and metrics table
        - Key quantitative findings
        - Comparison to baselines

        ## Reproducibility Assessment
        - Code/data availability
        - Missing information for replication

        ## Search Keywords
        - Terms for code search""",
        agent=agent,
        context=context,
    )


def create_literature_search_task(agent, paper_topic: str, keywords: list, context: Optional[list] = None) -> Task:
    """
    Create task for researching related literature.
    """
    keywords_str = ", ".join(keywords)

    return Task(
        description=f"""Conduct systematic literature research for topic: {paper_topic}

        Search Keywords: {keywords_str}

        === SEARCH STRATEGY ===

        1. **ARXIV SEARCH** (Primary Source):
           - Search using provided keywords + variations
           - Filter by relevance and recency (prioritize last 3 years)
           - Note arXiv IDs for all relevant papers
           - Check cs.LG, cs.CL, cs.CV, stat.ML categories as appropriate

        2. **FOUNDATIONAL WORK**:
           - Identify seminal papers this work builds upon
           - Find the original papers for key methods/architectures used
           - Note highly-cited foundational references

        3. **COMPETING APPROACHES**:
           - Find papers solving the same problem differently
           - Identify state-of-the-art alternatives
           - Note papers that benchmark against similar baselines

        4. **METHODOLOGY CONTEXT**:
           - Papers using similar techniques in other domains
           - Survey papers covering this research area
           - Tutorial papers explaining core concepts

        === PRIORITIZATION CRITERIA ===
        - Papers with code available (check Papers With Code)
        - Recent papers (last 2 years) for current SOTA
        - Highly-cited papers for foundational understanding
        - Papers from top venues (NeurIPS, ICML, ICLR, ACL, CVPR, etc.)

        === CONCEPT EXPLANATION ===
        Use Wikipedia/reliable sources to explain:
        - Core technical concepts a practitioner needs to understand
        - Mathematical foundations if applicable
        - Key terminology definitions""",
        expected_output="""Comprehensive literature review:

        ## Core Related Papers (5-10)
        For each paper:
        | arXiv ID | Title | Year | Relation | Code Available? |

        ## Foundational Papers (3-5)
        - Seminal works this research builds upon

        ## Competing Methods
        - Alternative approaches to the same problem
        - Comparative strengths/weaknesses

        ## Background Concepts
        - Key technical concepts explained
        - Important terminology

        ## Research Timeline
        - Evolution of approaches in this area

        ## Code Availability Summary
        - Papers most likely to have usable implementations""",
        agent=agent,
        context=context,
    )


def create_code_search_task(agent, paper_title: str, authors: list, keywords: list, context: Optional[list] = None) -> Task:
    """
    Create task for finding code implementations.
    """
    authors_str = ", ".join(authors[:3]) if authors else "Unknown"
    keywords_str = ", ".join(keywords)

    return Task(
        description=f"""Execute comprehensive code search for paper: "{paper_title}"

        Authors: {authors_str}
        Search Keywords: {keywords_str}

        === SEARCH PROTOCOL ===

        1. **PAPERS WITH CODE** (First Priority):
           - Search exact paper title
           - Check for linked implementations
           - Note benchmark rankings if available

        2. **GITHUB SEARCH** (Multiple Strategies):
           Strategy A - Direct Title Search:
           - Search exact paper title in quotes
           - Search abbreviated/acronym versions

           Strategy B - Author Search:
           - Search "[author_lastname] [method_name]"
           - Check author GitHub profiles directly
           - Look for lab/organization repositories

           Strategy C - Method/Architecture Search:
           - Search method names from paper
           - Search architecture names + "implementation"
           - Search dataset name + method

        3. **ALTERNATIVE SOURCES**:
           - HuggingFace Hub (for ML models)
           - GitLab, Bitbucket
           - Official project websites
           - Supplementary materials links in paper

        === QUALITY ASSESSMENT CRITERIA ===

        For EACH repository found, evaluate:

        | Criterion | Weight | Check |
        |-----------|--------|-------|
        | Official | High | Author affiliation match? |
        | Stars | Medium | >100 good, >1000 excellent |
        | Recency | High | Updated within 1 year? |
        | Documentation | High | README with setup instructions? |
        | Dependencies | Medium | requirements.txt/environment.yml? |
        | Pretrained Models | High | Checkpoints available? |
        | Tests | Medium | Test files present? |
        | Issues/Activity | Medium | Responsive maintainers? |

        === CLASSIFICATION ===

        - **Official**: Repository by paper authors or affiliated lab
        - **Verified Reproduction**: Community impl with reported matching results
        - **Community**: Third-party implementation, results not verified
        - **Partial**: Incomplete implementation or subset of paper

        === RED FLAGS TO NOTE ===
        - No license file
        - No documentation
        - Stale (>2 years no updates)
        - Reported issues about incorrect results
        - Missing core components""",
        expected_output="""Code Implementation Catalog:

        ## Official Implementations
        | Repo URL | Stars | Framework | Last Updated | Pretrained? |

        ## Verified Community Implementations
        | Repo URL | Stars | Framework | Verification Status |

        ## Other Implementations
        | Repo URL | Stars | Framework | Notes |

        ## Quality Rankings
        1. [Top recommendation with justification]
        2. [Second choice]
        3. [Alternative if specific needs]

        ## Implementation Gaps
        - Components not found in any implementation
        - Missing pretrained models
        - Framework gaps (e.g., no PyTorch version)

        ## Quick Start Recommendation
        - Best repo for getting started
        - Required setup steps""",
        agent=agent,
        context=context,
    )


def create_code_verification_task(agent, repo_url: str, paper_claims: str, context: Optional[list] = None) -> Task:
    """
    Create task for verifying code implementation.
    """
    return Task(
        description=f"""Perform rigorous verification of code implementation against paper claims.

        Repository: {repo_url}

        === PAPER'S KEY CLAIMS ===
        {paper_claims}

        === VERIFICATION PROTOCOL ===

        **PHASE 1: Repository Structure Analysis**
        - Map directory structure
        - Identify main implementation files
        - Locate configuration files
        - Find training/evaluation scripts
        - Check for pretrained model files or download scripts

        **PHASE 2: Documentation Review**
        - README completeness (setup, usage, results)
        - Requirements/dependencies specified
        - Hardware requirements documented
        - Expected results stated

        **PHASE 3: Architecture Verification**
        For each key component in paper:
        | Paper Description | Code Location | Match Status | Discrepancies |

        Check specifically:
        - Model architecture (layers, dimensions, attention heads)
        - Loss functions
        - Optimization settings
        - Data preprocessing pipeline
        - Augmentation strategies
        - Evaluation metrics implementation

        **PHASE 4: Completeness Assessment**

        | Component | Status | Notes |
        |-----------|--------|-------|
        | Data loading | ✓/✗/Partial | |
        | Model definition | ✓/✗/Partial | |
        | Training loop | ✓/✗/Partial | |
        | Evaluation script | ✓/✗/Partial | |
        | Pretrained weights | ✓/✗/Partial | |
        | Inference pipeline | ✓/✗/Partial | |
        | Results reproduction | ✓/✗/Partial | |

        **PHASE 5: Reproducibility Assessment**

        Score each factor (0-10):
        - Environment setup clarity: Can someone set up the environment?
        - Data preparation: Is data loading/preprocessing clear?
        - Training reproducibility: Random seeds, deterministic settings?
        - Result verification: Can claimed results be reproduced?
        - Documentation quality: Is usage clear?

        **PHASE 6: Red Flag Detection**

        Check for:
        - Hardcoded paths or credentials
        - Missing random seed settings
        - Undocumented hyperparameters
        - Mismatched dependencies
        - Broken imports or missing files
        - Discrepancies between paper and code

        === CRITICAL QUESTIONS ===
        1. Does the model architecture EXACTLY match the paper?
        2. Are ALL hyperparameters from the paper present in config?
        3. Can the reported results be reproduced?
        4. What would prevent someone from using this code?""",
        expected_output="""Code Verification Report:

        ## Executive Summary
        - Overall Alignment Score: [X/100]
        - Recommendation: [Use as-is / Use with caution / Not recommended]
        - Critical Issues: [count]

        ## Repository Overview
        - Structure diagram
        - Key files identified

        ## Architecture Verification
        | Component | Paper | Code | Match |
        [Table comparing paper claims to code implementation]

        ## Completeness Checklist
        [Checkbox list of implemented vs missing components]

        ## Reproducibility Score: [X/50]
        - Environment setup: [X/10]
        - Data preparation: [X/10]
        - Training: [X/10]
        - Evaluation: [X/10]
        - Documentation: [X/10]

        ## Issues Found
        ### Critical (blocks reproduction)
        ### Major (significant effort to fix)
        ### Minor (cosmetic or easy fixes)

        ## Recommendations
        1. [Specific actionable recommendation]
        2. [Specific actionable recommendation]

        ## Verdict
        [Clear statement on whether this code can reproduce paper results]""",
        agent=agent,
        context=context,
    )


def create_synthesis_task(agent, context: list) -> Task:
    """
    Create task for synthesizing all findings into a final report.
    """
    return Task(
        description="""Synthesize all research findings into a comprehensive, actionable SymbyAI report.

        === SYNTHESIS OBJECTIVES ===

        Create a unified report that transforms raw research findings into actionable intelligence for researchers and practitioners.

        === REPORT STRUCTURE ===

        **1. EXECUTIVE SUMMARY** (Max 150 words)
        - One-sentence paper description
        - Key finding: Is this reproducible?
        - Bottom-line recommendation

        **2. PAPER ANALYSIS**
        - Problem & Motivation (2-3 sentences)
        - Method Overview (key innovation in plain language)
        - Main Results (quantitative highlights)
        - Limitations (acknowledged by authors + our assessment)

        **3. RESEARCH LANDSCAPE**
        - Where this fits: incremental / significant / breakthrough
        - Key competing approaches
        - Research trajectory (what came before, what's likely next)

        **4. IMPLEMENTATION STATUS**

        | Aspect | Status | Details |
        |--------|--------|---------|
        | Official Code | ✓/✗ | URL or N/A |
        | Community Code | ✓/✗ | Best option |
        | Pretrained Models | ✓/✗ | Availability |
        | Data | ✓/✗ | Access instructions |

        **5. REPRODUCIBILITY VERDICT**

        Overall Score: [X/100]

        | Factor | Score | Notes |
        |--------|-------|-------|
        | Code-Paper Alignment | /25 | |
        | Documentation | /20 | |
        | Dependencies | /15 | |
        | Data Access | /20 | |
        | Pretrained Models | /20 | |

        **6. ACTIONABLE RECOMMENDATIONS**

        For Researchers:
        - [ ] Specific action item 1
        - [ ] Specific action item 2

        For Practitioners:
        - [ ] Specific action item 1
        - [ ] Specific action item 2

        **7. QUICK START GUIDE**
        Step-by-step instructions to get started with the best available implementation.

        **8. KNOWN ISSUES & WORKAROUNDS**
        Document any issues discovered and solutions.

        === SYNTHESIS PRINCIPLES ===

        1. ACTIONABILITY: Every section should help the reader DO something
        2. HONESTY: Don't oversell - be clear about limitations
        3. SPECIFICITY: Avoid vague statements; use concrete details
        4. ACCESSIBILITY: Write for both experts and newcomers
        5. COMPLETENESS: Address what's missing, not just what's present""",
        expected_output="""SymbyAI Comprehensive Research Report

        # [Paper Title]

        ## Executive Summary
        [150-word max summary with key takeaway]

        ## Paper Analysis
        ### Problem & Motivation
        ### Method
        ### Results
        ### Limitations

        ## Research Landscape
        ### Position in Field
        ### Competing Approaches
        ### Research Trajectory

        ## Implementation Status
        [Status table]

        ## Reproducibility Assessment
        **Overall Score: [X/100]**
        [Detailed scoring table]

        ## Recommendations
        ### For Researchers
        ### For Practitioners

        ## Quick Start Guide
        ```bash
        # Step-by-step commands
        ```

        ## Known Issues & Workarounds
        [Issue list with solutions]

        ---
        *Report generated by SymbyAI Research Crew*""",
        agent=agent,
        context=context,
    )


def create_quick_research_task(agent, topic: str) -> Task:
    """
    Create a quick research task for a single topic.
    """
    return Task(
        description=f"""Execute rapid research sweep for topic: {topic}

        === TIME-BOXED RESEARCH (Target: 5 minutes) ===

        **1. ARXIV QUICK SEARCH**
        - Find 3-5 most relevant recent papers
        - Prioritize: high citations, reputable venues, last 2 years
        - Note arXiv IDs for follow-up

        **2. CONCEPT OVERVIEW**
        - Wikipedia/reliable source for core concept explanation
        - Key terminology definitions
        - Prerequisites to understand this topic

        **3. CODE AVAILABILITY CHECK**
        - Quick GitHub search for implementations
        - Check Papers With Code for top paper
        - Note best available repository

        **4. SYNTHESIS**
        - What is this topic about? (2-3 sentences)
        - Current state of the field
        - Best starting point for deeper dive

        === OUTPUT PRIORITIES ===
        - Accuracy over comprehensiveness
        - Actionable pointers over exhaustive lists
        - Clear "next steps" guidance""",
        expected_output="""Quick Research Summary: {topic}

        ## What is it?
        [2-3 sentence explanation]

        ## Key Papers
        | arXiv ID | Title | Year | Why Relevant |
        [3-5 papers max]

        ## Core Concepts
        - [Concept 1]: [Brief explanation]
        - [Concept 2]: [Brief explanation]

        ## Code Available
        - Best repo: [URL]
        - Alternative: [URL]

        ## Next Steps
        1. [Recommended first action]
        2. [Recommended second action]

        ## Deeper Dive Resources
        - [Survey paper or tutorial if found]""",
        agent=agent,
    )
