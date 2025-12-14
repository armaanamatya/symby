"""
Paper Code Executor Module

Integrates paper analysis with code extraction and sandbox execution.
Executes code from papers in isolated environments and validates results.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

from .schemas import PaperAnalysisResult, ExtractedCode
from .analyzer import PaperAnalyzer

# Import PCE modules
try:
    from ..pce.code_extractor import CodeExtractor, CodeManifest as PCECodeManifest
    from ..pce.sandbox_runner import SandboxRunner, ExecutionResult as SandboxResult
    PCE_AVAILABLE = True
except ImportError:
    PCE_AVAILABLE = False
    logging.warning("PCE modules not available. Code execution disabled.")

logger = logging.getLogger(__name__)


class PaperCodeExecutor:
    """
    Executes code extracted from research papers.

    Features:
    - Extracts code from papers (PDF, text, or arXiv ID)
    - Runs code in isolated Docker containers
    - Validates execution results against paper claims
    - Provides comprehensive execution reports
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_gpu: bool = True,
        max_timeout_seconds: int = 300,
        max_memory_gb: float = 8.0
    ):
        """
        Initialize the paper code executor.

        Args:
            api_key: NVIDIA API key for LLM analysis
            enable_gpu: Enable GPU for code execution
            max_timeout_seconds: Maximum execution timeout
            max_memory_gb: Maximum memory for execution
        """
        self.analyzer = PaperAnalyzer(api_key=api_key)

        if PCE_AVAILABLE:
            self.code_extractor = CodeExtractor()
            try:
                self.sandbox = SandboxRunner(
                    enable_gpu=enable_gpu,
                    max_timeout_seconds=max_timeout_seconds,
                    max_memory_gb=max_memory_gb,
                    require_gpu=False
                )
            except Exception as e:
                logger.warning(f"Sandbox initialization failed: {e}. Code execution disabled.")
                self.sandbox = None
        else:
            self.code_extractor = None
            self.sandbox = None

    async def analyze_and_extract(
        self,
        text: Optional[str] = None,
        pdf_path: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        arxiv_id: Optional[str] = None,
        paper_id: Optional[str] = None
    ) -> PaperAnalysisResult:
        """
        Analyze paper and extract code.

        Args:
            text: Paper text content
            pdf_path: Path to PDF file
            pdf_bytes: PDF file bytes
            arxiv_id: arXiv paper ID (e.g., "2010.11929")
            paper_id: Paper identifier

        Returns:
            PaperAnalysisResult with analysis and extracted code
        """
        # If arXiv ID provided, fetch and extract
        if arxiv_id and self.code_extractor:
            try:
                manifest = await self.code_extractor.extract_from_arxiv(arxiv_id)

                # Convert PCE code blocks to our format
                code_blocks = [
                    ExtractedCode(
                        language=cb.language,
                        code=cb.code,
                        description=f"Extracted from {cb.source}",
                        dependencies=list(manifest.dependencies),
                        executable=True
                    )
                    for cb in manifest.code_blocks
                ]

                # Create analysis result with extracted code
                result = self.analyzer.analyze(
                    text=text or "",
                    paper_id=paper_id or arxiv_id
                )
                result.code_blocks = code_blocks
                result.github_repos = manifest.github_repos

                return result

            except Exception as e:
                logger.error(f"arXiv extraction failed: {e}")
                # Fall through to regular analysis

        # Regular analysis
        return self.analyzer.analyze(
            text=text,
            pdf_path=pdf_path,
            pdf_bytes=pdf_bytes,
            paper_id=paper_id
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        environment: Optional[Dict[str, str]] = None,
        image: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code in a sandbox.

        Args:
            code: Code to execute
            language: Programming language
            environment: Additional environment variables
            image: Docker image to use

        Returns:
            Execution result dictionary
        """
        if not self.sandbox:
            return {
                "success": False,
                "error": "Sandbox not available",
                "stdout": "",
                "stderr": "Code execution is not available. Docker may not be running.",
                "execution_time_seconds": 0.0
            }

        # Select appropriate image
        if not image:
            if language == "python":
                image = "python:3.10-slim"
            else:
                image = "python:3.10-slim"  # Default to Python

        try:
            result = await self.sandbox.run(
                image=image,
                code=code,
                environment=environment or {}
            )

            return {
                "success": result.status == "success",
                "status": result.status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "execution_time_seconds": result.execution_time_seconds,
                "memory_peak_mb": result.memory_peak_mb,
                "gpu_utilization_pct": result.gpu_utilization_pct,
                "gpu_memory_used_mb": result.gpu_memory_used_mb
            }

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
                "execution_time_seconds": 0.0
            }

    async def execute_paper_code(
        self,
        analysis_result: PaperAnalysisResult,
        code_block_index: int = 0,
        environment: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute code from an analyzed paper.

        Args:
            analysis_result: Paper analysis result with code blocks
            code_block_index: Index of code block to execute
            environment: Additional environment variables

        Returns:
            Execution result with paper context
        """
        if not analysis_result.code_blocks:
            return {
                "success": False,
                "error": "No code blocks found in paper",
                "paper_id": analysis_result.paper_id
            }

        if code_block_index >= len(analysis_result.code_blocks):
            return {
                "success": False,
                "error": f"Code block index {code_block_index} out of range",
                "paper_id": analysis_result.paper_id
            }

        code_block = analysis_result.code_blocks[code_block_index]

        # Execute the code
        result = await self.execute_code(
            code=code_block.code,
            language=code_block.language,
            environment=environment
        )

        # Update the code block with execution result
        code_block.execution_result = result

        return {
            "paper_id": analysis_result.paper_id,
            "code_block_index": code_block_index,
            "language": code_block.language,
            **result
        }

    async def execute_all_code_blocks(
        self,
        analysis_result: PaperAnalysisResult,
        environment: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute all code blocks from an analyzed paper.

        Args:
            analysis_result: Paper analysis result with code blocks
            environment: Additional environment variables

        Returns:
            List of execution results
        """
        results = []

        for i, code_block in enumerate(analysis_result.code_blocks):
            if code_block.executable:
                result = await self.execute_paper_code(
                    analysis_result=analysis_result,
                    code_block_index=i,
                    environment=environment
                )
                results.append(result)

        return results

    def validate_execution(
        self,
        analysis_result: PaperAnalysisResult,
        execution_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate execution results against paper claims.

        Args:
            analysis_result: Paper analysis result
            execution_results: List of execution results

        Returns:
            Validation report
        """
        total_blocks = len(analysis_result.code_blocks)
        executed_blocks = len(execution_results)
        successful_blocks = sum(1 for r in execution_results if r.get("success"))

        # Check for expected metrics mentioned in paper
        claims_to_verify = []
        if analysis_result.impact_indicators.quantitative_improvements_mentioned:
            claims_to_verify.append("quantitative_improvements")
        if analysis_result.ml_adoption.ml_method_primary:
            claims_to_verify.append("ml_method_execution")

        # Analyze execution outputs for metrics
        extracted_metrics = []
        for result in execution_results:
            stdout = result.get("stdout", "")
            # Simple metric extraction (could be enhanced with LLM)
            import re
            metrics = re.findall(r'(\w+):\s*([\d.]+)', stdout)
            extracted_metrics.extend(metrics)

        return {
            "paper_id": analysis_result.paper_id,
            "validated_at": datetime.utcnow().isoformat(),
            "total_code_blocks": total_blocks,
            "executed_blocks": executed_blocks,
            "successful_blocks": successful_blocks,
            "execution_success_rate": successful_blocks / max(executed_blocks, 1),
            "claims_to_verify": claims_to_verify,
            "extracted_metrics": extracted_metrics,
            "overall_status": "verified" if successful_blocks == executed_blocks else "partially_verified"
        }

    async def full_pipeline(
        self,
        text: Optional[str] = None,
        pdf_path: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        paper_id: Optional[str] = None,
        execute_code: bool = True
    ) -> Dict[str, Any]:
        """
        Run the full analysis and execution pipeline.

        Args:
            text: Paper text content
            pdf_path: Path to PDF file
            arxiv_id: arXiv paper ID
            paper_id: Paper identifier
            execute_code: Whether to execute extracted code

        Returns:
            Complete pipeline results
        """
        # Analyze and extract
        analysis = await self.analyze_and_extract(
            text=text,
            pdf_path=pdf_path,
            arxiv_id=arxiv_id,
            paper_id=paper_id
        )

        result = {
            "analysis": analysis.to_dict(),
            "execution_results": [],
            "validation": None
        }

        # Execute code if requested
        if execute_code and analysis.code_blocks:
            execution_results = await self.execute_all_code_blocks(analysis)
            result["execution_results"] = execution_results

            # Validate
            validation = self.validate_execution(analysis, execution_results)
            result["validation"] = validation

        return result


class PaperQueryEngine:
    """
    Query engine for answering questions about papers.

    Uses RAG-style approach with the analyzed paper content.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the query engine."""
        self.analyzer = PaperAnalyzer(api_key=api_key)

        # Store paper context for follow-up queries
        self._paper_cache: Dict[str, Dict[str, Any]] = {}

    def load_paper(
        self,
        paper_id: str,
        text: Optional[str] = None,
        pdf_path: Optional[str] = None,
        analysis: Optional[PaperAnalysisResult] = None
    ) -> PaperAnalysisResult:
        """
        Load a paper for querying.

        Args:
            paper_id: Paper identifier
            text: Paper text content
            pdf_path: Path to PDF file
            analysis: Pre-computed analysis result

        Returns:
            Paper analysis result
        """
        if analysis:
            self._paper_cache[paper_id] = {
                "analysis": analysis,
                "text": text or ""
            }
            return analysis

        # Analyze the paper
        result = self.analyzer.analyze(
            text=text,
            pdf_path=pdf_path,
            paper_id=paper_id
        )

        # Cache for follow-up queries
        if pdf_path:
            text = self.analyzer.extract_text_from_pdf(pdf_path)

        self._paper_cache[paper_id] = {
            "analysis": result,
            "text": text or ""
        }

        return result

    def query(
        self,
        paper_id: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Query a loaded paper.

        Args:
            paper_id: Paper identifier
            question: Question to answer

        Returns:
            Query response
        """
        if paper_id not in self._paper_cache:
            return {
                "error": f"Paper {paper_id} not loaded. Call load_paper first.",
                "answer": None
            }

        cache = self._paper_cache[paper_id]
        text = cache.get("text", "")

        if not text:
            return {
                "error": "Paper text not available for querying",
                "answer": None
            }

        return self.analyzer.query_paper(text, question, paper_id)

    def get_analysis(self, paper_id: str) -> Optional[PaperAnalysisResult]:
        """Get cached analysis for a paper."""
        cache = self._paper_cache.get(paper_id)
        if cache:
            return cache.get("analysis")
        return None

    def list_papers(self) -> List[str]:
        """List all loaded paper IDs."""
        return list(self._paper_cache.keys())


async def run_paper_analysis_pipeline(
    pdf_path: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    text: Optional[str] = None,
    execute_code: bool = False,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the full paper analysis pipeline.

    Args:
        pdf_path: Path to PDF file
        arxiv_id: arXiv paper ID
        text: Paper text content
        execute_code: Whether to execute extracted code
        output_path: Path to save JSON output

    Returns:
        Complete analysis results
    """
    executor = PaperCodeExecutor()

    result = await executor.full_pipeline(
        pdf_path=pdf_path,
        arxiv_id=arxiv_id,
        text=text,
        execute_code=execute_code
    )

    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python executor.py <pdf_path_or_arxiv_id> [--execute]")
            sys.exit(1)

        input_arg = sys.argv[1]
        execute = "--execute" in sys.argv

        if input_arg.endswith('.pdf'):
            result = await run_paper_analysis_pipeline(
                pdf_path=input_arg,
                execute_code=execute
            )
        else:
            result = await run_paper_analysis_pipeline(
                arxiv_id=input_arg,
                execute_code=execute
            )

        import json
        print(json.dumps(result, indent=2))

    asyncio.run(main())
