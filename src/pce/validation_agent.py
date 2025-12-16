"""
Validation Agent for Paper Code Execution (PCE)

This module validates execution results against quantitative claims from academic papers.
It uses LLM-based claim extraction and fuzzy metric matching to compare claimed vs actual values.

Supports multiple LLM backends with fallback chain:
1. Local NVIDIA NIM
2. Anthropic Claude API
3. OpenAI API
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


# Configure logging
logger = logging.getLogger(__name__)


class ClaimType(str, Enum):
    """Types of quantitative claims in papers."""
    ACCURACY = "accuracy"
    LOSS = "loss"
    TIME = "time"
    MEMORY = "memory"
    OTHER = "other"


class ComparisonType(str, Enum):
    """Comparison operators for claims."""
    EQ = "eq"      # Equal to
    GT = "gt"      # Greater than
    LT = "lt"      # Less than
    APPROX = "approx"  # Approximately equal


class ValidationStatus(str, Enum):
    """Status of validation results."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"


class DiscrepancySeverity(str, Enum):
    """Severity levels for discrepancies."""
    MINOR = "minor"      # < 5% difference
    MAJOR = "major"      # 5-20% difference
    CRITICAL = "critical"  # > 20% difference


@dataclass
class PaperClaim:
    """
    Represents a quantitative claim extracted from a paper.

    Attributes:
        claim_type: Category of the claim (accuracy, loss, time, memory, other)
        metric_name: Name of the metric (e.g., 'accuracy', 'F1 score', 'inference time')
        claimed_value: The numerical value claimed in the paper
        comparison: How to compare (eq, gt, lt, approx)
        tolerance_pct: Tolerance percentage for comparison (default 5.0%)
        context: Additional context (e.g., 'on ImageNet', 'with batch size 32')
        source_text: Original text from which the claim was extracted
    """
    claim_type: str
    metric_name: str
    claimed_value: float
    comparison: str = "approx"
    tolerance_pct: float = 5.0
    context: str = ""
    source_text: str = ""

    def __post_init__(self):
        """Validate claim types."""
        if self.claim_type not in [ct.value for ct in ClaimType]:
            self.claim_type = ClaimType.OTHER.value
        if self.comparison not in [cp.value for cp in ComparisonType]:
            self.comparison = ComparisonType.APPROX.value


@dataclass
class Discrepancy:
    """
    Represents a discrepancy between claimed and actual values.

    Attributes:
        claim: The original paper claim
        actual_value: The actual value from execution
        difference_pct: Percentage difference between claimed and actual
        severity: Severity classification (minor, major, critical)
    """
    claim: PaperClaim
    actual_value: float
    difference_pct: float
    severity: str

    def __post_init__(self):
        """Validate severity."""
        if self.severity not in [ds.value for ds in DiscrepancySeverity]:
            self.severity = DiscrepancySeverity.MINOR.value


@dataclass
class ValidationResult:
    """
    Complete result of validating paper claims against execution output.

    Attributes:
        status: Overall validation status (pass, fail, partial, inconclusive)
        claims_tested: Number of claims that were tested
        claims_passed: Number of claims that passed validation
        discrepancies: List of discrepancies found
        summary: Human-readable summary of validation results
        matched_metrics: Dict mapping claim metric names to actual metric names
        unmatched_claims: Claims that couldn't be matched to execution output
    """
    status: str
    claims_tested: int
    claims_passed: int
    discrepancies: List[Discrepancy]
    summary: str
    matched_metrics: Dict[str, str] = field(default_factory=dict)
    unmatched_claims: List[PaperClaim] = field(default_factory=list)

    def __post_init__(self):
        """Validate status."""
        if self.status not in [vs.value for vs in ValidationStatus]:
            self.status = ValidationStatus.INCONCLUSIVE.value


# Metric name normalization mappings
METRIC_ALIASES: Dict[str, List[str]] = {
    "accuracy": ["acc", "accuracy", "top-1", "top1", "top_1", "test_acc", "val_acc", "train_acc"],
    "loss": ["loss", "train_loss", "val_loss", "test_loss", "training_loss", "validation_loss"],
    "f1_score": ["f1", "f1_score", "f1-score", "f1score", "f-score", "fscore"],
    "precision": ["precision", "prec", "p"],
    "recall": ["recall", "rec", "r", "sensitivity"],
    "time": ["time", "runtime", "inference_time", "training_time", "latency", "elapsed", "duration"],
    "memory": ["mem", "memory", "gpu_memory", "ram", "vram", "gpu_mem", "memory_usage"],
    "throughput": ["throughput", "fps", "samples_per_sec", "images_per_second", "tokens_per_sec"],
    "perplexity": ["ppl", "perplexity"],
    "bleu": ["bleu", "bleu_score", "bleu-4", "bleu4"],
    "rouge": ["rouge", "rouge-l", "rouge_l", "rougel"],
    "auc": ["auc", "auroc", "roc_auc", "auc_score"],
    "mae": ["mae", "mean_absolute_error"],
    "mse": ["mse", "mean_squared_error"],
    "rmse": ["rmse", "root_mean_squared_error"],
}


class LLMClient:
    """
    Abstract LLM client with support for multiple backends.

    Implements fallback chain:
    1. Local NVIDIA NIM
    2. Anthropic Claude API
    3. OpenAI API
    """

    def __init__(
        self,
        nvidia_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        nvidia_base_url: str = "https://integrate.api.nvidia.com/v1",
        preferred_backend: Optional[str] = None
    ):
        """
        Initialize LLM client with available backends.

        Args:
            nvidia_api_key: NVIDIA API key (or NVIDIA_API_KEY env var)
            anthropic_api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
            openai_api_key: OpenAI API key (or OPENAI_API_KEY env var)
            nvidia_base_url: NVIDIA NIM API endpoint
            preferred_backend: Preferred backend to use ('nvidia', 'anthropic', 'openai')
        """
        self.nvidia_api_key = nvidia_api_key or os.environ.get('NVIDIA_API_KEY')
        self.anthropic_api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.nvidia_base_url = nvidia_base_url
        self.preferred_backend = preferred_backend

        # Initialize clients lazily
        self._nvidia_client = None
        self._anthropic_client = None
        self._openai_client = None

        # Determine available backends
        self._available_backends = []
        if self.nvidia_api_key:
            self._available_backends.append('nvidia')
        if self.anthropic_api_key:
            self._available_backends.append('anthropic')
        if self.openai_api_key:
            self._available_backends.append('openai')

        if not self._available_backends:
            logger.warning("No LLM API keys found. Claim extraction will not work.")

    @property
    def nvidia_client(self):
        """Lazy initialization of NVIDIA client."""
        if self._nvidia_client is None and self.nvidia_api_key:
            try:
                from openai import OpenAI
                self._nvidia_client = OpenAI(
                    base_url=self.nvidia_base_url,
                    api_key=self.nvidia_api_key
                )
            except ImportError:
                logger.warning("openai package not installed. NVIDIA NIM not available.")
        return self._nvidia_client

    @property
    def anthropic_client(self):
        """Lazy initialization of Anthropic client."""
        if self._anthropic_client is None and self.anthropic_api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(
                    api_key=self.anthropic_api_key
                )
            except ImportError:
                logger.warning("anthropic package not installed.")
        return self._anthropic_client

    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None and self.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except ImportError:
                logger.warning("openai package not installed.")
        return self._openai_client

    def _get_backend_order(self) -> List[str]:
        """Get the order of backends to try."""
        if self.preferred_backend and self.preferred_backend in self._available_backends:
            order = [self.preferred_backend]
            order.extend([b for b in self._available_backends if b != self.preferred_backend])
            return order
        return self._available_backends

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Get completion from LLM with fallback chain.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response text or None if all backends fail
        """
        import asyncio

        for backend in self._get_backend_order():
            try:
                if backend == 'nvidia':
                    # Run synchronous SDK call in thread pool
                    return await asyncio.to_thread(
                        self._complete_nvidia_sync,
                        system_prompt, user_prompt, temperature, max_tokens
                    )
                elif backend == 'anthropic':
                    return await asyncio.to_thread(
                        self._complete_anthropic_sync,
                        system_prompt, user_prompt, temperature, max_tokens
                    )
                elif backend == 'openai':
                    return await asyncio.to_thread(
                        self._complete_openai_sync,
                        system_prompt, user_prompt, temperature, max_tokens
                    )
            except Exception as e:
                logger.warning(f"Backend {backend} failed: {e}")
                continue

        logger.error("All LLM backends failed")
        return None

    def _complete_nvidia_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Complete using NVIDIA NIM (synchronous)."""
        if not self.nvidia_client:
            raise RuntimeError("NVIDIA client not available")

        response = self.nvidia_client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def _complete_anthropic_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Complete using Anthropic Claude (synchronous)."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not available")

        response = self.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text

    def _complete_openai_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Complete using OpenAI (synchronous)."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not available")

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class ValidationAgent:
    """
    Agent for validating execution results against paper claims.

    This agent:
    1. Extracts quantitative claims from paper text using LLM
    2. Parses execution output for metric values
    3. Compares claimed vs actual values with fuzzy matching
    4. Generates human-readable validation reports
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        nvidia_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the ValidationAgent.

        Args:
            llm_client: Pre-configured LLM client (optional)
            nvidia_api_key: NVIDIA API key for NIM
            anthropic_api_key: Anthropic API key for Claude
            openai_api_key: OpenAI API key
        """
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(
                nvidia_api_key=nvidia_api_key,
                anthropic_api_key=anthropic_api_key,
                openai_api_key=openai_api_key
            )

        # Build reverse mapping for fuzzy matching
        self._metric_reverse_map: Dict[str, str] = {}
        for canonical, aliases in METRIC_ALIASES.items():
            for alias in aliases:
                self._metric_reverse_map[alias.lower()] = canonical

    async def extract_claims_from_paper(
        self,
        paper_text: str
    ) -> List[PaperClaim]:
        """
        Use LLM to extract quantitative claims from paper text.

        Analyzes the paper text to find:
        - Performance metrics (accuracy, F1 score, etc.)
        - Timing information (training time, inference latency)
        - Resource usage (memory, GPU utilization)
        - Comparison statements (e.g., "outperforms baseline by 5%")

        Args:
            paper_text: Full text or relevant sections of the paper

        Returns:
            List of PaperClaim objects extracted from the paper
        """
        system_prompt = """You are an expert at extracting quantitative claims from scientific papers.

Your task is to extract ALL quantitative claims from the given paper text.

For each claim, identify:
1. metric_name: The name of the metric (e.g., 'accuracy', 'F1 score', 'inference time', 'memory usage')
2. claimed_value: The numerical value (convert percentages to decimals, e.g., 95% -> 0.95)
3. claim_type: One of: accuracy, loss, time, memory, other
4. comparison: How to interpret the value:
   - "eq" if exact value
   - "gt" if "greater than" or "at least" or "minimum"
   - "lt" if "less than" or "at most" or "maximum"
   - "approx" if "approximately" or "around" or no qualifier
5. context: Any relevant context (dataset name, batch size, hardware, etc.)
6. source_text: The original sentence containing the claim

Return a JSON array of claims. Example:
[
  {
    "metric_name": "accuracy",
    "claimed_value": 0.95,
    "claim_type": "accuracy",
    "comparison": "eq",
    "tolerance_pct": 5.0,
    "context": "on ImageNet validation set",
    "source_text": "Our model achieves 95% accuracy on ImageNet."
  },
  {
    "metric_name": "inference_time",
    "claimed_value": 0.5,
    "claim_type": "time",
    "comparison": "lt",
    "tolerance_pct": 10.0,
    "context": "per image on NVIDIA V100",
    "source_text": "Inference takes less than 0.5 seconds per image on V100."
  }
]

Important guidelines:
- Extract ALL numerical claims, even if they seem minor
- Convert all percentages to decimals (95% -> 0.95)
- Convert all time units to seconds
- Convert memory units to GB
- Set appropriate tolerance_pct based on claim precision (5% default, 10% for timing)
- Only output valid JSON, no explanations"""

        user_prompt = f"""Extract all quantitative claims from this paper text:

{paper_text[:8000]}  # Limit to avoid token limits

Return ONLY a JSON array of claims."""

        try:
            response = await self.llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=3000
            )

            if not response:
                logger.error("LLM returned empty response for claim extraction")
                return []

            # Parse JSON response
            claims_data = self._parse_json_response(response)

            if not claims_data:
                logger.warning("No claims extracted from paper")
                return []

            # Convert to PaperClaim objects
            claims = []
            for claim_dict in claims_data:
                try:
                    claim = PaperClaim(
                        claim_type=claim_dict.get('claim_type', 'other'),
                        metric_name=claim_dict.get('metric_name', 'unknown'),
                        claimed_value=float(claim_dict.get('claimed_value', 0)),
                        comparison=claim_dict.get('comparison', 'approx'),
                        tolerance_pct=float(claim_dict.get('tolerance_pct', 5.0)),
                        context=claim_dict.get('context', ''),
                        source_text=claim_dict.get('source_text', '')
                    )
                    claims.append(claim)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse claim: {claim_dict}, error: {e}")
                    continue

            logger.info(f"Extracted {len(claims)} claims from paper")
            return claims

        except Exception as e:
            logger.error(f"Failed to extract claims: {e}")
            return []

    def extract_metrics_from_output(
        self,
        execution_output: str
    ) -> Dict[str, float]:
        """
        Parse execution stdout/stderr for metric values.

        Recognizes patterns like:
        - "accuracy: 0.95" or "accuracy = 0.95"
        - "loss: 0.023" or "loss = 0.023"
        - "F1: 0.87" or "F1 score: 0.87"
        - "Time: 2.3s" or "time: 2.3 seconds"
        - "Memory: 4.2 GB" or "GPU memory: 4.2GB"
        - "Epoch 10: loss=0.023, acc=0.95"

        Args:
            execution_output: Combined stdout and stderr from execution

        Returns:
            Dictionary mapping normalized metric names to float values
        """
        metrics: Dict[str, float] = {}

        # Patterns for different metric formats
        patterns = [
            # Standard format: "metric: value" or "metric = value"
            r'(\b[\w_-]+)\s*[:=]\s*([\d.]+)\s*(%|percent|pct)?',

            # With units: "metric: 2.3s" or "metric: 4.2 GB"
            r'(\b[\w_-]+)\s*[:=]\s*([\d.]+)\s*(s|sec|seconds?|ms|milliseconds?|GB|MB|KB|gb|mb|kb)?',

            # Parenthetical: "accuracy (0.95)" or "F1 (87%)"
            r'(\b[\w_-]+)\s*\(\s*([\d.]+)\s*(%|percent)?\s*\)',

            # Verbose: "achieved an accuracy of 0.95"
            r'(?:achieved|obtained|reached|got|scored)\s+(?:an?\s+)?(\w+)\s+(?:of\s+)?([\d.]+)\s*(%)?',

            # Table-like: "| accuracy | 0.95 |"
            r'\|\s*(\w+)\s*\|\s*([\d.]+)\s*(%|percent)?\s*\|',

            # JSON-like: "\"accuracy\": 0.95"
            r'"(\w+)"\s*:\s*([\d.]+)',

            # Final metrics report format
            r'(?:Final|Test|Val|Validation|Eval|Evaluation)\s+(\w+)\s*[:=]\s*([\d.]+)\s*(%)?',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, execution_output, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                metric_name = groups[0].lower().strip()
                value_str = groups[1]
                is_percentage = len(groups) > 2 and groups[2] in ['%', 'percent', 'pct']

                try:
                    value = float(value_str)

                    # Handle percentages
                    if is_percentage and value > 1:
                        value = value / 100.0

                    # Handle time units
                    if len(groups) > 2 and groups[2]:
                        unit = groups[2].lower()
                        if unit in ['ms', 'milliseconds', 'millisecond']:
                            value = value / 1000.0  # Convert to seconds
                        elif unit in ['mb']:
                            value = value / 1024.0  # Convert to GB
                        elif unit in ['kb']:
                            value = value / (1024.0 * 1024.0)  # Convert to GB

                    # Normalize metric name
                    normalized_name = self._normalize_metric_name(metric_name)

                    # Keep the most recent/final value for each metric
                    metrics[normalized_name] = value

                except ValueError:
                    continue

        # Also try to extract from JSON blocks in output
        json_metrics = self._extract_json_metrics(execution_output)
        metrics.update(json_metrics)

        logger.info(f"Extracted {len(metrics)} metrics from output: {list(metrics.keys())}")
        return metrics

    def compare_claims(
        self,
        claims: List[PaperClaim],
        actual_metrics: Dict[str, float]
    ) -> ValidationResult:
        """
        Compare claimed vs actual values.

        Matching logic:
        1. Fuzzy match metric names (e.g., "acc" matches "accuracy")
        2. Apply tolerance for comparison
        3. Classify discrepancy severity:
           - minor: < 5% difference
           - major: 5-20% difference
           - critical: > 20% difference

        Args:
            claims: List of claims extracted from paper
            actual_metrics: Dictionary of metrics from execution output

        Returns:
            ValidationResult with comparison results and discrepancies
        """
        discrepancies: List[Discrepancy] = []
        matched_metrics: Dict[str, str] = {}
        unmatched_claims: List[PaperClaim] = []
        claims_tested = 0
        claims_passed = 0

        for claim in claims:
            # Try to find matching metric in actual output
            matched_metric, matched_value = self._find_matching_metric(
                claim.metric_name, actual_metrics
            )

            if matched_metric is None:
                unmatched_claims.append(claim)
                continue

            claims_tested += 1
            matched_metrics[claim.metric_name] = matched_metric

            # Compare values
            passed, difference_pct = self._compare_values(
                claim.claimed_value,
                matched_value,
                claim.comparison,
                claim.tolerance_pct
            )

            if passed:
                claims_passed += 1
            else:
                # Classify severity
                severity = self._classify_severity(difference_pct)

                discrepancy = Discrepancy(
                    claim=claim,
                    actual_value=matched_value,
                    difference_pct=difference_pct,
                    severity=severity
                )
                discrepancies.append(discrepancy)

        # Determine overall status
        status = self._determine_status(
            claims_tested, claims_passed, len(unmatched_claims), len(claims)
        )

        # Generate summary
        summary = self._generate_summary(
            status, claims_tested, claims_passed,
            len(discrepancies), len(unmatched_claims)
        )

        return ValidationResult(
            status=status,
            claims_tested=claims_tested,
            claims_passed=claims_passed,
            discrepancies=discrepancies,
            summary=summary,
            matched_metrics=matched_metrics,
            unmatched_claims=unmatched_claims
        )

    def generate_report(
        self,
        validation_result: ValidationResult,
        format: str = "markdown"
    ) -> str:
        """
        Generate human-readable validation report.

        Includes:
        - Summary (pass/fail/partial)
        - Claims tested with results
        - Discrepancies highlighted
        - Recommendations

        Args:
            validation_result: The validation result to report on
            format: Output format ("markdown" or "text")

        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._generate_markdown_report(validation_result)
        else:
            return self._generate_text_report(validation_result)

    def _normalize_metric_name(self, name: str) -> str:
        """Normalize metric name to canonical form."""
        name_lower = name.lower().strip().replace('-', '_').replace(' ', '_')

        # Remove common prefixes/suffixes
        prefixes = ['train_', 'val_', 'test_', 'final_', 'best_', 'avg_', 'mean_']
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                name_lower = name_lower[len(prefix):]
                break

        # Look up in reverse map
        if name_lower in self._metric_reverse_map:
            return self._metric_reverse_map[name_lower]

        return name_lower

    def _find_matching_metric(
        self,
        claim_metric: str,
        actual_metrics: Dict[str, float]
    ) -> Tuple[Optional[str], Optional[float]]:
        """Find best matching metric using fuzzy matching."""
        # Normalize claim metric name
        normalized_claim = self._normalize_metric_name(claim_metric)

        # Direct match
        if normalized_claim in actual_metrics:
            return normalized_claim, actual_metrics[normalized_claim]

        # Try original name
        claim_lower = claim_metric.lower()
        if claim_lower in actual_metrics:
            return claim_lower, actual_metrics[claim_lower]

        # Fuzzy matching: check if any actual metric maps to the same canonical name
        for actual_name, actual_value in actual_metrics.items():
            normalized_actual = self._normalize_metric_name(actual_name)
            if normalized_actual == normalized_claim:
                return actual_name, actual_value

        # Check for partial matches
        for actual_name, actual_value in actual_metrics.items():
            if normalized_claim in actual_name.lower() or actual_name.lower() in normalized_claim:
                return actual_name, actual_value

        return None, None

    def _compare_values(
        self,
        claimed: float,
        actual: float,
        comparison: str,
        tolerance_pct: float
    ) -> Tuple[bool, float]:
        """Compare claimed vs actual value and return (passed, difference_pct)."""
        if claimed == 0:
            difference_pct = abs(actual) * 100 if actual != 0 else 0
        else:
            difference_pct = abs(actual - claimed) / abs(claimed) * 100

        if comparison == ComparisonType.EQ.value:
            passed = difference_pct <= tolerance_pct
        elif comparison == ComparisonType.GT.value:
            passed = actual >= claimed * (1 - tolerance_pct / 100)
        elif comparison == ComparisonType.LT.value:
            passed = actual <= claimed * (1 + tolerance_pct / 100)
        elif comparison == ComparisonType.APPROX.value:
            passed = difference_pct <= tolerance_pct
        else:
            passed = difference_pct <= tolerance_pct

        return passed, difference_pct

    def _classify_severity(self, difference_pct: float) -> str:
        """Classify discrepancy severity based on percentage difference."""
        if difference_pct < 5:
            return DiscrepancySeverity.MINOR.value
        elif difference_pct < 20:
            return DiscrepancySeverity.MAJOR.value
        else:
            return DiscrepancySeverity.CRITICAL.value

    def _determine_status(
        self,
        tested: int,
        passed: int,
        unmatched: int,
        total: int
    ) -> str:
        """Determine overall validation status."""
        if total == 0:
            return ValidationStatus.INCONCLUSIVE.value

        if tested == 0:
            return ValidationStatus.INCONCLUSIVE.value

        if passed == tested:
            if unmatched == 0:
                return ValidationStatus.PASS.value
            else:
                return ValidationStatus.PARTIAL.value
        elif passed > 0:
            return ValidationStatus.PARTIAL.value
        else:
            return ValidationStatus.FAIL.value

    def _generate_summary(
        self,
        status: str,
        tested: int,
        passed: int,
        discrepancies: int,
        unmatched: int
    ) -> str:
        """Generate summary text."""
        if status == ValidationStatus.PASS.value:
            return f"All {tested} claims validated successfully."
        elif status == ValidationStatus.PARTIAL.value:
            return f"Partial validation: {passed}/{tested} claims passed, {discrepancies} discrepancies found, {unmatched} claims could not be matched."
        elif status == ValidationStatus.FAIL.value:
            return f"Validation failed: 0/{tested} claims passed, {discrepancies} critical discrepancies found."
        else:
            return f"Inconclusive: Could not match {unmatched} claims to execution output."

    def _extract_json_metrics(self, output: str) -> Dict[str, float]:
        """Extract metrics from JSON blocks in output."""
        metrics = {}

        # Find JSON-like structures
        json_patterns = [
            r'\{[^{}]*\}',  # Simple objects
            r'\{(?:[^{}]|\{[^{}]*\})*\}'  # Nested objects
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, output)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                normalized = self._normalize_metric_name(key)
                                metrics[normalized] = float(value)
                except (json.JSONDecodeError, ValueError):
                    continue

        return metrics

    def _parse_json_response(self, response: str) -> List[Dict]:
        """Parse JSON from LLM response, handling common issues."""
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find individual JSON objects
        objects = re.findall(r'\{[^{}]*\}', response)
        results = []
        for obj in objects:
            try:
                results.append(json.loads(obj))
            except json.JSONDecodeError:
                continue

        return results

    def _generate_markdown_report(self, result: ValidationResult) -> str:
        """Generate markdown format report."""
        lines = []

        # Header
        status_emoji = {
            ValidationStatus.PASS.value: "\u2705",
            ValidationStatus.PARTIAL.value: "\u26a0\ufe0f",
            ValidationStatus.FAIL.value: "\u274c",
            ValidationStatus.INCONCLUSIVE.value: "\u2754"
        }

        lines.append(f"# Validation Report {status_emoji.get(result.status, '')}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"**Status:** {result.status.upper()}")
        lines.append(f"**Claims Tested:** {result.claims_tested}")
        lines.append(f"**Claims Passed:** {result.claims_passed}")
        lines.append(f"**Discrepancies:** {len(result.discrepancies)}")
        lines.append("")
        lines.append(result.summary)
        lines.append("")

        # Discrepancies
        if result.discrepancies:
            lines.append("## Discrepancies")
            lines.append("")

            for i, disc in enumerate(result.discrepancies, 1):
                severity_emoji = {
                    DiscrepancySeverity.MINOR.value: "\U0001f7e1",
                    DiscrepancySeverity.MAJOR.value: "\U0001f7e0",
                    DiscrepancySeverity.CRITICAL.value: "\U0001f534"
                }

                lines.append(f"### {i}. {disc.claim.metric_name} {severity_emoji.get(disc.severity, '')}")
                lines.append("")
                lines.append(f"- **Claimed:** {disc.claim.claimed_value}")
                lines.append(f"- **Actual:** {disc.actual_value}")
                lines.append(f"- **Difference:** {disc.difference_pct:.1f}%")
                lines.append(f"- **Severity:** {disc.severity}")
                if disc.claim.context:
                    lines.append(f"- **Context:** {disc.claim.context}")
                if disc.claim.source_text:
                    lines.append(f"- **Source:** \"{disc.claim.source_text}\"")
                lines.append("")

        # Unmatched claims
        if result.unmatched_claims:
            lines.append("## Unmatched Claims")
            lines.append("")
            lines.append("The following claims could not be matched to execution output:")
            lines.append("")

            for claim in result.unmatched_claims:
                lines.append(f"- **{claim.metric_name}**: {claim.claimed_value}")
                if claim.context:
                    lines.append(f"  - Context: {claim.context}")
            lines.append("")

        # Matched metrics
        if result.matched_metrics:
            lines.append("## Metric Matching")
            lines.append("")
            lines.append("| Paper Claim | Matched To |")
            lines.append("|-------------|------------|")
            for claim_metric, actual_metric in result.matched_metrics.items():
                lines.append(f"| {claim_metric} | {actual_metric} |")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        if result.status == ValidationStatus.PASS.value:
            lines.append("- Results match paper claims within tolerance")
            lines.append("- Consider running additional tests with different configurations")
        elif result.status == ValidationStatus.PARTIAL.value:
            lines.append("- Review discrepancies to determine if they are expected")
            lines.append("- Check if hardware/environment differences explain variations")
            lines.append("- Consider adjusting tolerance for noisy metrics")
        elif result.status == ValidationStatus.FAIL.value:
            lines.append("- Significant discrepancies detected - review code implementation")
            lines.append("- Check for bugs in data preprocessing or model configuration")
            lines.append("- Verify dataset versions match those used in the paper")
        else:
            lines.append("- Many claims could not be matched to output")
            lines.append("- Ensure execution produces expected metric outputs")
            lines.append("- Add logging/printing of metrics in standard formats")

        return "\n".join(lines)

    def _generate_text_report(self, result: ValidationResult) -> str:
        """Generate plain text format report."""
        lines = []

        lines.append("=" * 60)
        lines.append("VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Status: {result.status.upper()}")
        lines.append(f"Claims Tested: {result.claims_tested}")
        lines.append(f"Claims Passed: {result.claims_passed}")
        lines.append(f"Discrepancies: {len(result.discrepancies)}")
        lines.append("")
        lines.append(result.summary)
        lines.append("")

        if result.discrepancies:
            lines.append("-" * 40)
            lines.append("DISCREPANCIES")
            lines.append("-" * 40)

            for i, disc in enumerate(result.discrepancies, 1):
                lines.append(f"\n{i}. {disc.claim.metric_name} [{disc.severity.upper()}]")
                lines.append(f"   Claimed: {disc.claim.claimed_value}")
                lines.append(f"   Actual:  {disc.actual_value}")
                lines.append(f"   Diff:    {disc.difference_pct:.1f}%")

        if result.unmatched_claims:
            lines.append("")
            lines.append("-" * 40)
            lines.append("UNMATCHED CLAIMS")
            lines.append("-" * 40)

            for claim in result.unmatched_claims:
                lines.append(f"- {claim.metric_name}: {claim.claimed_value}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# Convenience function for quick validation
async def validate_paper_execution(
    paper_text: str,
    execution_output: str,
    nvidia_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> ValidationResult:
    """
    Convenience function to validate execution against paper claims.

    Args:
        paper_text: Text content of the paper
        execution_output: Combined stdout/stderr from execution
        nvidia_api_key: Optional NVIDIA API key
        anthropic_api_key: Optional Anthropic API key
        openai_api_key: Optional OpenAI API key

    Returns:
        ValidationResult with comparison results
    """
    agent = ValidationAgent(
        nvidia_api_key=nvidia_api_key,
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key
    )

    claims = await agent.extract_claims_from_paper(paper_text)
    metrics = agent.extract_metrics_from_output(execution_output)
    result = agent.compare_claims(claims, metrics)

    return result


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def main():
        print("\n=== Validation Agent Test ===\n")

        # Create agent
        agent = ValidationAgent()

        # Test metric extraction
        test_output = """
        Training complete!
        Final Results:
        - accuracy: 0.9523
        - loss: 0.0234
        - F1 score: 0.9156
        - inference_time: 0.45s
        - GPU memory: 4.2 GB

        {"test_accuracy": 0.9501, "test_loss": 0.0251}
        """

        print("Testing metric extraction...")
        metrics = agent.extract_metrics_from_output(test_output)
        print(f"Extracted metrics: {metrics}")

        # Test claim comparison
        test_claims = [
            PaperClaim(
                claim_type="accuracy",
                metric_name="accuracy",
                claimed_value=0.95,
                comparison="approx",
                tolerance_pct=5.0,
                context="on test set"
            ),
            PaperClaim(
                claim_type="loss",
                metric_name="loss",
                claimed_value=0.02,
                comparison="lt",
                tolerance_pct=10.0
            ),
            PaperClaim(
                claim_type="time",
                metric_name="inference_time",
                claimed_value=0.5,
                comparison="lt",
                tolerance_pct=20.0
            )
        ]

        print("\nTesting claim comparison...")
        result = agent.compare_claims(test_claims, metrics)
        print(f"Status: {result.status}")
        print(f"Passed: {result.claims_passed}/{result.claims_tested}")

        # Generate report
        print("\nGenerated Report:")
        print("=" * 50)
        report = agent.generate_report(result, format="markdown")
        print(report)

        print("\n=== Test Complete ===")

    asyncio.run(main())
