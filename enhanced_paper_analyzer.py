"""
Enhanced Paper Analyzer - Closer to SymbyAI Vision
Includes code analysis, data validation, and actionable insights
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import zipfile


class EnhancedPaperAnalyzer:
    """
    Enhanced analyzer that goes beyond basic text analysis
    to provide SymbyAI-like comprehensive insights
    """

    def __init__(self):
        self.issues = []
        self.suggestions = []
        self.inconsistencies = []

    def analyze_paper_comprehensive(
        self,
        paper_text: str,
        code_files: Optional[List[Path]] = None,
        data_files: Optional[List[Path]] = None,
        supplementary_files: Optional[List[Path]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis matching SymbyAI vision

        Args:
            paper_text: Main paper text
            code_files: List of code files submitted
            data_files: List of data files submitted
            supplementary_files: List of supplementary materials

        Returns:
            Comprehensive analysis report
        """
        self.issues = []
        self.suggestions = []
        self.inconsistencies = []

        report = {
            'paper_analysis': self._analyze_paper_text(paper_text),
            'code_analysis': self._analyze_code(code_files) if code_files else None,
            'data_analysis': self._analyze_data(data_files) if data_files else None,
            'reproducibility_assessment': self._assess_reproducibility(
                paper_text, code_files, data_files
            ),
            'method_validation': self._validate_methods(paper_text, code_files),
            'consistency_check': self._check_consistency(paper_text, code_files, data_files),
            'issues_found': self.issues,
            'suggestions_for_improvement': self.suggestions,
            'inconsistencies': self.inconsistencies,
            'overall_quality_score': 0,  # Will calculate
            'ready_for_publication': False  # Will determine
        }

        # Calculate overall quality score
        report['overall_quality_score'] = self._calculate_quality_score(report)
        report['ready_for_publication'] = report['overall_quality_score'] >= 70

        return report

    def _analyze_paper_text(self, text: str) -> Dict[str, Any]:
        """Analyze paper text for structure and quality"""
        analysis = {
            'has_abstract': bool(re.search(r'\babstract\b', text, re.IGNORECASE)),
            'has_introduction': bool(re.search(r'\bintroduction\b', text, re.IGNORECASE)),
            'has_methods': bool(re.search(r'\b(methods|methodology)\b', text, re.IGNORECASE)),
            'has_results': bool(re.search(r'\bresults\b', text, re.IGNORECASE)),
            'has_discussion': bool(re.search(r'\bdiscussion\b', text, re.IGNORECASE)),
            'has_conclusion': bool(re.search(r'\bconclusion\b', text, re.IGNORECASE)),
            'has_references': bool(re.search(r'\b(references|bibliography)\b', text, re.IGNORECASE)),
            'word_count': len(text.split()),
            'section_count': len(re.findall(r'\n[A-Z][a-z]+\n', text))
        }

        # Check for required sections
        if not analysis['has_abstract']:
            self.issues.append("Missing abstract section")
            self.suggestions.append("Add an abstract summarizing your work")

        if not analysis['has_methods']:
            self.issues.append("Missing methods/methodology section")
            self.suggestions.append("Add detailed methodology section for reproducibility")

        if not analysis['has_results']:
            self.issues.append("Missing results section")
            self.suggestions.append("Add results section with quantitative findings")

        if analysis['word_count'] < 2000:
            self.issues.append("Paper appears too short (< 2000 words)")
            self.suggestions.append("Expand paper with more detail, especially in methods and results")

        return analysis

    def _analyze_code(self, code_files: List[Path]) -> Dict[str, Any]:
        """Analyze submitted code files"""
        if not code_files:
            self.issues.append("No code files submitted")
            self.suggestions.append("Include code files for reproducibility")
            return {'code_provided': False}

        analysis = {
            'code_provided': True,
            'file_count': len(code_files),
            'languages_detected': set(),
            'has_requirements': False,
            'has_readme': False,
            'has_tests': False,
            'estimated_complexity': 'unknown',
            'code_quality_issues': []
        }

        for code_file in code_files:
            # Detect language
            ext = code_file.suffix.lower()
            if ext in ['.py', '.ipynb']:
                analysis['languages_detected'].add('Python')
            elif ext in ['.r', '.R']:
                analysis['languages_detected'].add('R')
            elif ext in ['.jl']:
                analysis['languages_detected'].add('Julia')

            # Check for key files
            if code_file.name.lower() in ['requirements.txt', 'environment.yml', 'setup.py']:
                analysis['has_requirements'] = True
            if 'readme' in code_file.name.lower():
                analysis['has_readme'] = True
            if 'test' in code_file.name.lower():
                analysis['has_tests'] = True

            # Basic code quality checks
            if code_file.suffix == '.py':
                try:
                    content = code_file.read_text()
                    if 'import' not in content:
                        analysis['code_quality_issues'].append(f"{code_file.name}: No imports found")
                    if 'def ' not in content and 'class ' not in content:
                        analysis['code_quality_issues'].append(f"{code_file.name}: No functions or classes defined")
                except Exception as e:
                    analysis['code_quality_issues'].append(f"Error reading {code_file.name}: {e}")

        # Provide suggestions
        if not analysis['has_requirements']:
            self.suggestions.append("Add requirements.txt or environment.yml for dependency management")

        if not analysis['has_readme']:
            self.suggestions.append("Add README.md with setup and usage instructions")

        if not analysis['has_tests']:
            self.suggestions.append("Add test files to validate code correctness")

        analysis['languages_detected'] = list(analysis['languages_detected'])
        return analysis

    def _analyze_data(self, data_files: List[Path]) -> Dict[str, Any]:
        """Analyze submitted data files"""
        if not data_files:
            self.issues.append("No data files submitted")
            self.suggestions.append("Include sample data or data documentation")
            return {'data_provided': False}

        analysis = {
            'data_provided': True,
            'file_count': len(data_files),
            'total_size_mb': sum(f.stat().st_size for f in data_files) / (1024 * 1024),
            'data_formats': set(),
            'has_documentation': False
        }

        for data_file in data_files:
            ext = data_file.suffix.lower()
            analysis['data_formats'].add(ext)

            if 'readme' in data_file.name.lower() or 'documentation' in data_file.name.lower():
                analysis['has_documentation'] = True

        if not analysis['has_documentation']:
            self.suggestions.append("Add data documentation explaining format and variables")

        if analysis['total_size_mb'] > 100:
            self.suggestions.append("Consider hosting large datasets externally (e.g., Zenodo, OSF)")

        analysis['data_formats'] = list(analysis['data_formats'])
        return analysis

    def _assess_reproducibility(
        self,
        paper_text: str,
        code_files: Optional[List[Path]],
        data_files: Optional[List[Path]]
    ) -> Dict[str, Any]:
        """Assess overall reproducibility"""
        assessment = {
            'reproducibility_score': 0,
            'level': 'poor',
            'barriers': []
        }

        score = 0

        # Code availability (+30 points)
        if code_files and len(code_files) > 0:
            score += 30
        else:
            assessment['barriers'].append("No code provided")

        # Data availability (+30 points)
        if data_files and len(data_files) > 0:
            score += 30
        else:
            assessment['barriers'].append("No data provided")

        # Documentation (+20 points)
        has_readme = any('readme' in f.name.lower() for f in (code_files or []))
        if has_readme:
            score += 20
        else:
            assessment['barriers'].append("Missing documentation")

        # Method details in paper (+20 points)
        if re.search(r'\b(algorithm|procedure|implementation)\b', paper_text, re.IGNORECASE):
            score += 20
        else:
            assessment['barriers'].append("Insufficient method details in paper")

        assessment['reproducibility_score'] = score

        # Determine level
        if score >= 80:
            assessment['level'] = 'excellent'
        elif score >= 60:
            assessment['level'] = 'good'
        elif score >= 40:
            assessment['level'] = 'moderate'
        else:
            assessment['level'] = 'poor'

        return assessment

    def _validate_methods(
        self,
        paper_text: str,
        code_files: Optional[List[Path]]
    ) -> Dict[str, Any]:
        """Validate that methods described match code"""
        validation = {
            'methods_described': False,
            'code_matches_description': 'unknown',
            'validation_issues': []
        }

        # Check if methods are described
        methods_section = re.search(
            r'(methods|methodology)[\s\S]{100,}',
            paper_text,
            re.IGNORECASE
        )

        if methods_section:
            validation['methods_described'] = True
            methods_text = methods_section.group()

            # Extract claimed methods/algorithms
            claimed_methods = re.findall(
                r'\b(neural network|random forest|SVM|regression|clustering|'
                r'deep learning|CNN|RNN|LSTM|transformer)\b',
                methods_text,
                re.IGNORECASE
            )

            if code_files:
                # Check if code implements claimed methods
                # (This is simplified - real implementation would parse code AST)
                for code_file in code_files:
                    if code_file.suffix == '.py':
                        try:
                            code_content = code_file.read_text().lower()
                            for method in claimed_methods:
                                if method.lower() not in code_content:
                                    validation['validation_issues'].append(
                                        f"Method '{method}' described but not found in code"
                                    )
                        except:
                            pass

                validation['code_matches_description'] = 'partial' if validation['validation_issues'] else 'yes'
            else:
                validation['code_matches_description'] = 'no_code_to_verify'
                self.inconsistencies.append("Methods described but no code provided for verification")
        else:
            validation['methods_described'] = False
            self.issues.append("No clear methods section found")

        return validation

    def _check_consistency(
        self,
        paper_text: str,
        code_files: Optional[List[Path]],
        data_files: Optional[List[Path]]
    ) -> Dict[str, Any]:
        """Check for inconsistencies between paper, code, and data"""
        consistency = {
            'consistent': True,
            'issues_found': []
        }

        # Check if paper claims code availability but none provided
        if re.search(r'(code|repository|github).*(available|provided)', paper_text, re.IGNORECASE):
            if not code_files:
                consistency['consistent'] = False
                consistency['issues_found'].append("Paper claims code is available but no code submitted")
                self.inconsistencies.append("Code availability claimed but not provided")

        # Check if paper claims data availability but none provided
        if re.search(r'(data|dataset).*(available|provided|shared)', paper_text, re.IGNORECASE):
            if not data_files:
                consistency['consistent'] = False
                consistency['issues_found'].append("Paper claims data is available but no data submitted")
                self.inconsistencies.append("Data availability claimed but not provided")

        # Check for result claims vs code
        results_mentioned = re.findall(r'(\d+\.?\d*)\s*%\s*(accuracy|precision|recall|f1)', paper_text, re.IGNORECASE)
        if results_mentioned and not code_files:
            self.inconsistencies.append("Quantitative results reported but no code to verify")

        return consistency

    def _calculate_quality_score(self, report: Dict[str, Any]) -> int:
        """Calculate overall quality score (0-100)"""
        score = 0

        # Paper structure (30 points)
        paper = report['paper_analysis']
        if paper['has_abstract']:
            score += 5
        if paper['has_methods']:
            score += 10
        if paper['has_results']:
            score += 10
        if paper['has_discussion']:
            score += 5

        # Reproducibility (40 points)
        repro = report['reproducibility_assessment']
        score += min(40, repro['reproducibility_score'] * 0.4)

        # Consistency (20 points)
        if report['consistency_check']['consistent']:
            score += 20
        else:
            score += max(0, 20 - len(report['consistency_check']['issues_found']) * 5)

        # Bonus for comprehensive submission (10 points)
        if report['code_analysis'] and report['code_analysis']['code_provided']:
            score += 5
        if report['data_analysis'] and report['data_analysis']['data_provided']:
            score += 5

        return min(100, int(score))


def analyze_submission(
    paper_path: Path,
    code_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Main entry point for analyzing a complete submission

    Args:
        paper_path: Path to paper text file
        code_dir: Optional directory containing code files
        data_dir: Optional directory containing data files

    Returns:
        Comprehensive analysis report
    """
    analyzer = EnhancedPaperAnalyzer()

    # Read paper
    paper_text = paper_path.read_text(encoding='utf-8')

    # Collect code files
    code_files = None
    if code_dir and code_dir.exists():
        code_files = list(code_dir.glob('**/*'))
        code_files = [f for f in code_files if f.is_file()]

    # Collect data files
    data_files = None
    if data_dir and data_dir.exists():
        data_files = list(data_dir.glob('**/*'))
        data_files = [f for f in data_files if f.is_file()]

    # Run comprehensive analysis
    return analyzer.analyze_paper_comprehensive(
        paper_text=paper_text,
        code_files=code_files,
        data_files=data_files
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_paper_analyzer.py <paper_file> [code_dir] [data_dir]")
        sys.exit(1)

    paper_path = Path(sys.argv[1])
    code_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    data_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    report = analyze_submission(paper_path, code_dir, data_dir)

    print("="*70)
    print("SYMBYAI COMPREHENSIVE ANALYSIS REPORT")
    print("="*70)
    print(json.dumps(report, indent=2))
