# ML Papers Evaluation Results

## Summary

Successfully evaluated **all 3,739 ML papers** and saved each as individual JSON file.

### Output Location
**Directory:** `/home/asus/symby/ml_evaluations/`
- **Total files:** 3,740 (3,739 papers + 1 summary)
- **Total size:** 15 MB
- **Format:** Individual JSON files named by paper ID

### Processing Statistics

- ✅ **Successfully processed:** 3,739 papers (100%)
- ❌ **Errors:** 0
- ⚡ **Processing speed:** ~95 papers/second
- ⏱️ **Total time:** ~39 seconds

## Results Breakdown

### ML Adoption Levels
| Level | Count | Percentage |
|-------|-------|------------|
| **Moderate** | 1,386 | 37.1% |
| **Minimal** | 1,115 | 29.8% |
| **Substantial** | 662 | 17.7% |
| **None** | 458 | 12.2% |
| **Core** | 118 | 3.2% |

### Reproducibility Metrics
- **Code Available:** 441 papers (11.8%)
- **Data Available:** 716 papers (19.1%)

### Impact Indicators
- **Novelty Claims:** 2,329 papers (62.3%)

## File Structure

Each paper is saved as `{paper_id}.json` with complete evaluation structure:

```json
{
  "paper_id": "11064586",
  "field": "Medicine",
  "publication_date": "2007",
  "ml_adoption": {
    "ml_frameworks_mentioned": [],
    "specific_models_architectures": [],
    "ml_adoption_level": "none",
    ...
  },
  "reproducibility": {
    "code_availability_mentioned": false,
    "methodology_detail_level": "moderate",
    "replication_feasibility": "difficult",
    ...
  },
  "research_outcomes": {
    "mentions_clinical_trials": true,
    "validation_type": ["statistical validation", "clinical validation"],
    ...
  },
  "impact_indicators": {
    "claims_novelty": true,
    "potential_impact_scope": "moderate",
    ...
  },
  "overall_assessment": "Traditional research",
  "confidence_score": "high"
}
```

## Special Files

- **`_summary.json`** - Aggregated statistics for all papers
- **`_errors.json`** - Error log (empty - no errors!)

## Key Insights

1. **Low True ML Adoption:** Only 20.9% have substantial/core ML (780 papers)
   - This suggests the "ML" dataset includes papers that mention ML but don't deeply use it

2. **Low Reproducibility:** Only 11.8% mention code availability
   - Opportunity for improvement in research transparency

3. **High Novelty Claims:** 62.3% claim novelty
   - Common in research papers, shows ambition

4. **Data More Available Than Code:** 19.1% vs 11.8%
   - Data sharing is more common than code sharing

## Usage Examples

### Load a single paper
```python
import json

with open('ml_evaluations/11064586.json', 'r') as f:
    paper = json.load(f)

print(f"ML Level: {paper['ml_adoption']['ml_adoption_level']}")
print(f"Code Available: {paper['reproducibility']['code_availability_mentioned']}")
```

### Load all papers
```python
import json
from pathlib import Path

papers = []
for file in Path('ml_evaluations').glob('*.json'):
    if file.name.startswith('_'):
        continue  # Skip summary files
    with open(file) as f:
        papers.append(json.load(f))

print(f"Loaded {len(papers)} papers")
```

### Filter papers by criteria
```python
# Find papers with code available
papers_with_code = [
    p for p in papers
    if p['reproducibility']['code_availability_mentioned']
]

print(f"Papers with code: {len(papers_with_code)}")
```

### Analyze by field
```python
from collections import defaultdict

by_field = defaultdict(list)
for paper in papers:
    by_field[paper['field']].append(paper)

for field, field_papers in sorted(by_field.items()):
    ml_count = sum(1 for p in field_papers
                   if p['ml_adoption']['ml_adoption_level'] != 'none')
    ml_rate = ml_count / len(field_papers) * 100
    print(f"{field}: {len(field_papers)} papers, {ml_rate:.1f}% ML")
```

## Next Steps

1. **Evaluate non-ML dataset** (25,882 papers)
   ```bash
   python evaluate_ml_papers.py  # Modify for non-ML
   ```

2. **Build visualizations** using this data in your Streamlit app

3. **Analyze patterns:**
   - Which fields have highest ML adoption?
   - Does ML adoption correlate with reproducibility?
   - What frameworks are most popular?

4. **Compare ML vs non-ML papers** for hackathon objective #3

## Files Generated

Run the evaluation script:
```bash
python evaluate_ml_papers.py
```

This created:
- 3,739 individual JSON files in `ml_evaluations/`
- Summary statistics in `ml_evaluations/_summary.json`
- Total of 15 MB data, fully indexed by paper ID
