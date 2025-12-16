"""
Generate synthetic S2ORC-format sample data for pipeline testing.

Creates ~500 papers across 3 fields (CS, Biology, Medicine) with:
- Realistic paper structure (title, abstract, sections)
- ML-related content to test influence tracing
- Temporal spread (2018-2023)
"""

import json
import gzip
import random
from pathlib import Path
from datetime import datetime, timedelta

# Sample paper templates by field
CS_PAPERS = [
    {
        "title": "Transformer Architecture for Natural Language Understanding",
        "abstract": "We propose a novel transformer-based architecture that achieves state-of-the-art results on multiple NLP benchmarks. Our approach uses multi-head self-attention mechanisms to capture long-range dependencies in text.",
        "keywords": ["transformer", "attention", "NLP", "deep learning", "BERT"],
    },
    {
        "title": "Deep Reinforcement Learning for Game Playing",
        "abstract": "This paper presents a deep reinforcement learning agent that masters complex games through self-play. We combine convolutional neural networks with Monte Carlo tree search.",
        "keywords": ["reinforcement learning", "deep learning", "neural network", "game AI"],
    },
    {
        "title": "Graph Neural Networks for Social Network Analysis",
        "abstract": "We introduce a graph neural network framework for analyzing large-scale social networks. Our method captures both local and global graph structures through message passing.",
        "keywords": ["graph neural network", "GNN", "social network", "deep learning"],
    },
    {
        "title": "Federated Learning for Privacy-Preserving Machine Learning",
        "abstract": "We present a federated learning approach that enables collaborative model training without sharing raw data. Our method ensures differential privacy guarantees.",
        "keywords": ["federated learning", "privacy", "distributed learning", "machine learning"],
    },
    {
        "title": "Vision Transformers for Image Classification",
        "abstract": "We apply the transformer architecture to image classification tasks, demonstrating that attention mechanisms can replace convolutional operations entirely.",
        "keywords": ["vision transformer", "ViT", "image classification", "attention", "computer vision"],
    },
    {
        "title": "Large Language Models and Emergent Abilities",
        "abstract": "We study the scaling behavior of large language models and identify emergent abilities that appear only at sufficient scale. Our findings have implications for AI safety.",
        "keywords": ["large language model", "LLM", "GPT", "scaling laws", "emergent behavior"],
    },
    {
        "title": "Neural Architecture Search with Reinforcement Learning",
        "abstract": "We propose an automated method for designing neural network architectures using reinforcement learning. Our approach discovers novel architectures that outperform hand-designed networks.",
        "keywords": ["neural architecture search", "NAS", "AutoML", "reinforcement learning"],
    },
    {
        "title": "Diffusion Models for Image Generation",
        "abstract": "We present a diffusion-based generative model that produces high-quality images. Our method iteratively denoises samples from a Gaussian distribution.",
        "keywords": ["diffusion model", "generative model", "image synthesis", "deep learning"],
    },
]

BIOLOGY_PAPERS = [
    {
        "title": "AlphaFold: Protein Structure Prediction with Deep Learning",
        "abstract": "We present a deep learning system for predicting protein 3D structures from amino acid sequences with unprecedented accuracy. Our method uses attention mechanisms to model residue interactions.",
        "keywords": ["protein structure", "AlphaFold", "deep learning", "structural biology"],
    },
    {
        "title": "Single-Cell RNA Sequencing Analysis with Neural Networks",
        "abstract": "We develop a neural network approach for analyzing single-cell RNA sequencing data. Our method identifies cell types and trajectories in complex tissues.",
        "keywords": ["single-cell", "RNA-seq", "neural network", "cell type", "genomics"],
    },
    {
        "title": "Machine Learning for Drug Discovery",
        "abstract": "We apply graph neural networks to predict molecular properties relevant to drug discovery. Our model accelerates the identification of promising drug candidates.",
        "keywords": ["drug discovery", "machine learning", "molecular properties", "GNN", "pharmaceuticals"],
    },
    {
        "title": "Deep Learning for Genomic Variant Calling",
        "abstract": "We present a convolutional neural network for identifying genetic variants from sequencing data. Our method achieves higher accuracy than traditional alignment-based approaches.",
        "keywords": ["genomics", "variant calling", "deep learning", "CNN", "sequencing"],
    },
    {
        "title": "Attention-Based Models for Gene Expression Prediction",
        "abstract": "We use transformer architectures to predict gene expression levels from DNA sequences. Our model captures long-range regulatory interactions.",
        "keywords": ["gene expression", "transformer", "attention", "regulatory genomics"],
    },
    {
        "title": "Reinforcement Learning for Protein Design",
        "abstract": "We apply reinforcement learning to design novel proteins with desired functions. Our agent learns to navigate the sequence space efficiently.",
        "keywords": ["protein design", "reinforcement learning", "synthetic biology", "protein engineering"],
    },
]

MEDICINE_PAPERS = [
    {
        "title": "Deep Learning for Medical Image Diagnosis",
        "abstract": "We develop a convolutional neural network for automated diagnosis from medical images. Our system achieves radiologist-level performance on chest X-ray interpretation.",
        "keywords": ["medical imaging", "diagnosis", "CNN", "radiology", "deep learning"],
    },
    {
        "title": "Natural Language Processing for Clinical Notes",
        "abstract": "We apply transformer models to extract clinical information from electronic health records. Our method identifies diagnoses, medications, and outcomes from unstructured text.",
        "keywords": ["clinical NLP", "EHR", "transformer", "medical informatics", "health records"],
    },
    {
        "title": "Machine Learning for Predicting Patient Outcomes",
        "abstract": "We develop a gradient boosting model for predicting patient outcomes in intensive care units. Our method incorporates temporal patterns in vital signs.",
        "keywords": ["patient outcomes", "ICU", "machine learning", "clinical prediction", "healthcare"],
    },
    {
        "title": "Graph Neural Networks for Drug-Drug Interaction Prediction",
        "abstract": "We use graph neural networks to predict adverse drug-drug interactions. Our model learns from molecular structures and known interaction databases.",
        "keywords": ["drug interaction", "GNN", "pharmacology", "adverse events", "drug safety"],
    },
    {
        "title": "Federated Learning for Multi-Site Clinical Studies",
        "abstract": "We apply federated learning to train models across multiple hospital sites without sharing patient data. Our approach maintains privacy while improving model generalization.",
        "keywords": ["federated learning", "clinical study", "privacy", "multi-site", "healthcare AI"],
    },
    {
        "title": "Attention Mechanisms for ECG Classification",
        "abstract": "We develop an attention-based model for classifying cardiac arrhythmias from ECG signals. Our method highlights the most diagnostically relevant signal segments.",
        "keywords": ["ECG", "arrhythmia", "attention", "cardiac", "deep learning"],
    },
]


def generate_paper_text(template: dict, field: str) -> str:
    """Generate full paper text from template."""
    sections = [
        f"{template['title']}\n",
        f"\nAbstract\n{template['abstract']}\n",
        f"\nIntroduction\nThis paper addresses important challenges in {field}. "
        f"Recent advances in machine learning, particularly deep learning and neural networks, "
        f"have opened new possibilities for solving complex problems. "
        f"We build upon prior work in {', '.join(template['keywords'][:3])} to develop our approach.\n",
        f"\nMethods\nWe employ a novel methodology combining {template['keywords'][0]} with "
        f"advanced techniques from {template['keywords'][1]}. Our approach involves training "
        f"neural networks on large-scale datasets and evaluating performance on standard benchmarks.\n",
        f"\nResults\nOur experiments demonstrate significant improvements over baseline methods. "
        f"We achieve state-of-the-art results on multiple evaluation metrics, validating the "
        f"effectiveness of our {template['keywords'][0]}-based approach.\n",
        f"\nDiscussion\nThe success of our method highlights the potential of {template['keywords'][0]} "
        f"for {field} applications. Future work could extend this to related domains and "
        f"explore connections with {template['keywords'][-1]}.\n",
        f"\nConclusion\nWe have presented a new approach to {template['title'].lower()} that "
        f"achieves strong results. Our findings contribute to the growing body of work on "
        f"applying machine learning to {field}.\n",
    ]
    return "".join(sections)


def generate_paper(paper_id: int, template: dict, field: str, year: int) -> dict:
    """Generate a single paper in S2ORC format."""
    created_date = datetime(year, random.randint(1, 12), random.randint(1, 28))

    return {
        "id": str(paper_id),
        "added": created_date.isoformat() + "Z",
        "created": created_date.strftime("%Y-%m-%dT00:00:00.000Z"),
        "text": generate_paper_text(template, field),
        "source": "s2",
        "version": "v3-fos",
        "metadata": {
            "year": year,
            "s2fieldsofstudy": [field],
            "extfieldsofstudy": [field, "Machine Learning"],
            "sha1": f"sample_{paper_id:08d}",
            "provenance": f"sample-{year}.gz:{paper_id}",
        },
        "attributes": {
            "bff_duplicate_paragraph_spans_decontamination": [],
        },
        "bff_contained_ngram_count": 0,
        "bff_duplicate_spans": [],
    }


def generate_dataset(output_dir: str, papers_per_field: int = 50):
    """Generate complete sample dataset."""
    output_dir = Path(output_dir)

    fields = {
        "CS,2020-2022": CS_PAPERS,
        "Biology,2020-2022": BIOLOGY_PAPERS,
        "Medicine,2020-2022": MEDICINE_PAPERS,
    }

    paper_id = 100000  # Starting ID
    total_papers = 0

    for field_dir, templates in fields.items():
        field_name = field_dir.split(",")[0]
        field_path = output_dir / field_dir / "train"
        field_path.mkdir(parents=True, exist_ok=True)

        # Generate papers
        papers = []
        for i in range(papers_per_field):
            template = templates[i % len(templates)]
            year = random.choice([2018, 2019, 2020, 2021, 2022, 2023])
            paper = generate_paper(paper_id, template, field_name, year)
            papers.append(paper)
            paper_id += 1

        # Write to gzipped JSONL file
        output_file = field_path / "0000.json.gz"
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper) + '\n')

        total_papers += len(papers)
        print(f"Generated {len(papers)} papers for {field_name} -> {output_file}")

    print(f"\nTotal: {total_papers} sample papers generated")
    print(f"Dataset location: {output_dir}")

    return total_papers


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "s2orc_data"
    papers_per_field = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    generate_dataset(output_dir, papers_per_field)
