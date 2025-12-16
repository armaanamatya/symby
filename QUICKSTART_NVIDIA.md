# Quick Start: Add NVIDIA NIM to Your Existing Pipeline

## Overview

You already have a solid foundation:
- ✅ BGE-M3 embeddings
- ✅ FAISS vector index
- ✅ Hybrid search (dense + sparse + graph)

**This guide adds NVIDIA NIM in 3 steps (< 30 minutes).**

---

## Step 1: Get NVIDIA API Key (5 minutes)

1. Go to **https://build.nvidia.com/explore/discover**
2. Sign in with your account
3. Select **Llama 3.1 70B** or **Mixtral 8x22B**
4. Click **"Get API Key"**
5. Copy your key

```bash
# Add to your environment
export NVIDIA_API_KEY="nvapi-xxxxx"

# Or add to .env file
echo "NVIDIA_API_KEY=nvapi-xxxxx" >> .env
```

---

## Step 2: Install Dependencies (2 minutes)

```bash
# Install OpenAI SDK (NVIDIA NIM is OpenAI-compatible)
pip install openai

# Optional: For structured outputs
pip install pydantic
```

---

## Step 3: Add NIM to Your Pipeline (10 minutes)

### Option A: Minimal Integration (Recommended for Hackathon)

Add this to your existing `app.py`:

```python
from openai import OpenAI
import os

# Initialize NVIDIA NIM client
nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ['NVIDIA_API_KEY']
)

def analyze_ml_impact_with_nim(query: str, search_results):
    """Enhance your search results with NIM reasoning."""

    # Build context from your search results
    context = "\n\n".join([
        f"Paper: {r.title}\nYear: {r.year}\nField: {r.source_field}\nAbstract: {r.metadata.get('abstract', '')}"
        for r in search_results[:5]
    ])

    # Call NIM for reasoning
    response = nim_client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are an expert analyzing ML's impact on scientific research. Provide quantitative insights with citations."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide attribution scores and evidence."
            }
        ],
        temperature=0.2,
        max_tokens=1500
    )

    return response.choices[0].message.content
```

**Usage in your Streamlit app:**

```python
# In your app.py, after search results

if st.button("Analyze with AI"):
    with st.spinner("Analyzing ML impact..."):
        analysis = analyze_ml_impact_with_nim(
            query=search_query,
            search_results=results
        )
        st.markdown("### AI Analysis")
        st.write(analysis)
```

### Option B: Full Integration

Use the complete `nvidia_nim_integration.py` we created:

```python
from src.rag.nvidia_nim_integration import EnhancedRAGPipeline, NVIDIANIMClient

# In your app initialization
enhanced_rag = EnhancedRAGPipeline(
    hybrid_search_engine=search_engine,  # Your existing engine
    citation_graph=citation_graph,  # Optional
    nvidia_api_key=os.environ['NVIDIA_API_KEY']
)

# In your query handler
results = enhanced_rag.query(
    query=user_query,
    k=10,
    analyze_attribution=True  # Enable attribution analysis
)

st.write(results['answer'])
```

---

## Step 4: Test It! (5 minutes)

```python
# test_nim.py
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ['NVIDIA_API_KEY']
)

response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain how AlphaFold used deep learning to predict protein structures."
        }
    ],
    temperature=0.3,
    max_tokens=500
)

print(response.choices[0].message.content)
```

```bash
python test_nim.py
```

**Expected output:**
```
AlphaFold leverages deep learning through several key innovations:

1. Attention-based architecture: Uses transformer-like attention mechanisms to model
   relationships between amino acid residues...

2. End-to-end prediction: Directly predicts 3D coordinates from sequence data...

[continues...]
```

---

## Use Cases for Your Hackathon Project

### 1. Attribution Scoring Dashboard

```python
# Add to your Streamlit sidebar
st.sidebar.header("ML Attribution Analysis")

if st.sidebar.button("Analyze Top Papers"):
    for paper in top_papers[:10]:
        attribution = nim_client.analyze_attribution(
            paper=paper,
            cited_papers=related_papers
        )

        st.metric(
            label=paper['title'][:50],
            value=f"{attribution.attribution_score:.2f}",
            delta=f"{len(attribution.ml_methods)} ML methods"
        )
```

### 2. Acceleration Metrics Visualization

```python
# Compute acceleration for breakthrough papers
breakthroughs = papers[papers['citation_count'] > 100]

accelerations = []
for paper in breakthroughs:
    baseline_papers = find_historical_papers(paper)
    acceleration = nim_client.analyze_acceleration(
        paper=paper,
        baseline_papers=baseline_papers
    )
    accelerations.append(acceleration)

# Visualize speedup factors
fig = px.bar(
    x=[a.paper_id for a in accelerations],
    y=[a.speedup_factor for a in accelerations],
    title="Discovery Acceleration by ML Adoption"
)
st.plotly_chart(fig)
```

### 3. Interactive Q&A

```python
# Add chatbot interface
st.header("Ask About ML Impact")

query = st.text_input("Your question:")

if query:
    # Use your hybrid search
    search_results = search_engine.search(query, k=10)

    # Enhance with NIM
    answer = nim_client.answer_query_with_rag(
        query=query,
        retrieved_papers=[r.metadata for r in search_results]
    )

    st.markdown(answer)

    # Show sources
    with st.expander("Sources"):
        for r in search_results[:5]:
            st.write(f"- {r.title} ({r.year})")
```

### 4. Batch Classification for Visualizations

```python
# Classify all papers by ML adoption level
import pandas as pd

papers_df = pd.read_parquet("data/papers.parquet")

# Batch classify with Mixtral (fast)
classifications = nim_client.classify_ml_adoption(
    papers=papers_df.to_dict('records'),
    model="mistralai/mixtral-8x22b-instruct-v0.1"
)

# Add to dataframe
papers_df['ml_adoption_level'] = [c['adoption_level'] for c in classifications]

# Visualize adoption trends
adoption_over_time = papers_df.groupby(['year', 'ml_adoption_level']).size().reset_index()

fig = px.line(
    adoption_over_time,
    x='year',
    y=0,
    color='ml_adoption_level',
    title="ML Adoption Trends (2016-2024)"
)
st.plotly_chart(fig)
```

---

## Cost Estimates

### Free Tier (Hackathon)
- **1000 requests/day** with free API key
- Enough for:
  - 100 attribution analyses
  - 500 classifications
  - 400 Q&A queries

### If You Need More (Production)
- **Llama 3.1 70B**: $0.40 per 1M tokens
- **Mixtral 8x22B**: $0.60 per 1M tokens

**Example costs for 81K papers:**
- Classify all papers: ~$20 (one-time)
- Attribution for top 1000: ~$50 (one-time)
- Q&A queries: ~$0.001 per query

---

## Troubleshooting

### Error: "Invalid API key"
```bash
# Check your key is set
echo $NVIDIA_API_KEY

# Should start with "nvapi-"
```

### Error: "Rate limit exceeded"
```python
# Add retry logic
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(min=1, max=10))
def call_nim_with_retry(...):
    return nim_client.chat.completions.create(...)
```

### Error: "Model not found"
```python
# List available models
models = client.models.list()
for model in models:
    print(model.id)

# Common models:
# - meta/llama-3.1-70b-instruct
# - meta/llama-3.1-405b-instruct
# - mistralai/mixtral-8x22b-instruct-v0.1
# - nvidia/llama-3.1-nemotron-70b-instruct
```

---

## Next Steps After Hackathon

### 1. Add RAPIDS for Analytics (Week 1)

```bash
# Install RAPIDS
conda install -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.10 cuda-version=12.0

# Convert data to Parquet
python scripts/convert_to_parquet.py

# Use cuDF instead of pandas
import cudf
papers_gdf = cudf.read_parquet("data/papers.parquet")
```

### 2. Migrate to Milvus (Week 1-2)

```bash
# Install Milvus
docker-compose up -d milvus-standalone

# Index with GPU
python scripts/migrate_to_milvus.py
```

### 3. Add cuGraph for Citation Analysis (Week 2)

```python
import cugraph
import cudf

# Build citation graph
edges = cudf.DataFrame({
    'src': citing_papers,
    'dst': cited_papers
})

G = cugraph.Graph()
G.from_cudf_edgelist(edges, source='src', destination='dst')

# PageRank
pagerank = cugraph.pagerank(G)
```

---

## Summary

**Minimal integration (today):**
```python
# 5 lines to add NIM
from openai import OpenAI

nim = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ['NVIDIA_API_KEY']
)

# Use with your existing search results
response = nim.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": query}]
)
```

**That's it! You now have:**
- ✅ Hybrid semantic search (your existing code)
- ✅ LLM reasoning (NVIDIA NIM)
- ✅ Ready for hackathon demo

**Total setup time: < 30 minutes**

---

## Resources

- **NVIDIA Build**: https://build.nvidia.com
- **API Docs**: https://docs.nvidia.com/nim
- **Model Catalog**: https://build.nvidia.com/explore/discover
- **Example Notebooks**: https://github.com/NVIDIA/GenerativeAIExamples

---

## Questions?

**Q: Do I need a GPU to use NIM?**
A: No! NIM runs on NVIDIA's cloud GPUs. You just make API calls.

**Q: Can I use NIM locally?**
A: Yes, with NVIDIA NIM containers (requires local GPU with 80GB+ VRAM).

**Q: What's the best model for my use case?**
A: Llama 3.1 70B for deep analysis, Mixtral for fast classification.

**Q: How do I cite papers in responses?**
A: Include "Always cite using [Paper ID: xxx]" in your system prompt.

**Q: Can I fine-tune models?**
A: Not directly on NIM, but you can use NVIDIA NeMo for custom training.
