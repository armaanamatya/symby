# Logging Guide

All demos now automatically log their output to timestamped files in the `logs/` directory.

## How It Works

### Automatic Logging (Built-in)

Both demo scripts now automatically log output:

```bash
# Run demo - automatically logs to logs/demo_fixed_vectorstore_YYYYMMDD_HHMMSS.log
python demo_fixed_with_vectorstore.py

# Run original demo - automatically logs to logs/demo_nvidia_rag_YYYYMMDD_HHMMSS.log
python demo_nvidia_rag.py
```

**What gets logged:**
- ✅ All console output (everything you see)
- ✅ Timestamps for when the run started/ended
- ✅ API responses
- ✅ Error messages and tracebacks
- ✅ User interactions

### Manual Logging (Any Script)

Use `run_with_logging.py` to log ANY script:

```bash
# Run any script with logging
python run_with_logging.py <script_name>

# Examples:
python run_with_logging.py test_nvidia_integration.py
python run_with_logging.py check_api_key.py
python run_with_logging.py your_custom_script.py
```

## Log File Format

### Log Filename

```
logs/<script_name>_<timestamp>.log

Examples:
logs/demo_fixed_vectorstore_20231214_143052.log
logs/test_nvidia_integration_20231214_144521.log
```

### Log Content

```
======================================================================
Script: demo_fixed_with_vectorstore.py
Started: 2023-12-14T14:30:52.123456
Command: python demo_fixed_with_vectorstore.py
======================================================================

[All output from the script...]

======================================================================
Completed: 2023-12-14T14:35:12.654321
Exit code: 0
======================================================================
```

## Viewing Logs

### List all logs
```bash
ls -lht logs/
# Shows logs sorted by time (most recent first)
```

### View latest log
```bash
# For fixed demo
cat logs/demo_fixed_vectorstore_*.log | tail -1000

# For original demo
cat logs/demo_nvidia_rag_*.log | tail -1000
```

### Search logs
```bash
# Find errors
grep -r "Error" logs/

# Find specific API responses
grep -r "AlphaFold" logs/

# Find retrieval scores
grep -r "Relevance Score" logs/
```

### View specific log
```bash
# Use tab completion to find the file
cat logs/demo_fixed_vectorstore_20231214_143052.log

# Or use less for scrolling
less logs/demo_fixed_vectorstore_20231214_143052.log
```

## Log Examples

### Example 1: Successful Run

```
logs/demo_fixed_vectorstore_20231214_143052.log

======================================================================
Log started: 2023-12-14T14:30:52.123456
Log file: logs/demo_fixed_vectorstore_20231214_143052.log
======================================================================

======================================================================
FIXED NVIDIA RAG - WITH VECTOR STORE INTEGRATION
======================================================================

📚 Retrieved 3 sources:
  [1] Highly accurate protein structure prediction (score: 0.950)
  [2] Accelerating drug discovery (score: 0.881)
  [3] Understanding SARS-CoV-2 proteins (score: 0.850)

======================================================================
ANSWER:
======================================================================
AlphaFold accelerated drug discovery by...

======================================================================
Log ended: 2023-12-14T14:35:12.654321
======================================================================
```

### Example 2: Error Log

```
logs/demo_nvidia_rag_20231214_151234.log

======================================================================
DEMO 2: Knowledge Graph Extraction (txt2kg pattern)
======================================================================

[KG Extraction] Failed: Expecting property name enclosed in double quotes: line 2 column 1 (char 2)

✅ Extracted 0 knowledge triples:
```

This shows the JSON parsing error we fixed!

## Using Logs for Debugging

### Find when errors occur
```bash
# Get line numbers of errors
grep -n "Failed" logs/demo_nvidia_rag_*.log
```

### Compare runs
```bash
# Compare before and after fixes
diff logs/demo_nvidia_rag_20231214_143052.log \
     logs/demo_fixed_vectorstore_20231214_151234.log
```

### Extract metrics
```bash
# Get all retrieval scores
grep "Relevance Score:" logs/*.log

# Count successful extractions
grep "Extracted.*triples" logs/*.log
```

## Programmatic Logging (In Your Code)

### Use in your own scripts

```python
from src.utils.logger import setup_logging

# Method 1: Context manager (recommended)
with setup_logging("my_script"):
    print("This goes to console AND log file")
    # Your code here

# Method 2: With timestamps
with setup_logging("my_script", timestamped=True):
    print("This line gets a timestamp prefix")

# Method 3: Manual control
logger = setup_logging("my_script")
logger.__enter__()  # Start logging
print("Logged output")
logger.__exit__(None, None, None)  # Stop logging
```

### Log specific sections

```python
# Log only critical sections
print("This is NOT logged")

with setup_logging("critical_section"):
    print("This IS logged")
    # Critical code here

print("Back to normal, NOT logged")
```

## Log Rotation

Logs accumulate over time. To manage:

### Clean old logs (manual)
```bash
# Delete logs older than 7 days
find logs/ -name "*.log" -mtime +7 -delete

# Delete all logs
rm logs/*.log
```

### Keep only recent logs
```bash
# Keep only last 10 logs
ls -t logs/*.log | tail -n +11 | xargs rm -f
```

### Archive logs
```bash
# Compress old logs
gzip logs/*.log

# Create dated archive
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/*.log
rm logs/*.log
```

## Tips

### 1. Always check logs after errors
```bash
# View latest log
ls -t logs/ | head -1 | xargs -I {} cat logs/{}
```

### 2. Keep logs during hackathon
Don't delete logs during the hackathon - they're useful for:
- Debugging issues
- Showing judges what you tried
- Reproducing results

### 3. Share logs with team
```bash
# Share specific log
cat logs/demo_fixed_vectorstore_20231214_143052.log | pbcopy  # macOS
cat logs/demo_fixed_vectorstore_20231214_143052.log | xclip  # Linux
```

### 4. Compare performance
```bash
# Time from logs
grep "Completed:" logs/*.log
```

## Log Location

```
symby/
├── logs/                                      ← All logs here
│   ├── demo_fixed_vectorstore_TIMESTAMP.log
│   ├── demo_nvidia_rag_TIMESTAMP.log
│   ├── test_nvidia_integration_TIMESTAMP.log
│   └── ...
├── demo_fixed_with_vectorstore.py            ← Auto-logs on run
├── demo_nvidia_rag.py                         ← Auto-logs on run
├── run_with_logging.py                        ← Manual logger
└── src/utils/logger.py                        ← Logging utility
```

## Troubleshooting

### Logs directory missing
```bash
mkdir -p logs
```

### Permission denied
```bash
chmod 755 logs/
```

### Log file too large
```bash
# View last 100 lines
tail -100 logs/demo_fixed_vectorstore_*.log

# Compress large log
gzip logs/demo_fixed_vectorstore_20231214_143052.log
```

### Can't find log
```bash
# List all logs with timestamps
ls -lht logs/

# Search by date
ls logs/*20231214*.log
```

## Summary

✅ **Automatic logging** in both demo scripts
✅ **Manual logging** with `run_with_logging.py`
✅ **Timestamped filenames** for easy tracking
✅ **Full output capture** (stdout + stderr)
✅ **Easy to search** and analyze

**Start logging:**
```bash
python demo_fixed_with_vectorstore.py  # Auto-logs
python run_with_logging.py your_script.py  # Manual
```

**View logs:**
```bash
ls -lht logs/  # List all logs
cat logs/demo_fixed_vectorstore_*.log  # View log
```

Done! 🎉
