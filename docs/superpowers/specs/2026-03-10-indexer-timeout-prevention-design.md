# Indexer Timeout Prevention Design

**Date:** 2026-03-10
**Status:** Approved
**Problem:** Full repository ingestion times out on GitHub Actions free runners (6-hour limit) for medium-sized repos (500-2000 files).

## Goals

1. Speed up embedding computation without sacrificing model quality
2. Enable interrupted runs to resume without re-processing
3. Gracefully handle timeout situations

## Non-Goals

- Switching to a different embedding model (Qwen3-Embedding-0.6B benchmarks significantly better than alternatives for code retrieval)
- Parallel processing across multiple runners (future enhancement)

## Design

### 1. ONNX Optimization

Use ONNX Runtime backend for 2-3x faster CPU inference.

**Changes to `EmbeddingCalculator.py`:**
- Detect CPU device and use `backend="onnx"` when loading SentenceTransformer
- Fall back to PyTorch if ONNX loading fails
- SentenceTransformers 3.0+ has native ONNX support

**Dependencies:**
- Add `onnxruntime>=1.17.0` to requirements.txt

### 2. Checkpointing via Database

Use the existing database as the checkpoint mechanism.

**On startup (full index mode):**
1. Query: `SELECT DISTINCT path FROM chunks WHERE repo = ? AND branch = ?`
2. Build set of already-indexed paths
3. Filter: `to_process = all_files - already_indexed`

**Behavior:**
- No separate state file needed
- Works across runners (DB is remote)
- Idempotent: re-running skips completed files
- Accepts possibility of duplicate chunks from partial batch writes (simple, negligible impact)

### 3. Dynamic max_seq_length

Reduce wasted padding computation for batches with short chunks.

**Changes to `EmbeddingCalculator.py`:**
- Before encoding, calculate max token length in batch
- Round up to nearest power of 2 (minimum overhead)
- Cap at configured `MAX_SEQ_LENGTH`
- Toggleable via `DYNAMIC_SEQ_LENGTH` env var (default: true)

**Trade-off:** Small tokenization overhead, but faster encoding for short chunks.

### 4. Soft Timeout Detection

Exit gracefully before hard timeout, enabling retry.

**Changes to `Indexer.py`:**
- Track elapsed time from start
- Check against `SOFT_TIMEOUT_SECONDS` env var before each file
- On timeout: flush pending work, output summary with `"timeout": true`, exit with code 75

**GitHub Actions workflow changes:**
- Set `SOFT_TIMEOUT: 19800` (5.5 hours)
- Add retry step triggered on exit code 75
- Retry dispatches same workflow with `full_scan: true`

**Flow:**
1. First run: processes ~5.5 hours worth, exits 75, triggers retry
2. Subsequent runs: skip already-indexed files, continue
3. Final run: completes, exits 0

## Files Modified

| File | Changes |
|------|---------|
| `packages/core/src/constants.py` | Add `DYNAMIC_SEQ_LENGTH`, `SOFT_TIMEOUT_SECONDS`, `EXIT_CODE_TIMEOUT` |
| `packages/core/src/Calculators/EmbeddingCalculator.py` | ONNX backend loading, dynamic max_seq_length |
| `packages/core/src/Indexer.py` | Query existing paths on startup, soft timeout check |
| `requirements.txt` | Add `onnxruntime>=1.17.0` |
| `.github/workflows/index.yml` | Add `SOFT_TIMEOUT` env, retry logic |

## Testing

- `test_embeddings.py`: Test ONNX fallback behavior
- `test_indexer_cli.py`: Test soft timeout exit code
- Existing persistence tests unchanged (checkpointing uses existing queries)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| ONNX conversion issues with Qwen3 model | Fallback to PyTorch with warning log |
| Dynamic seq_length less effective with ONNX | Make toggleable, test empirically |
| Duplicate chunks from partial batches | Acceptable trade-off; minor retrieval impact |
| Workflow retry loops forever | Cap retries in workflow (e.g., 5 max) |
