# Changelog

All notable changes to Forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.5] - 2026-05-15

### Testing
- Added comprehensive test suite for custom exceptions (`test_exceptions.py`)
- Added configuration validation tests (`test_config.py`) with Pydantic field validation
- Added retry decorator tests (`test_retry.py`) covering exponential backoff
- Added summariser tests (`test_summariser.py`) for checkpoint management
- Added edge case tests for VectorStore, ProjectGraph, Worker, FeedbackLoop
- Added plan parser edge case tests (invalid JSON, malformed structure, unicode)
- Fixed test coverage gaps across all core modules

### Bug Fixes
- Removed broken dependency test that was added incorrectly

## [0.2.4] - 2025-05-15

### New Features
- Query caching for faster repeated searches (LRU cache, 128 entries)
- Parallel file indexing for large files (>1KB) using ThreadPoolExecutor
- HTTP connection pooling for reduced latency in Brain and Worker

### Reliability
- Custom exception types for better error handling (`ForgeError`, `BrainError`, `WorkerError`, etc.)
- Retry decorator with exponential backoff for transient failures
- Graceful degradation when optional components fail
- Improved node existence checks in project graph

### Configuration
- Pydantic-based configuration validation (`src/forge/config.py`)
- Clear error messages for missing environment variables

### Bug Fixes
- Fixed chunk ID collision in vector store (using stem + index ID)
- Fixed IndexIDMap for efficient ID-based removal
- Fixed O(n) FAISS lookup in persistent memory (added reverse lookup)
- Fixed recency score edge cases with proper clamping
- Fixed patch matching with whitespace normalization
- Fixed missing `_save_index()` in `_rebuild_index()`

### Dependencies
- Added `pydantic>=2.0.0`
- Added `pydantic-settings>=2.0.0`

## [0.2.3] - 2025-01-20

### Bug Fixes
- Fixed patch application for files with exact matches
- Improved vector store index removal efficiency

## [0.2.2] - 2025-01-15

### New Features
- Persistent memory statistics in `forge memory-status`
- Improved session handoff packets

## [0.2.1] - 2025-01-10

### New Features
- Cross-session memory system
- Session logger for tracking operations

## [0.2.0] - 2025-01-01

### New Features
- Initial release with dual-LLM architecture
- Autonomous build mode with checkpoints
- Vector store with FAISS indexing
- Project graph with NetworkX