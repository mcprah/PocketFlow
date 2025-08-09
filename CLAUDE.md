# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PocketFlow is a 100-line minimalist LLM framework designed for building AI applications. The core framework is implemented in a single file (`pocketflow/__init__.py`) and provides fundamental abstractions for creating flows of AI-powered nodes.

## Commands for Development

### Testing
```bash
python -m pytest tests/
# Run specific test file
python -m pytest tests/test_flow_basic.py
# Run test with verbose output
python -m pytest -v tests/
```

### Running Examples
```bash
# Run any cookbook example
cd cookbook/pocketflow-hello-world
python main.py

# Install requirements for specific examples
pip install -r requirements.txt
```

### Installing the Framework
```bash
pip install pocketflow
# OR install in development mode
pip install -e .
```

## Core Architecture

### Framework Components

The framework consists of 5 core classes in `pocketflow/__init__.py`:

1. **BaseNode**: Base class with node lifecycle methods (`prep`, `exec`, `post`)
2. **Node**: Synchronous node with retry/fallback capabilities
3. **Flow**: Orchestrates node execution with conditional branching
4. **BatchNode/BatchFlow**: Process collections of data
5. **AsyncNode/AsyncFlow**: Asynchronous execution variants

### Key Concepts

- **Shared Store**: Dictionary passed between nodes for data communication
- **Node Lifecycle**: `prep()` → `exec()` → `post()` execution pattern
- **Actions**: String returns from `post()` method control flow branching
- **Transitions**: `>>` for default connections, `-"action">>` for conditional branching

### Project Structure Pattern

All cookbook examples follow this structure:
```
project_name/
├── main.py          # Entry point with shared store initialization
├── flow.py          # Flow definitions connecting nodes
├── nodes.py         # Node class definitions (optional)
├── utils/           # Utility functions (API calls, external integrations)
│   ├── __init__.py
│   ├── call_llm.py  # LLM API wrapper
│   └── [other utilities]
├── requirements.txt
└── docs/
    └── design.md    # High-level design documentation
```

## Development Guidelines

### Agentic Coding Approach

This project follows the "Agentic Coding" paradigm where humans design and AI agents implement:

1. **Always read relevant `.cursor/rules/*.mdc` files first** before implementing
2. Start with `docs/design.md` for high-level architecture
3. Implement utility functions in `utils/` directory with test functions
4. Design shared store data structure before implementing nodes
5. Implement nodes following the `prep/exec/post` pattern
6. Connect nodes in `flow.py` using `>>` and conditional branching

### Node Implementation Patterns

```python
class ExampleNode(Node):
    def prep(self, shared):
        # Extract data from shared store
        return shared.get("input_data")
    
    def exec(self, prep_result):
        # Core logic - avoid try/except, let Node handle retries
        return process_data(prep_result)
    
    def post(self, shared, prep_res, exec_res):
        # Store results and return action for flow control
        shared["output"] = exec_res
        return "next_action"  # or None for default transition
```

### Flow Connection Patterns

```python
# Default transitions
node1 >> node2 >> node3

# Conditional branching
node1 >> decision_node
decision_node - "positive" >> positive_action_node
decision_node - "negative" >> negative_action_node

# Creating flows
flow = Flow(start=node1)
```

### Testing Approach

- Tests are in `tests/` directory using unittest
- Test files follow pattern `test_[component].py`
- Tests verify node behavior, flow execution, and error handling
- Run tests to verify framework functionality after changes

### Common Patterns

1. **Agent Pattern**: Single node that makes decisions and takes actions
2. **Workflow Pattern**: Linear sequence of processing steps
3. **RAG Pattern**: Retrieval → Generation flow with external data
4. **Batch Pattern**: Process collections using BatchNode/BatchFlow
5. **Map-Reduce Pattern**: Split data → Process → Combine results

## Important Notes

- Framework has ZERO dependencies - keep it lightweight
- Utility functions should be project-specific, not built into framework
- Use shared store for all inter-node communication
- Let Node retry mechanism handle failures instead of manual try/catch
- Follow the established cookbook example patterns for consistency