# Dynamic MCP Client

A robust implementation of a Model Context Protocol (MCP) client with LLM orchestration.

## Features

- Dynamic capability discovery
- LLM-powered research planning
- Integrated error handling
- Session management
- Comprehensive logging

## Setup

1. Install dependencies:
```bash
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .
```

2. Configure environment variables:
```bash
export ANTHROPIC_API_KEY="your-key"
export TAVILY_API_KEY="your-key"
```

## Usage

```python
from clienttry1.console import ResearchConsole

console = ResearchConsole()
console.start()
```