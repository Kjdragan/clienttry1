# llm.py

import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    """Orchestrates LLM interactions with dynamic MCP capabilities."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Anthropic API key is required")
            
        self.client = Anthropic(api_key=api_key)
        self.current_session: Dict[str, Any] = {}
        logger.info("LLM Orchestrator initialized")

    async def analyze_capabilities(self, capabilities: Dict[str, List[Dict[str, Any]]]) -> str:
        """Analyze available MCP capabilities."""
        try:
            capabilities_desc = json.dumps(capabilities, indent=2)
            
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze these MCP capabilities and summarize how they could be used for research:

Available Capabilities:
{capabilities_desc}

Explain:
1. What kinds of research tasks are possible
2. How the tools could be combined
3. Any limitations or gaps
4. Best practices for usage"""
                }]
            )
            
            # Extract text from the response content
            if response.content and len(response.content) > 0:
                return response.content[0].text
            return "No capability analysis available"
            
        except Exception as e:
            logger.error(f"Capability analysis failed: {e}")
            raise

    async def plan_research(self, query: str, capabilities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Plan research using available capabilities."""
        try:
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Research query: {query}

Available MCP capabilities:
{json.dumps(capabilities, indent=2)}

Create a research plan that:
1. Identifies which capabilities would be useful
2. Explains how to use them effectively
3. Specifies the order of operations
4. Includes any required parameters

Return the plan as JSON with:
- steps: Array of actions to take (tool/resource/prompt usage)
- parameters: Parameters for each step
- expected_outcomes: What each step should produce
- fallback_options: Alternative approaches if steps fail"""
                }]
            )
            
            # Parse JSON from the response text
            if response.content and len(response.content) > 0:
                try:
                    return json.loads(response.content[0].text)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from response")
                    return {
                        "error": "Invalid JSON response",
                        "raw_response": response.content[0].text
                    }
            return {
                "error": "No response content available"
            }