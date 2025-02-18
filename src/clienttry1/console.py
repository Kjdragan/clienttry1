# console.py

import os
import asyncio
import logging
from typing import Optional, Dict, Any

from .client import MCPClient
from .llm import LLMOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchConsole:
    """Interactive console for research with MCP capabilities."""
    
    def __init__(self):
        self.mcp_client = MCPClient()
        self.llm: Optional[LLMOrchestrator] = None
        self.session_active = False
        self.current_query: Optional[str] = None
        self.results: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize components and connections."""
        try:
            logger.info("Initializing components...")
            
            # Initialize LLM
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            self.llm = LLMOrchestrator(api_key)
            
            # Connect to MCP server
            logger.info("Connecting to MCP server...")
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                raise ValueError("TAVILY_API_KEY environment variable is required")
                
            await self.mcp_client.connect_to_server(
                env={"TAVILY_API_KEY": tavily_api_key}
            )
            
            # Analyze capabilities
            capabilities = await self.mcp_client.discover_capabilities()
            analysis = await self.llm.analyze_capabilities(capabilities)
            logger.info("Capability analysis complete")
            
            self.session_active = True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise
            
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.mcp_client:
                await self.mcp_client.cleanup()
                logger.info("MCP client cleanup completed")
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
            
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a research query using available capabilities."""
        if not self.session_active or not self.llm:
            raise ValueError("Console not initialized")
            
        try:
            self.current_query = query
            
            # Get capabilities for planning
            capabilities = await self.mcp_client.discover_capabilities()
            
            # Plan research steps
            plan = await self.llm.plan_research(query, capabilities)
            
            # Execute research plan
            results = await self.llm.execute_research_plan(plan, self.mcp_client)
            
            # Analyze results
            analysis = await self.llm.analyze_results(results)
            
            # Store results
            self.results[query] = {
                'plan': plan,
                'results': results,
                'analysis': analysis,
                'timestamp': results['metadata']['start_time']
            }
            
            return self.results[query]
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise
            
    def get_session_state(self) -> Dict[str, Any]:
        """Get current session state summary."""
        return {
            'active': self.session_active,
            'current_query': self.current_query,
            'query_count': len(self.results),
            'latest_query': max(
                (r['timestamp'] for r in self.results.values()),
                default=None
            ) if self.results else None
        }
        
    async def run(self):
        """Run the interactive console."""
        print("\nInitializing Research Console...")
        
        try:
            await self.initialize()
            print("\nResearch Console Ready!")
            print("Enter your research queries. Type 'exit' to quit.\n")
            
            while True:
                query = input("\nResearch Query > ").strip()
                
                if query.lower() in ['exit', 'quit']:
                    break
                    
                if not query:
                    continue
                    
                try:
                    results = await self.process_query(query)
                    
                    # Display results summary
                    if results.get('analysis'):
                        print("\nKey Findings:")
                        for finding in results['analysis'].get('findings', []):
                            print(f"- {finding}")
                            
                        print("\nRecommendations:")
                        for rec in results['analysis'].get('recommendations', []):
                            print(f"- {rec}")
                    else:
                        print("\nNo analysis available for results")
                        
                except Exception as e:
                    print(f"\nError processing query: {str(e)}")
                    
        except Exception as e:
            print(f"\nConsole error: {str(e)}")
            
        finally:
            await self.cleanup()
            print("\nSession ended.")
            
    @classmethod
    def start(cls):
        """Start the research console."""
        console = cls()
        asyncio.run(console.run())