"""
Sequential Workflow with MAF and Microsoft Foundry

This script demonstrates a simple sequential workflow:
1. Researcher Agent gathers information on a topic.
2. Writer Agent writes an essay based on the research.

To run:
    python sequential_workflow.py
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    WorkflowViz,
)
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIClient
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from agent_framework.devui import serve

# Load environment variables
load_dotenv()
project_endpoint = os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT")
model = os.getenv("AI_FOUNDRY_DEPLOYMENT_NAME")

print("Project Endpoint:", project_endpoint)
print("Model:", model)

# Async agent creation utility
async def create_agent(agent_name: str, agent_instructions: str) -> ChatAgent:
    credential = AzureCliCredential()
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential
    )
    openai_client = project_client.get_openai_client()
    conversation = await openai_client.conversations.create()
    conversation_id = conversation.id
    print("Conversation ID:", conversation_id)

    chat_client = AzureAIClient(
        project_client=project_client,
        conversation_id=conversation_id,
        model_deployment_name=model
    )

    try:
        agent = chat_client.create_agent(
            name=agent_name,
            instructions=agent_instructions,
        )
        print(f"{agent_name} Agent created successfully!")
        return agent
    finally:
        await chat_client.close()
        await credential.close()

# Executor definitions
class ResearcherExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, query: str, ctx: WorkflowContext[str]) -> None:
        response = await self.agent.run(query)
        await ctx.send_message(str(response))

class WriterExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, research_data: str, ctx: WorkflowContext[str]) -> None:
        response = await self.agent.run(research_data)
        await ctx.yield_output(str(response))

async def build_workflow():
    # Create agents
    researcher_agent = await create_agent(
        agent_name="Researcher-Agent",
        agent_instructions=(
            "You are a knowledgeable researcher. Your task is to gather information and provide insights on a given topic. "
            "You should use reliable sources and present the information in a clear and concise manner."
        )
    )
    writer_agent = await create_agent(
        agent_name="Writer-Agent",
        agent_instructions=(
            "You are a creative writer. Your task is to write an essay on a given topic. "
            "You should focus on clarity, coherence, and engaging storytelling."
        )
    )

    # Instantiate Executors
    researcher_executor = ResearcherExecutor(researcher_agent, id="ResearcherExecutor")
    writer_executor = WriterExecutor(writer_agent, id="WriterExecutor")

    # Build the workflow
    workflow = (
        WorkflowBuilder(
            name="Sequential Research & Writing Workflow",
            description="A two-step workflow: research a topic, then write an essay."
        )
        .set_start_executor(researcher_executor)
        .add_edge(researcher_executor, writer_executor)
        .build()
    )

    # Optionally, visualize the workflow (prints Mermaid diagram to console)
    viz = WorkflowViz(workflow)
    mermaid_content = viz.to_mermaid()
    print("Mermaid Diagram:\n", mermaid_content)

    return workflow

def main():
    """Launch the sequential workflow in DevUI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting Sequential Research & Writing Workflow")
    logger.info("Available at: http://localhost:8090")
    logger.info("Entity ID: workflow_sequential_research_writer")

    # Run async workflow builder and launch DevUI
    workflow = asyncio.run(build_workflow())
    serve(entities=[workflow], port=8090, auto_open=True, tracing_enabled=True)

if __name__ == "__main__":
    main()