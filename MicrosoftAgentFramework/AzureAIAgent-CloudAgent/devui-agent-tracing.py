# ! pip install agent-framework-devui==1.0.0b251001
import os
from dotenv import load_dotenv
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIClient
from azure.identity.aio import AzureCliCredential
from agent_framework.devui import serve
from azure.ai.projects.aio import AIProjectClient
import asyncio

# creating a function for agent creation
async def create_docs_agent(project_endpoint: str, model: str) -> ChatAgent:
    # Create async Azure credential
    credential = AzureCliCredential()
    
    # creating the Foundry Project Client
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential
    )

    # creating a conversation using the OpenAI Client
    openai_client = project_client.get_openai_client()
    conversation = await openai_client.conversations.create()
    conversation_id = conversation.id
    print("Conversation ID: ", conversation_id)

    # Initialize the Azure AI Agent Client
    chat_client = AzureAIClient(project_client=project_client,
                                conversation_id=conversation_id,
                                model_deployment_name=model)

    try:

        # Create the Docs Agent
        agent = chat_client.create_agent(
            name="docs-agent",
            instructions="You are a helpful assistant that can help with documentation questions."
        )

        print("✅ Agent created successfully!")
        return agent
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None

async def load_agent():
    load_dotenv()
    # loading environment variables
    project_endpoint = os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT")
    model = os.getenv("AI_FOUNDRY_DEPLOYMENT_NAME")

    agent = await create_docs_agent(project_endpoint=project_endpoint, model=model)

    return agent

def main():
    agent = asyncio.run(load_agent()) 
    serve(entities=[agent], auto_open=True, tracing_enabled=True)

if __name__ == "__main__":
    main()