from fastapi import FastAPI
from httpx import request
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

class GeoHussarResponse(BaseModel):
    answer: str = Field(description="The response to the user's question(s)")

class GeoHussarRequest(BaseModel):
    chat_message: str = Field(description="The user's chat message")


SYSTEM_PROMPT = """
You are known as Nexus, you specialize in the topic of geo-politics and geopolitical analysis with topics mainly around global conflicts.

You will not just simply provide a list of facts, you will also provide a analysis of the topic and a conclusion.

The way you conduct your analysis should be a result of your deep understanding of the topic and the context of the question.

Additionally, you should base your analysis on different reliable sources and perspectives in order to provide a comprehensive analysis.

The user would want your analysis to be concise and impactful.

Lastly, you shall also provide a list of sources where you got your information from.


"""

agent = Agent(
    'ollama:qwen3:8b',
    output_type=GeoHussarResponse,
    system_prompt=SYSTEM_PROMPT
)

@app.post("/")
async def test():
    try:
        result = await agent.run("What is the capital of France?")
        return result.output.answer
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def nexus_chat(chat: GeoHussarRequest):
    chat_message = chat.chat_message
    try:
        async def generate():
            async with agent.run_stream(chat_message) as result:
                async for chunk in result.stream_output():
                    answer = chunk.answer
                    yield answer.encode("utf-8")

        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )

    except Exception as e:
        return {"error": e}