import os
from dotenv import load_dotenv

from livekit.agents import cli
from livekit.agents.voice import VoiceAgent
from livekit.plugins import deepgram, silero

from rag.chain import create_rag_chain

load_dotenv()

rag_chain = create_rag_chain()


class RAGVoiceAgent(VoiceAgent):
    def __init__(self):
        super().__init__(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
        )

    async def on_message(self, message: str):
        print("🗣 User:", message)

        try:
            response = rag_chain.invoke(message)
            print("🤖 Bot:", response)
            await self.say(response)
        except Exception as e:
            print("Error:", e)
            await self.say("Sorry, something went wrong.")


if __name__ == "__main__":
    cli.run_app(RAGVoiceAgent)
