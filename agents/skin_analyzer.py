from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
import random

class SkinAnalyzerAgent(Agent):
    class AnalyzeSkin(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"[SkinAnalyzer] Received request: {msg.body}")
                skin_types = ["oily", "dry", "combination", "normal"]
                selected = random.choice(skin_types)

                reply = Message(to=str(msg.sender))
                reply.body = f"Skin type: {selected}"
                await self.send(reply)
                print(f"[SkinAnalyzer] Sent skin analysis result: {selected}")
            else:
                print("[SkinAnalyzer] No message received.")

    async def setup(self):
        print(f"[{self.name}] Agent starting...")
        self.add_behaviour(self.AnalyzeSkin())
