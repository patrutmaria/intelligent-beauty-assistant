from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class SkinAnalyzerAgent(Agent):
    class AnalyzeSkin(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"[SkinAnalyzer] Received description: {msg.body}")
                description = msg.body.lower()

                # Keywords for analysis
                keywords = {
                    "oily": ["shiny", "greasy", "breakouts", "acne", "large pores"],
                    "dry": ["flaky", "tight", "rough", "dull", "itchy"],
                    "combination": ["t-zone", "mixed", "both dry and oily", "combination"],
                    "normal": ["balanced", "smooth", "clear", "normal"]
                }

                # Determine skin type
                result = "normal"  # default
                for skin_type, keys in keywords.items():
                    if any(k in description for k in keys):
                        result = skin_type
                        break

                reply = Message(to=str(msg.sender))
                reply.body = f"Skin type: {result}"
                await self.send(reply)
                print(f"[SkinAnalyzer] Sent skin analysis result: {result}")
            else:
                print("[SkinAnalyzer] No message received.")

    async def setup(self):
        print(f"[{self.name}] SkinAnalyzerAgent is starting...")
        self.add_behaviour(self.AnalyzeSkin())
