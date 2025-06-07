from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message

class MakeUpStylistAgent(Agent):
    class RecommendLookAndAskProduct(OneShotBehaviour):
        async def run(self):
            print("[MakeUpStylist] Recommending a look...")
            look = "Soft glam with neutral tones"
            print(f"[MakeUpStylist] Suggested Look: {look}")

            msg = Message(to="productagent123@xmpp.jp")  # ← înlocuiește cu JID real dacă e diferit
            msg.body = "Can you recommend a product for this Soft glam look?"
            await self.send(msg)
            print("[MakeUpStylist] Message sent to ProductRecommenderAgent.")

    class ReceiveRecommendation(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"[MakeUpStylist] Received product recommendation: {msg.body}")
            else:
                print("[MakeUpStylist] Waiting for product recommendation...")

    async def setup(self):
        print(f"[{self.name}] Agent starting...")
        self.add_behaviour(self.RecommendLookAndAskProduct())
        self.add_behaviour(self.ReceiveRecommendation())
