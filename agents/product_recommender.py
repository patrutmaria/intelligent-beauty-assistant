from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class ProductRecommenderAgent(Agent):
    class RecommendProduct(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"[ProductRecommender] Received request: {msg.body}")
                reply = Message(to=str(msg.sender))
                reply.body = "Suggested product: Maybelline Fit Me Foundation"
                await self.send(reply)
                print("[ProductRecommender] Sent recommendation.")
            else:
                print("[ProductRecommender] No message received.")

    async def setup(self):
        print(f"[{self.name}] Agent starting...")
        self.add_behaviour(self.RecommendProduct())  # fără template!
