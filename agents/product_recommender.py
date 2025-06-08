from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class ProductRecommenderAgent(Agent):
    class RecommendProduct(CyclicBehaviour):
        def __init__(self):
            super().__init__()
            self.last_skin_type = None
            self.products = {
                "wedding": {
                    "oily": "Estee Lauder Double Wear",
                    "dry": "Dior Forever Skin Glow",
                    "combination": "MAC Studio Fix",
                    "normal": "NARS Sheer Glow"
                },
                "office": {
                    "oily": "L'Oreal Infallible Matte",
                    "dry": "Maybelline Dream Satin",
                    "combination": "Revlon ColorStay",
                    "normal": "NYX Born to Glow"
                },
                "day": {
                    "oily": "Neutrogena Skin Tint",
                    "dry": "Garnier BB Cream",
                    "combination": "L'Oreal True Match",
                    "normal": "Clinique Even Better"
                },
                "night": {
                    "oily": "Fenty Soft Matte",
                    "dry": "Charlotte Tilbury Light Wonder",
                    "combination": "Milani Conceal+Perfect",
                    "normal": "Too Faced Born This Way"
                }
            }

        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"[ProductRecommender] Received message: {msg.body}")
                reply = Message(to=str(msg.sender))
                body = msg.body.lower()

                # If it's a skin type update
                if body.startswith("skin type:"):
                    self.last_skin_type = body.split(":", 1)[1].strip()
                    print(f"[ProductRecommender] Updated skin type to: {self.last_skin_type}")
                    return

                # Try to determine the event
                event = None
                for key in self.products:
                    if key in body:
                        event = key
                        break

                if not event or not self.last_skin_type:
                    reply.body = "Missing event or skin type. Cannot recommend."
                else:
                    skin = self.last_skin_type
                    suggestion = self.products.get(event, {}).get(skin, "No matching product.")
                    reply.body = f"Suggested product for {event} and {skin} skin: {suggestion}"

                await self.send(reply)
                print(f"[ProductRecommender] Sent: {reply.body}")
            else:
                print("[ProductRecommender] No message received.")

    async def setup(self):
        print(f"[{self.name}] ProductRecommenderAgent is starting...")
        self.add_behaviour(self.RecommendProduct())
