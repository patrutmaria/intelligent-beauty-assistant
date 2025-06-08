from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class ShopperAgent(Agent):
    class ShoppingBehaviour(CyclicBehaviour):
        def __init__(self):
            super().__init__()
            self.cart = []

        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                content = msg.body.strip().lower()
                print(f"[Shopper] Received message: {msg.body}")
                reply = Message(to=str(msg.sender))

                if content.startswith("add"):
                    product = msg.body[4:].strip()
                    self.cart.append(product)
                    reply.body = f"✅ Product '{product}' added to your cart."
                    print(f"[Shopper] Added '{product}' to cart.")
                elif "checkout" in content:
                    if self.cart:
                        summary = ", ".join(self.cart)
                        reply.body = f"🛍️ Your cart: {summary}"
                    else:
                        reply.body = "🛒 Your cart is empty."
                    print("[Shopper] Sent cart summary.")
                else:
                    reply.body = "⚠️ Unrecognized command. Please use 'add <product>' or 'checkout'."

                await self.send(reply)
            else:
                print("[Shopper] No message received.")

    async def setup(self):
        print(f"[{self.name}] ShopperAgent is starting...")
        self.add_behaviour(self.ShoppingBehaviour())
