from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
import asyncio

class MakeUpStylistAgent(Agent):
    class StyleFlow(OneShotBehaviour):
        async def run(self):
            # 1. Appointment
            available_slots = [
                "monday 10:00", "monday 14:00",
                "tuesday 10:00", "tuesday 14:00",
                "friday 15:00", "friday 17:00"
            ]
            print("Available appointment slots:")
            for slot in available_slots:
                print(f"  - {slot}")

            appointment = ""
            while appointment not in available_slots:
                appointment = input("➡️ Choose one of the slots exactly as shown above: ").lower().strip()
                if appointment not in available_slots:
                    print("Invalid slot. Please choose from the list.")

            msg = Message(to="appointmentagent123@xmpp.jp")
            msg.body = f"Please schedule an appointment: {appointment}"
            await self.send(msg)
            print("[MakeUpStylist] Message sent to AppointmentSchedulerAgent.")

            # 2. Event
            self.event = input("🎉 What event are you preparing for? (e.g., wedding, work): ").lower()
            print(f"[MakeUpStylist] Event recorded: {self.event}")

            # 3. Skin description
            skin_desc = input("💆 Describe your skin (e.g., shiny, dry patches, visible pores): ")
            msg2 = Message(to="skinagent123@xmpp.jp")
            msg2.body = skin_desc
            await self.send(msg2)
            print("[MakeUpStylist] Message sent to SkinAnalyzerAgent.")

        async def on_end(self):
            print("[MakeUpStylist] Initial data collected.")

    class ReceiveResponses(CyclicBehaviour):
        def __init__(self):
            super().__init__()
            self.skin_type = None
            self.event = None
            self.product_sent = False

        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                sender = str(msg.sender).split("/")[0]
                content = msg.body

                if "appointmentagent123" in sender:
                    print(f"[MakeUpStylist] Appointment confirmed: {content}")

                elif "skinagent123" in sender:
                    print(f"[MakeUpStylist] Skin analysis: {content}")
                    self.skin_type = content.split(":", 1)[1].strip().lower()

                    skin_msg = Message(to="productagent123@xmpp.jp")
                    skin_msg.body = f"Skin type: {self.skin_type}"
                    await self.send(skin_msg)
                    print("[MakeUpStylist] Sent skin type to ProductRecommenderAgent.")

                    await asyncio.sleep(1)
                    event_msg = Message(to="productagent123@xmpp.jp")
                    event_msg.body = f"Can you recommend a product for this {self.agent.style_flow.event} look?"
                    await self.send(event_msg)
                    print("[MakeUpStylist] Requested product recommendation.")

                elif "productagent123" in sender and not self.product_sent:
                    print(f"[MakeUpStylist] Product recommendation: {content}")
                    decision = input("🛍️ Do you want to add this product to your cart? (yes/no): ").strip().lower()

                    if decision == "yes":
                        forward = Message(to="shopperagent123@xmpp.jp")
                        forward.body = f"add {content}"
                        await self.send(forward)
                        print("[MakeUpStylist] Product added to ShopperAgent.")
                    else:
                        print("[MakeUpStylist] Product declined.")

                    await asyncio.sleep(1)
                    checkout = Message(to="shopperagent123@xmpp.jp")
                    checkout.body = "checkout"
                    await self.send(checkout)
                    print("[MakeUpStylist] Sent checkout request.")
                    self.product_sent = True

                elif "shopperagent123" in sender:
                    print(f"[MakeUpStylist] ShopperAgent replied: {content}")
                else:
                    print(f"[MakeUpStylist] Unknown message: {content}")
            else:
                print("[MakeUpStylist] Waiting for messages...")

    async def setup(self):
        print(f"[{self.name}] Agent starting...")
        self.style_flow = self.StyleFlow()
        self.responses = self.ReceiveResponses()
        self.add_behaviour(self.style_flow)
        self.add_behaviour(self.responses)
