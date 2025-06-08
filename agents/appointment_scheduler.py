from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class AppointmentSchedulerAgent(Agent):
    class AppointmentBehaviour(CyclicBehaviour):
        def __init__(self):
            super().__init__()
            self.booked_times = []
            self.available_slots = [
                "monday 10:00", "monday 15:00",
                "tuesday 10:00", "tuesday 14:00",
                "friday 15:00", "friday 17:00"
            ]

        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"[AppointmentScheduler] Received request: {msg.body}")
                content = msg.body.strip().lower()

                if "please schedule an appointment:" in content:
                    requested_time = content.split(":", 1)[1].strip()

                    reply = Message(to=str(msg.sender))

                    if requested_time not in self.available_slots:
                        reply.body = (
                            f"Sorry, '{requested_time}' is not a valid slot.\n"
                            f"Available options: {', '.join(self.available_slots)}"
                        )
                        print(f"[AppointmentScheduler] Invalid request: {requested_time}")
                    elif requested_time in self.booked_times:
                        available = [s for s in self.available_slots if s not in self.booked_times]
                        reply.body = (
                            f"Sorry, {requested_time} is already booked.\n"
                            f"Try one of these: {', '.join(available)}"
                        )
                        print(f"[AppointmentScheduler] Rejected (already booked): {requested_time}")
                    else:
                        self.booked_times.append(requested_time)
                        reply.body = f"Appointment confirmed for: {requested_time}"
                        print(f"[AppointmentScheduler] Confirmed: {requested_time}")

                    await self.send(reply)
            else:
                print("[AppointmentScheduler] No message received.")

    async def setup(self):
        print(f"[{self.name}] AppointmentSchedulerAgent is starting...")
        self.add_behaviour(self.AppointmentBehaviour())
