from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

class AppointmentSchedulerAgent(Agent):
    class AppointmentBehaviour(CyclicBehaviour):
        def __init__(self):
            super().__init__()
            self.booked_times = []

        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                print(f"[AppointmentScheduler] Received request: {msg.body}")
                content = msg.body.strip()

                # Extract time from message (simple way)
                if "Please schedule an appointment:" in content:
                    requested_time = content.split(":", 1)[1].strip()

                    reply = Message(to=str(msg.sender))
                    if requested_time in self.booked_times:
                        reply.body = f"Sorry, {requested_time} is already booked. Please choose another time."
                        print(f"[AppointmentScheduler] Rejected booking for: {requested_time}")
                    else:
                        self.booked_times.append(requested_time)
                        reply.body = f"Appointment confirmed for: {requested_time}"
                        print(f"[AppointmentScheduler] Confirmed booking for: {requested_time}")

                    await self.send(reply)
            else:
                print("[AppointmentScheduler] No message received.")

    async def setup(self):
        print(f"[{self.name}] AppointmentSchedulerAgent is starting...")
        self.add_behaviour(self.AppointmentBehaviour())
