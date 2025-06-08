import asyncio
from agents.makeup_stylist import MakeUpStylistAgent
from agents.product_recommender import ProductRecommenderAgent
from agents.skin_analyzer import SkinAnalyzerAgent
from agents.appointment_scheduler import AppointmentSchedulerAgent
from agents.shopper import ShopperAgent

async def main():
    stylist_agent = MakeUpStylistAgent("beautytest123@xmpp.jp", "Test1234!")
    recommender_agent = ProductRecommenderAgent("productagent123@xmpp.jp", "Test1234!")
    skin_agent = SkinAnalyzerAgent("skinagent123@xmpp.jp", "Test1234!")
    appointment_agent = AppointmentSchedulerAgent("appointmentagent123@xmpp.jp", "Test1234!")
    shopper_agent = ShopperAgent("shopperagent123@xmpp.jp", "Test1234!")
   
    await recommender_agent.start(auto_register=True)
    await skin_agent.start(auto_register=True)
    await appointment_agent.start(auto_register=True)  
    await shopper_agent.start(auto_register=True)
    await stylist_agent.start(auto_register=True)

    print("Toți agenții sunt porniți!")

    await asyncio.sleep(60)

    await stylist_agent.stop()
    await recommender_agent.stop()
    await skin_agent.stop()
    await appointment_agent.stop()
    await shopper_agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
