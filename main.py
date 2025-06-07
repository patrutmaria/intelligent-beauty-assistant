import asyncio
from agents.makeup_stylist import MakeUpStylistAgent
from agents.product_recommender import ProductRecommenderAgent
from agents.skin_analyzer import SkinAnalyzerAgent

async def main():
    stylist_agent = MakeUpStylistAgent("beautytest123@xmpp.jp", "Test1234!")
    recommender_agent = ProductRecommenderAgent("productagent123@xmpp.jp", "Test1234!")
    skin_agent = SkinAnalyzerAgent("skinagent123@xmpp.jp", "Test1234!")

    await recommender_agent.start(auto_register=True)
    await skin_agent.start(auto_register=True)
    await stylist_agent.start(auto_register=True)

    await asyncio.sleep(15)

    await stylist_agent.stop()
    await recommender_agent.stop()
    await skin_agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
