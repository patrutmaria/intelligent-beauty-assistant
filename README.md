# 💄 Intelligent Beauty & Make-Up Assistant

A multi-agent system that helps users prepare for events by recommending beauty products based on skin type and event context, scheduling appointments, and managing a shopping cart with price and stock info.

---

## 🧠 Overview

This assistant simulates a virtual beauty consultation workflow using the [SPADE](https://spade-mas.readthedocs.io/) multi-agent framework. It includes five intelligent agents that collaborate via XMPP messaging:

- 🧑‍🎨 **Make-Up Stylist Agent** – collects user inputs and orchestrates the flow
- 🧴 **Skin Analyzer Agent** – determines the user's skin type from descriptive keywords
- 🛍️ **Product Recommender Agent** – suggests products based on event and skin type
- 📅 **Appointment Scheduler Agent** – handles simulated beauty salon bookings
- 🛒 **Shopper Agent** – manages the shopping cart, deals with stock and price

---

## 📦 Features

✅ Interactive console-based assistant  
✅ Custom event types (e.g., wedding, work, day)  
✅ Natural language skin description → automatic skin type detection  
✅ Personalized product recommendations with catalog lookup  
✅ Real-time appointment booking with slot availability  
✅ Shopping cart with **stock & price management**  
✅ Optional product acceptance (`yes`/`no`) before adding to cart

---

## 🚀 How to Run

1. **Install dependencies**
```bash
pip install spade
```

2. **Register XMPP agents**
- Use [xmpp.jp](https://xmpp.jp/signup.html) to register 5 usernames:
  - `beautytest123@xmpp.jp`
  - `productagent123@xmpp.jp`
  - `skinagent123@xmpp.jp`
  - `appointmentagent123@xmpp.jp`
  - `shopperagent123@xmpp.jp`

3. **Run the project**
```bash
python3 main.py
```

> ✅ The program will guide you through appointment, skin type, event, recommendation, and cart steps.

---

## 📁 Project Structure

```
beauty_assistant/
├── agents/
│   ├── makeup_stylist.py
│   ├── skin_analyzer.py
│   ├── product_recommender.py
│   ├── appointment_scheduler.py
│   └── shopper.py
├── main.py
└── README.md
```

---

## 🧬 Technologies Used

- **Python 3**
- **SPADE** – Multi-Agent System framework
- **XMPP** – Communication protocol (via xmpp.jp)

---

## ✨ Example Output

```text
🎉 What event are you preparing for? wedding
💆 Describe your skin: shiny
💡 Suggested product for wedding and oily skin: Estee Lauder Double Wear
💲 Price: $45 | Stock: 2
🛒 Do you want to add this product to your cart? yes
✅ Added to cart!
```

---

## 📌 Future Enhancements

- GUI interface (Tkinter or web)
- Product search by brand, price range, type
- External API for real-time product info

---

## 👩‍💻 Author

**Maria Pătruț** – Master’s Student in Artificial Intelligence  
📫 GitHub: [@patrutmaria](https://github.com/patrutmaria)

---

## 📃 License

This project is for educational/demo purposes. No commercial use of product names/brands.