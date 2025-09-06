import re
import pandas as pd
import json
from openai import OpenAI

attractions = pd.read_csv("uae_attractions.csv")
hotels = pd.read_csv("uae_hotels.csv")
restaurants = pd.read_csv("uae_restaurants.csv")

attractions.columns = attractions.columns.str.strip()
hotels.columns = hotels.columns.str.strip()
restaurants.columns = restaurants.columns.str.strip()

rating_map = {
    "OneStar": 1,
    "TwoStar": 2,
    "ThreeStar": 3,
    "FourStar": 4,
    "FiveStar": 5
}
hotels["HotelRating"] = hotels["HotelRating"].map(rating_map)
restaurants["Average Cost for two"] = pd.to_numeric(
    restaurants["Average Cost for two"], errors="coerce"
)

def clean_value(val, default="Not Available"):
    if pd.isna(val) or str(val).strip().lower() in ["nan", "none", ""]:
        return default
    return str(val)

def format_rating(rating):
    if pd.isna(rating):
        return "Not Rated"
    return f"{int(rating)}-Star"

def build_itinerary(query):
    city, days, budget, currency, preferences = None, None, None, "AED", []

    days_match = re.search(r'(\d+)\s*[- ]?\s*(day|days|night|nights)', query, re.IGNORECASE)
    if days_match:
        days = int(days_match.group(1))
        if "night" in days_match.group(2).lower():
            days += 1
            
    budget_match = re.search(r'(?:under|budget|cost|price)\s*(\d+)\s*(AED|Dhs|\$|USD)?', query, re.IGNORECASE)
    if budget_match:
        budget = int(budget_match.group(1))
        if budget_match.group(2):
            currency = budget_match.group(2).upper().replace("DHS", "AED").replace("$", "USD")

    known_cities = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Fujairah", "Ras Al Khaimah", "Umm Al Quwain"]
    for c in known_cities:
        if c.lower() in query.lower():
            city = c
            break
        
    keywords = ["culture", "food", "shopping", "adventure", "nature", "beach", "museum", "luxury","theme park"]
    for kw in keywords:
        if kw in query.lower():
            preferences.append(kw)

    parsed = {
        "city": city,
        "days": days,
        "budget": budget,
        "currency": currency,
        "preferences": preferences
    }

    city_attractions = attractions[attractions["City"].str.lower() == city.lower()]
    city_hotels = hotels[hotels["cityName"].str.lower() == city.lower()]
    city_restaurants = restaurants[restaurants["City"].str.lower() == city.lower()]

    city_attractions = city_attractions.sample(frac=1).reset_index(drop=True)
    city_hotels = city_hotels.sample(frac=1).reset_index(drop=True)
    city_restaurants = city_restaurants.sample(frac=1).reset_index(drop=True)

    if preferences:
        pref_matches = city_attractions[
            city_attractions["Category"].str.lower().isin([p.lower() for p in preferences])
        ]
        if not pref_matches.empty:
            non_pref = city_attractions[~city_attractions.index.isin(pref_matches.index)]
            city_attractions = pd.concat([pref_matches, non_pref]).drop_duplicates().reset_index(drop=True)

    if budget:
        if budget <= 2000:
            filtered_hotels = city_hotels[city_hotels["HotelRating"] <= 3]
            filtered_restaurants = city_restaurants[city_restaurants["Average Cost for two"] <= 150]
        elif 2000 < budget <= 5000:
            filtered_hotels = city_hotels[(city_hotels["HotelRating"] > 3) & (city_hotels["HotelRating"] <= 4)]
            filtered_restaurants = city_restaurants[
                (city_restaurants["Average Cost for two"] > 150) &
                (city_restaurants["Average Cost for two"] <= 300)
            ]
        else:
            filtered_hotels = city_hotels[city_hotels["HotelRating"] > 4]
            filtered_restaurants = city_restaurants[city_restaurants["Average Cost for two"] > 300]

        if not filtered_hotels.empty:
            city_hotels = filtered_hotels
        if not filtered_restaurants.empty:
            city_restaurants = filtered_restaurants

    itinerary = {}
    for day in range(1, days + 1):
        morning = city_attractions.iloc[(day * 2 - 2) % len(city_attractions)]
        afternoon = city_attractions.iloc[(day * 2 - 1) % len(city_attractions)]
        restaurant = city_restaurants.iloc[(day - 1) % len(city_restaurants)] if not city_restaurants.empty else None
        hotel = city_hotels.iloc[(day - 1) % len(city_hotels)] if not city_hotels.empty else None

        itinerary[f"Day {day}"] = {
            "Morning": f"{morning['Name']} ({morning['Category']}) – {morning['Description']}",
            "Afternoon": f"{afternoon['Name']} ({afternoon['Category']}) – {afternoon['Description']}",
            "Dinner": "No restaurants available" if restaurant is None else f"{clean_value(restaurant['Restaurant Name'])} 🍴 {clean_value(restaurant['Cuisines'])} | ⭐ {clean_value(restaurant['Aggregate rating'], 'Not Rated')} ({clean_value(restaurant['Votes'], '0')} reviews) | 💰 {clean_value(restaurant['Average Cost for two'], 'N/A')} AED for 2 people",
            "Hotel": "No hotels available" if hotel is None else f"{clean_value(hotel['HotelName'])} ⭐ {format_rating(hotel['HotelRating'])} | 📞 {clean_value(hotel['PhoneNumber'], 'No contact')} | 🌐 {clean_value(hotel['HotelWebsiteUrl'], 'No website listed')}"
        }

    return parsed, itinerary


from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID if OPENAI_ORG_ID else None,
    project=OPENAI_PROJECT_ID if OPENAI_PROJECT_ID else None
)


def make_human_like(parsed, itinerary):
    import json
    
    days = parsed.get("days", len(itinerary))
    city = parsed.get("city", "your destination")

    prompt = f"""
    You are a professional travel curator.
    Create a {days}-day travel itinerary for {city} in the **Mindtrip.ai style**.

    Formatting rules:
    - Title: "**{city} – {days} Day Itinerary**" + add country flag if known.
    - Add a short tagline (one catchy sentence).
    - Use headings: "**Day X – …**" with an emoji.
    - Subsections: "**☀️ Morning:**", "**🌤️ Afternoon:**", "**🌙 Evening:**"
    - Each subsection should be a short paragraph (2–3 sentences), not bullet points.
    - **Day 1 Morning must include hotel check-in** (pick one realistic hotel).
    - From Day 2 onwards, only say "Breakfast at hotel" (same hotel throughout).
    - **Bold all hotels, restaurants, landmarks, and key experiences.**
    - Keep tone lively, polished, and smooth storytelling — like a premium travel app.
    - No robotic listing, no filler blog tone.
    - Always cover exactly {days} days.

    JSON itinerary data (for reference only):
    {json.dumps(itinerary, indent=2, ensure_ascii=False)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    query = input("Enter your travel query: ")
    parsed, itinerary = build_itinerary(query)
    print(make_human_like(parsed, itinerary))

# print("\nParsed:", parsed)
# print("\nStructured JSON:\n", json.dumps(itinerary, indent=2, ensure_ascii=False))
# print("\n✨ Itinerary suggestion:")
