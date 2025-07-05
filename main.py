from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import folium
import time
from datetime import datetime
from geopy.geocoders import Nominatim
import pandas as pd
import os
import asyncio
import google.generativeai as genai

# -------------------- Config --------------------
print("📦 Starting FastAPI AQI backend...")
API_KEY = os.getenv("OWM_API_KEY", "fallback-openweather-key")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

HEATMAP_FILE = "aqi_heatmap.html"

# -------------------- App Init --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Helper Functions --------------------
def get_coordinates(place):
    try:
        geolocator = Nominatim(user_agent="aqi_app")
        location = geolocator.geocode(place + ", India", timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print("❌ Geolocation error:", e)
    return None

def get_aqi(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return data["list"][0]["main"]["aqi"]
    except Exception as e:
        print("❌ AQI fetch error:", e)
        return None

def get_color(aqi):
    return {
        1: "green", 2: "yellow", 3: "orange", 4: "red", 5: "purple"
    }.get(aqi, "gray")

def generate_health_advice(city: str, aqi_val: int):
    prompt = (
        f"The AQI in {city} is {aqi_val}. "
        f"Give a short health tip with risk level and precautions in one sentence."
    )
    try:
        res = gemini_model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        print("❌ Gemini error:", e)
        return "AQI data available. Consider staying indoors if sensitive."

def generate_heatmap():
    try:
        with open("States and Districts.json", "r", encoding="utf-8") as f:
            states_data = json.load(f)
        districts = [d["name"] for s in states_data for d in s["districts"]]
        map_center = get_coordinates("India")
        m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")

        for district in districts:
            coords = get_coordinates(district)
            if coords:
                lat, lon = coords
                aqi = get_aqi(lat, lon)
                if aqi:
                    color = get_color(aqi)
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        popup=f"{district} — AQI: {aqi}",
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7
                    ).add_to(m)
                time.sleep(1)  # API rate limit friendly

        legend = """
        <div style="position: fixed; bottom: 30px; left: 30px; width: 150px;
        border:2px solid grey; background-color:white; padding:10px; z-index:9999;">
        <b>AQI Legend</b><br>
        <span style='background-color:green'>&nbsp;&nbsp;&nbsp;</span> Good (1)<br>
        <span style='background-color:yellow'>&nbsp;&nbsp;&nbsp;</span> Fair (2)<br>
        <span style='background-color:orange'>&nbsp;&nbsp;&nbsp;</span> Moderate (3)<br>
        <span style='background-color:red'>&nbsp;&nbsp;&nbsp;</span> Poor (4)<br>
        <span style='background-color:purple'>&nbsp;&nbsp;&nbsp;</span> Very Poor (5)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))
        m.save(HEATMAP_FILE)
        print("✅ Heatmap generated")
    except Exception as e:
        print("❌ Heatmap generation failed:", e)

# -------------------- Background Heatmap Refresh --------------------
@app.on_event("startup")
async def safe_background_refresh():
    async def loop():
        while True:
            try:
                print("🔁 Background: Generating heatmap...")
                generate_heatmap()
            except Exception as e:
                print(f"❌ Heatmap generation failed in background: {e}")
            await asyncio.sleep(3600)  # every hour

    try:
        asyncio.create_task(loop())
        print("✅ Background refresh loop started.")
    except Exception as e:
        print(f"❌ Failed to create background task: {e}")

# -------------------- Routes --------------------
@app.get("/")
def root():
    print("✅ '/' route hit")
    return {"message": "AQI backend is live"}

@app.get("/heatmap")
def serve_heatmap():
    if not os.path.exists(HEATMAP_FILE):
        generate_heatmap()
    return FileResponse(HEATMAP_FILE, media_type="text/html")

@app.get("/aqi")
def get_aqi_json(city: str = Query(...)):
    coords = get_coordinates(city)
    if not coords:
        return JSONResponse(status_code=404, content={"error": "Location not found"})
    lat, lon = coords

    forecast_url = f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    current_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    try:
        forecast_data = requests.get(forecast_url).json().get("list", [])
        current_data = requests.get(current_url).json().get("list", [])

        df_current = pd.DataFrame([{
            "datetime": datetime.utcfromtimestamp(item["dt"]).isoformat(),
            "aqi": item["main"]["aqi"]
        } for item in current_data])

        df_forecast = pd.DataFrame([{
            "datetime": datetime.utcfromtimestamp(item["dt"]).isoformat(),
            "aqi": item["main"]["aqi"]
        } for item in forecast_data])

        latest_aqi = pd.concat([df_current, df_forecast]).iloc[-1]["aqi"] if not df_forecast.empty else None
        advice = generate_health_advice(city, latest_aqi) if latest_aqi else "No AQI data available."

        return {
            "city": city,
            "current_trend": df_current.to_dict(orient="records"),
            "forecast": df_forecast.to_dict(orient="records"),
            "gemini_advice": advice
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    print("🚀 Launching server on port 10000")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
