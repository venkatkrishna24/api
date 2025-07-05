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
API_KEY = os.getenv("OWM_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

HEATMAP_FILE = "aqi_heatmap.html"

# -------------------- App Setup --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        res = requests.get(url)
        res.raise_for_status()
        return res.json()["list"][0]["main"]["aqi"]
    except Exception as e:
        print(f"❌ AQI fetch error: {e}")
        return None

def get_color(aqi):
    return {
        1: "green", 2: "yellow", 3: "orange", 4: "red", 5: "purple"
    }.get(aqi, "gray")

def generate_health_advice(city, aqi_val):
    prompt = f"The AQI in {city} is {aqi_val}. Give a short health tip with risk level and precautions."
    try:
        res = gemini_model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        print("❌ Gemini Error:", e)
        return "No health advice available."

def generate_heatmap():
    try:
        with open("district_coords.json", "r", encoding="utf-8") as f:
            coords_data = json.load(f)

        m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB positron")

        for city, (lat, lon) in coords_data.items():
            aqi = get_aqi(lat, lon)
            if aqi:
                color = get_color(aqi)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    popup=f"{city} — AQI: {aqi}",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(m)
            time.sleep(1)

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
        print(f"✅ Heatmap generated at {datetime.now().isoformat()}")
    except Exception as e:
        print("❌ Heatmap generation failed:", e)

@app.on_event("startup")
async def schedule_refresh():
    async def refresh_loop():
        while True:
            print("🔁 Background: Generating heatmap...")
            generate_heatmap()
            print("✅ Heatmap refresh done.")
            await asyncio.sleep(3600)
    asyncio.create_task(refresh_loop())

@app.get("/")
def root():
    print("✅ '/' route hit")
    return {"message": "AQI backend running"}

@app.get("/heatmap")
def get_heatmap():
    if not os.path.exists(HEATMAP_FILE):
        print("📁 No heatmap found, generating once...")
        generate_heatmap()
    return FileResponse(HEATMAP_FILE, media_type="text/html")

@app.get("/aqi")
def get_aqi_json(city: str = Query(...)):
    coords = get_coordinates(city)
    if not coords:
        return JSONResponse(status_code=404, content={"error": "Location not found"})
    lat, lon = coords

    try:
        forecast_res = requests.get(
            f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
        )
        current_res = requests.get(
            f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        )

        try:
            forecast_data = forecast_res.json().get("list", [])
            current_data = current_res.json().get("list", [])
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "API JSON error: " + str(e)})

        df_current = pd.DataFrame([{
            "datetime": datetime.utcfromtimestamp(i["dt"]).isoformat(),
            "aqi": i["main"]["aqi"]
        } for i in current_data])

        df_forecast = pd.DataFrame([{
            "datetime": datetime.utcfromtimestamp(i["dt"]).isoformat(),
            "aqi": i["main"]["aqi"]
        } for i in forecast_data])

        combined = pd.concat([df_current, df_forecast])
        latest_aqi = combined.iloc[-1]["aqi"] if not combined.empty else None
        advice = generate_health_advice(city, latest_aqi)

        return {
            "city": city,
            "current_trend": df_current.to_dict(orient="records"),
            "forecast": df_forecast.to_dict(orient="records"),
            "gemini_advice": advice
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🔧 Local Run (if needed)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
