from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
import matplotlib.pyplot as plt
from io import BytesIO
import google.generativeai as genai

# -------------------- CONFIG --------------------
API_KEY = os.getenv("OWM_API_KEY", "your_openweather_api_key")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key")

genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
HEATMAP_FILE = "aqi_heatmap.html"

# -------------------- APP SETUP --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- UTILITY --------------------
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

# -------------------- BACKGROUND TASK --------------------
@app.on_event("startup")
async def schedule_refresh():
    async def refresh_loop():
        while True:
            print("🔁 Generating heatmap...")
            generate_heatmap()
            await asyncio.sleep(3600)
    asyncio.create_task(refresh_loop())

# -------------------- ROUTES --------------------
@app.get("/")
def root():
    return {"message": "AQI backend running"}

@app.get("/heatmap")
def get_heatmap():
    if not os.path.exists(HEATMAP_FILE):
        generate_heatmap()
    return FileResponse(HEATMAP_FILE, media_type="text/html")

@app.get("/aqi")
def get_aqi_json(city: str = Query(...)):
    coords = get_coordinates(city)
    if not coords:
        return JSONResponse(status_code=404, content={"error": "Location not found"})
    lat, lon = coords

    try:
        forecast_url = f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
        current_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

        forecast_data = requests.get(forecast_url).json().get("list", [])
        current_data = requests.get(current_url).json().get("list", [])

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

@app.get("/aqi-graph")
def get_aqi_graph(city: str = Query(...)):
    coords = get_coordinates(city)
    if not coords:
        return JSONResponse(status_code=404, content={"error": "Location not found"})
    lat, lon = coords

    try:
        END = int(time.time())
        START = END - (5 * 24 * 60 * 60)

        def fetch_api_data(url):
            try:
                res = requests.get(url)
                res.raise_for_status()
                return res.json().get("list", [])
            except Exception as e:
                print("❌ API error:", e)
                return []

        def build_dataframe(data, label):
            return pd.DataFrame([{
                "datetime": datetime.utcfromtimestamp(item["dt"]),
                "aqi": item["main"]["aqi"]
            } for item in data]).assign(source=label)

        hist_url = f"https://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={START}&end={END}&appid={API_KEY}"
        fore_url = f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"

        df_hist = build_dataframe(fetch_api_data(hist_url), "Historical")
        df_fore = build_dataframe(fetch_api_data(fore_url), "Forecast")
        df_combined = pd.concat([df_hist, df_fore])

        if df_combined.empty:
            return JSONResponse(status_code=500, content={"error": "No AQI data to plot."})

        plt.figure(figsize=(14, 6))
        for label, group in df_combined.groupby("source"):
            plt.plot(group["datetime"], group["aqi"], marker="o", label=label)

        plt.axhspan(0.5, 1.5, color='green', alpha=0.2, label='Good')
        plt.axhspan(1.5, 2.5, color='yellow', alpha=0.2, label='Fair')
        plt.axhspan(2.5, 3.5, color='orange', alpha=0.2, label='Moderate')
        plt.axhspan(3.5, 4.5, color='red', alpha=0.2, label='Poor')
        plt.axhspan(4.5, 5.5, color='purple', alpha=0.2, label='Very Poor')

        plt.title(f"AQI: Historical + Forecast — {city.title()}")
        plt.xlabel("Datetime")
        plt.ylabel("AQI (1=Good, 5=Very Poor)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------- LOCAL RUN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
