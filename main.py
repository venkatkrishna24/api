from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import folium
import time
from datetime import datetime, date
import pandas as pd
import os
import asyncio
import google.generativeai as genai

# -------------------- Config --------------------
API_KEY = os.getenv("OWM_API_KEY", "fallback-openweather-key")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_KEY) if GEMINI_KEY else None

HEATMAP_FILE = "aqi_heatmap.html"
DISTRICTS_FILE = "States_and_Districts.json"
HISTORICAL_CSV = "aqi_history.csv"

app = FastAPI()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Utility Functions --------------------
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
    if not GEMINI_KEY:
        return "Gemini API not configured."
    try:
        prompt = (
            f"The AQI in {city} is {aqi_val}. "
            f"Give a short health tip with risk level and precautions in one sentence."
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini error:", e)
        return "AQI available. Consider staying indoors if sensitive."

def log_daily_aqi(city, aqi):
    today = date.today().isoformat()
    with open(HISTORICAL_CSV, mode="a", encoding="utf-8") as f:
        f.write(f"{city},{today},{aqi}\n")

def generate_heatmap():
    try:
        if not os.path.exists(DISTRICTS_FILE):
            print("⚠️ Districts JSON missing. Skipping heatmap.")
            return

        with open(DISTRICTS_FILE, "r", encoding="utf-8") as f:
            city_coords = json.load(f)

        map_center = [20.5937, 78.9629]  # Center of India
        m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")

        for city, coords in city_coords.items():
            lat, lon = coords
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
                log_daily_aqi(city, aqi)
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
        print("✅ Heatmap generated successfully.")

    except Exception as e:
        print("❌ Error in generate_heatmap():", e)

# -------------------- Background Task --------------------
@app.on_event("startup")
async def refresh_heatmap_every_hour():
    async def loop():
        while True:
            print("🔁 Refreshing AQI heatmap...")
            generate_heatmap()
            await asyncio.sleep(3600)  # 1 hour
    asyncio.create_task(loop())

# -------------------- Routes --------------------
@app.get("/")
def home():
    return {"message": "✅ AQI backend is running."}

@app.get("/heatmap")
def serve_heatmap():
    if not os.path.exists(HEATMAP_FILE):
        generate_heatmap()
    return FileResponse(HEATMAP_FILE, media_type="text/html")

@app.get("/aqi")
def get_aqi_data(city: str = Query(...)):
    try:
        with open(DISTRICTS_FILE, "r", encoding="utf-8") as f:
            city_coords = json.load(f)

        if city not in city_coords:
            return JSONResponse(status_code=404, content={"error": "City not found"})

        lat, lon = city_coords[city]

        current_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        forecast_url = f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"

        current_data = requests.get(current_url).json().get("list", [])
        forecast_data = requests.get(forecast_url).json().get("list", [])

        df_current = pd.DataFrame([{
            "datetime": datetime.utcfromtimestamp(i["dt"]).isoformat(),
            "aqi": i["main"]["aqi"]
        } for i in current_data])

        df_forecast = pd.DataFrame([{
            "datetime": datetime.utcfromtimestamp(i["dt"]).isoformat(),
            "aqi": i["main"]["aqi"]
        } for i in forecast_data])

        df_forecast["date"] = pd.to_datetime(df_forecast["datetime"]).dt.date
        df_7day = df_forecast.groupby("date")["aqi"].mean().reset_index().head(7)

        # Load historical AQI
        if os.path.exists(HISTORICAL_CSV):
            df_hist = pd.read_csv(HISTORICAL_CSV, names=["city", "date", "aqi"])
            city_history = df_hist[df_hist["city"] == city].tail(30)
        else:
            city_history = pd.DataFrame(columns=["city", "date", "aqi"])

        latest = df_current.iloc[-1]["aqi"] if not df_current.empty else None
        advice = generate_health_advice(city, latest) if latest else "No AQI available."

        return {
            "city": city,
            "current_trend": df_current.to_dict(orient="records"),
            "forecast": df_forecast.to_dict(orient="records"),
            "forecast_7days": df_7day.to_dict(orient="records"),
            "history": city_history.to_dict(orient="records"),
            "gemini_advice": advice
        }

    except Exception as e:
        print("❌ AQI fetch failed:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.get("/aqi_by_coords")
def get_aqi_by_coords(lat: float = Query(...), lon: float = Query(...)):
    try:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        aqi = data["list"][0]["main"]["aqi"]
        return {
            "lat": lat,
            "lon": lon,
            "aqi": aqi
        }
    except Exception as e:
        print("❌ AQI fetch from coords failed:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------- Entry Point --------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Launching server...")
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
