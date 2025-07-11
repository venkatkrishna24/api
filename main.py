from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests, json, folium, time, os, asyncio, pandas as pd
from datetime import datetime, date
import google.generativeai as genai

# -------------------- Config --------------------
API_KEY = os.getenv("OWM_API_KEY", "fallback-openweather-key")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_KEY) if GEMINI_KEY else None

HEATMAP_FILE    = "aqi_heatmap.html"
DISTRICTS_FILE  = "States_and_Districts.json"
HISTORICAL_CSV  = "aqi_history.csv"

app = FastAPI()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Utility --------------------
def get_aqi(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json()["list"][0]["main"]["aqi"]
    except Exception as e:
        print("❌ AQI fetch error:", e)
        return None

def get_color(aqi):
    return {1: "green", 2: "yellow", 3: "orange", 4: "red", 5: "purple"}.get(aqi, "gray")

def generate_health_advice(city, aqi_val):
    if not (GEMINI_KEY and aqi_val):
        return "No advice available."
    try:
        prompt = f"The AQI in {city} is {aqi_val}. Give a short health tip with risk level and precautions in one sentence."
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        print("❌ Gemini error:", e)
        return "AQI available. Consider staying indoors if sensitive."

def log_daily_aqi(city, aqi):
    today = date.today().isoformat()
    with open(HISTORICAL_CSV, "a", encoding="utf-8") as f:
        f.write(f"{city},{today},{aqi}\n")

def generate_heatmap():
    if not os.path.exists(DISTRICTS_FILE):
        print("⚠️ City JSON missing. Skipping heatmap.")
        return

    with open(DISTRICTS_FILE, "r", encoding="utf-8") as f:
        city_coords = json.load(f)

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    for city, (lat, lon) in city_coords.items():
        aqi = get_aqi(lat, lon)
        if aqi:
            color = get_color(aqi)
            folium.CircleMarker(
                [lat, lon], radius=6,
                popup=f"{city} — AQI: {aqi}",
                color=color, fill=True, fill_color=color, fill_opacity=0.7
            ).add_to(m)
            log_daily_aqi(city, aqi)
        time.sleep(1)

    legend = """<div style="position: fixed; bottom: 30px; left: 30px; width: 150px;
    border:2px solid grey; background:white; padding:10px; z-index:9999;">
    <b>AQI Legend</b><br>
    <span style='background-color:green'>&nbsp;&nbsp;&nbsp;</span> Good (1)<br>
    <span style='background-color:yellow'>&nbsp;&nbsp;&nbsp;</span> Fair (2)<br>
    <span style='background-color:orange'>&nbsp;&nbsp;&nbsp;</span> Moderate (3)<br>
    <span style='background-color:red'>&nbsp;&nbsp;&nbsp;</span> Poor (4)<br>
    <span style='background-color:purple'>&nbsp;&nbsp;&nbsp;</span> Very Poor (5)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(HEATMAP_FILE)
    print("✅ Heatmap updated.")

# -------------------- Background Task --------------------
@app.on_event("startup")
async def refresh_loop():
    async def loop():
        while True:
            print("🔁 Refreshing heatmap...")
            generate_heatmap()
            await asyncio.sleep(3600)
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

        cur = requests.get(f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}").json()
        fc  = requests.get(f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}").json()

        df_cur = pd.DataFrame([{
            "datetime": datetime.utcfromtimestamp(i["dt"]).isoformat(),
            "aqi": i["main"]["aqi"]
        } for i in cur.get("list", [])])

        df_fc = pd.DataFrame([{
            "datetime": datetime.utcfromtimestamp(i["dt"]).isoformat(),
            "aqi": i["main"]["aqi"]
        } for i in fc.get("list", [])])
        df_fc["date"] = pd.to_datetime(df_fc["datetime"]).dt.date
        df_7d = df_fc.groupby("date")["aqi"].mean().reset_index().head(7)

        # 30‑day history
        if os.path.exists(HISTORICAL_CSV):
            df_hist = pd.read_csv(HISTORICAL_CSV, names=["city", "date", "aqi"])
            history = df_hist[df_hist["city"] == city].tail(30)
        else:
            history = pd.DataFrame(columns=["city", "date", "aqi"])

        latest = df_cur.iloc[-1]["aqi"] if not df_cur.empty else None

        return {
            "city": city,
            "current_trend": df_cur.to_dict("records"),
            "forecast": df_fc.to_dict("records"),
            "forecast_7days": df_7d.to_dict("records"),
            "history": history.to_dict("records"),
            "gemini_advice": generate_health_advice(city, latest)
        }
    except Exception as e:
        print("❌ /aqi failed:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/aqi_by_coords")
def aqi_by_coords(lat: float = Query(...), lon: float = Query(...)):
    try:
        aqi = get_aqi(lat, lon)
        return {"lat": lat, "lon": lon, "aqi": aqi}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 👉 NEW: download CSV
@app.get("/download_history")
def download_history():
    if os.path.exists(HISTORICAL_CSV):
        return FileResponse(HISTORICAL_CSV, media_type="text/csv", filename=HISTORICAL_CSV)
    return JSONResponse(status_code=404, content={"error": "History not found"})

# -------------------- Local entry --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
