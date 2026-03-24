import requests
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# ============================================================
# DISTRICT COORDINATES
# ============================================================
DISTRICT_COORDS = {
    "beed":                      {"lat": 18.9901, "lon": 75.7531},
    "Chhatrapati Sambhajinagar": {"lat": 19.8762, "lon": 75.3433},
    "Dhule":                     {"lat": 20.9042, "lon": 74.7749},
    "Jalgaon":                   {"lat": 21.0077, "lon": 75.5626},
    "Jalna":                     {"lat": 19.8410, "lon": 75.8864},
    "Wardha":                    {"lat": 20.7453, "lon": 78.6022},
    "Yavatmal":                  {"lat": 20.3899, "lon": 78.1307},
}

# ============================================================
# USER INPUT
# ============================================================
district_input = input("Enter district: ").strip()
date_input     = input("Enter prediction date (YYYY-MM-DD): ").strip()

# safe_name for model/scaler filenames
district_key = district_input.lower().replace(" ", "_")

# match exact key from DISTRICT_COORDS
district_cap = None
for key in DISTRICT_COORDS:
    if key.lower() == district_input.lower():
        district_cap = key
        break

if district_cap is None:
    raise ValueError(
        f"District '{district_input}' not found.\n"
        f"Available districts: {list(DISTRICT_COORDS.keys())}"
    )


# ============================================================
# FETCHER — meteorological inputs  (unit-corrected to match ERA5)
#
# ERA5 training data units:
#   msl              → Pa        (Open-Meteo gives hPa  → × 100)
#   wind_speed       → m/s       (Open-Meteo gives km/h → ÷ 3.6)
#   solar_radiation  → J/m²      (Open-Meteo gives W/m² hourly sum → × 3600)
#   relative_humidity→ %         (Open-Meteo gives % at 2m → same)
#   rainfall         → mm        (same)
# ============================================================
def fetch_paper_style_inputs(date_str, district):
    lat = DISTRICT_COORDS[district]["lat"]
    lon = DISTRICT_COORDS[district]["lon"]

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_date  = target_date - timedelta(days=4)
    end_date    = target_date
    today_real  = date.today()

    if end_date < today_real - timedelta(days=2):
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        base_url = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base_url}?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=relativehumidity_2m,"
        f"pressure_msl,windspeed_10m,"
        f"shortwave_radiation,precipitation"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=Asia/Kolkata"
    )

    for _ in range(3):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            break
        except:
            print("Retrying API call...")

    data   = response.json()
    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.date

    daily = hourly.groupby("date").agg({
        "relativehumidity_2m": "mean",   # %  → same
        "pressure_msl":        "mean",   # hPa → will convert below
        "windspeed_10m":       "mean",   # km/h → will convert below
        "shortwave_radiation": "sum",    # W/m² sum → will convert below
        "precipitation":       "sum",    # mm → same
    }).reset_index()

    # ---- Unit conversions to match ERA5 training data ----
    daily["pressure_msl"]    = daily["pressure_msl"]    * 100      # hPa  → Pa
    daily["windspeed_10m"]   = daily["windspeed_10m"]   / 3.6      # km/h → m/s
    daily["shortwave_radiation"] = daily["shortwave_radiation"] * 3600  # W/m² → J/m²

    daily["month_index"] = pd.to_datetime(daily["date"]).dt.month

    daily = daily.rename(columns={
        "pressure_msl":        "msl",
        "windspeed_10m":       "wind_speed",
        "shortwave_radiation": "solar_radiation",
        "relativehumidity_2m": "relative_humidity",
        "precipitation":       "rainfall",
        "month_index":         "month",
    })

    daily = daily[["date","msl","wind_speed","solar_radiation",
                   "relative_humidity","rainfall","month"]]

    daily = daily.fillna(method="ffill").fillna(method="bfill")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")
    return daily


# ============================================================
# FETCHER — actual Tmax  (°C, same as IMD training data)
# ============================================================
def fetch_tmax(date_str, district):
    lat = DISTRICT_COORDS[district]["lat"]
    lon = DISTRICT_COORDS[district]["lon"]

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_date  = target_date - timedelta(days=4)
    end_date    = target_date
    today_real  = date.today()

    if end_date < today_real - timedelta(days=2):
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        base_url = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base_url}?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=Asia/Kolkata"
    )

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.date

    daily = hourly.groupby("date").agg({"temperature_2m": "max"}).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")
    daily = daily.ffill().bfill()

    try:
        return float(daily["temperature_2m"].iloc[-1])
    except:
        return None


# ============================================================
# LOAD MODEL + SCALERS
# ============================================================
model = load_model(f"models/{district_key}_model.keras")

with open(f"scalers/{district_key}_scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

scaler_X = scalers["scaler_X"]
scaler_y = scalers["scaler_y"]

print(f"\nModel and scalers loaded for: {district_cap}\n")


# ============================================================
# FETCH INPUTS & RUN PREDICTION
# ============================================================
df = fetch_paper_style_inputs(date_input, district_cap)
df = df.sort_values("date")

features    = df[["msl","wind_speed","solar_radiation",
                   "relative_humidity","rainfall","month"]]
last_4_days = features.head(4)
scaled      = scaler_X.transform(last_4_days)
X_input     = np.expand_dims(scaled, axis=0)   # shape (1, 4, 6)

pred_scaled = model.predict(X_input)
pred_tmax   = scaler_y.inverse_transform(pred_scaled)[0]


# ============================================================
# PREDICTION RESULTS DATAFRAME
# ============================================================
base_date    = pd.to_datetime(date_input)
future_dates = [base_date + pd.Timedelta(days=i+1) for i in range(15)]

results_df = pd.DataFrame({
    "Day":            [f"Day {i+1:02d}" for i in range(15)],
    "Date":           [d.strftime("%Y-%m-%d") for d in future_dates],
    "Predicted_Tmax": [round(float(t), 2) for t in pred_tmax],
})

print("\n15-Day Tmax Forecast")
print("=" * 40)
print(results_df.to_string(index=False))
print("=" * 40)


# ============================================================
# RMSE CALCULATION
# Fetches actual Tmax for each of the 15 forecast days
# and computes RMSE per horizon
# ============================================================
print("\nFetching actual Tmax values for RMSE...")

pred_horizon   = [[] for _ in range(15)]
actual_horizon = [[] for _ in range(15)]

for i in range(15):
    target_date = base_date + pd.Timedelta(days=i+1)
    target_str  = target_date.strftime("%Y-%m-%d")

    try:
        actual = fetch_tmax(target_str, district_cap)
        if actual is None:
            print(f"  Day {i+1:02d}: actual Tmax not available")
            continue
        actual = float(actual)
    except Exception as e:
        print(f"  Day {i+1:02d}: fetch failed — {e}")
        continue

    pred_horizon[i].append(float(np.squeeze(pred_tmax[i])))
    actual_horizon[i].append(actual)

# Build RMSE dataframe
rmse_rows = []
for i in range(15):
    pred_val   = round(float(pred_tmax[i]), 2)
    if len(actual_horizon[i]) == 0:
        rmse_rows.append({
            "Day":            f"Day {i+1:02d}",
            "Date":           future_dates[i].strftime("%Y-%m-%d"),
            "Predicted_Tmax": pred_val,
            "Actual_Tmax":    "N/A",
            "RMSE":           "N/A",
        })
    else:
        rmse = round(np.sqrt(mean_squared_error(actual_horizon[i], pred_horizon[i])), 2)
        rmse_rows.append({
            "Day":            f"Day {i+1:02d}",
            "Date":           future_dates[i].strftime("%Y-%m-%d"),
            "Predicted_Tmax": pred_val,
            "Actual_Tmax":    round(actual_horizon[i][0], 2),
            "RMSE":           rmse,
        })

rmse_df = pd.DataFrame(rmse_rows)

print("\n===================================")
print(" HORIZON-WISE RMSE")
print("===================================\n")
print(rmse_df.to_string(index=False))
print("=" * 55)
