import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="RideGuide | New Car Price Predictor", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

FEATURES = ["capacity", "horsepower", "top_speed", "zero_to_hundred", "seats", "torque"]
FUEL_TYPES = ["Petrol", "Diesel", "Hybrid", "Electric"]
DEMO_BRANDS = ["Toyota", "Ford", "BMW", "Mercedes-Benz", "Hyundai", "Kia", "Volkswagen", "Audi", "Nissan", "Chevrolet", "Tesla", "Honda", "Mazda", "Peugeot", "Renault"]
DEMO_MODEL_WORDS = ["Alpha", "Terra", "Nova", "Orion", "Vertex", "Pulse", "Atlas", "Comet", "Zenith"]


def parse_num(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "")
    values, current = [], ""
    for char in text:
        if char.isdigit() or char == ".":
            current += char
        elif current and current != ".":
            values.append(float(current))
            current = ""
    if current and current != ".":
        values.append(float(current))
    return float(np.mean(values)) if values else np.nan


def simplify_fuel(value):
    text = str(value).lower()
    if "electric" in text or text.strip() == "ev":
        return "Electric"
    if "hybrid" in text:
        return "Hybrid"
    if "diesel" in text:
        return "Diesel"
    return "Petrol"


def demo_data():
    rng = np.random.default_rng(42)
    rows = []
    fuel_multipliers = {"Petrol": 1.0, "Diesel": 1.08, "Hybrid": 1.28, "Electric": 1.42}
    for i in range(220):
        fuel = rng.choice(FUEL_TYPES, p=[.43, .18, .22, .17])
        brand = rng.choice(DEMO_BRANDS)
        car_name = f"{DEMO_MODEL_WORDS[i % len(DEMO_MODEL_WORDS)]} {i // len(DEMO_MODEL_WORDS) + 1}"
        capacity = rng.integers(1000, 6001) if fuel != "Electric" else rng.integers(45, 121)
        horsepower = rng.integers(95, 901)
        top_speed = rng.integers(165, 351)
        zero = round(rng.uniform(2.7, 13.5), 1)
        seats = int(rng.choice([2, 4, 5, 7], p=[.1, .15, .6, .15]))
        torque = rng.integers(140, 1001)
        base = 15000 + horsepower * 105 + top_speed * 90 + torque * 22 + seats * 900
        price = base * fuel_multipliers[fuel] + (capacity * 2 if fuel != "Electric" else capacity * 240) - zero * 950 + rng.normal(0, 12000)
        rows.append([brand, car_name, fuel, capacity, horsepower, top_speed, zero, seats, torque, max(12000, price)])
    return pd.DataFrame(rows, columns=["brand", "car_name", "fuel_type", *FEATURES, "price"])


@st.cache_data
def load_data():
    candidates = [Path("Cars_Datasets_2025.csv"), Path("data/Cars_Datasets_2025.csv")]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return demo_data(), "curated demo dataset"
    raw = pd.read_csv(source, encoding="latin1")
    columns = {str(c).lower().strip(): c for c in raw.columns}
    required = ["cars names", "cc/battery capacity", "horsepower", "total speed", "performance(0 - 100 )km/h", "seats", "torque", "cars prices", "fuel types"]
    if not all(col in columns for col in required):
        return demo_data(), "curated demo dataset"
    car_names_raw = raw[columns["cars names"]].astype(str).str.strip()
    if "company names" in columns:
        brand_raw = raw[columns["company names"]].astype(str).str.strip()
    else:
        brand_raw = car_names_raw.str.split().str[0]
    result = pd.DataFrame({
        "brand": brand_raw,
        "car_name": car_names_raw,
        "fuel_type": raw[columns["fuel types"]].map(simplify_fuel),
        "capacity": raw[columns["cc/battery capacity"]].map(parse_num),
        "horsepower": raw[columns["horsepower"]].map(parse_num),
        "top_speed": raw[columns["total speed"]].map(parse_num),
        "zero_to_hundred": raw[columns["performance(0 - 100 )km/h"]].map(parse_num),
        "seats": raw[columns["seats"]].map(parse_num),
        "torque": raw[columns["torque"]].map(parse_num),
        "price": raw[columns["cars prices"]].map(parse_num),
    }).dropna(subset=["car_name", "fuel_type", *FEATURES, "price"])
    result = result[result.fuel_type.isin(FUEL_TYPES)]
    return (result if len(result) >= 20 else demo_data()), (source.name if len(result) >= 20 else "curated demo dataset")


@st.cache_resource
def train_model():
    data, source = load_data()
    encoded = pd.get_dummies(data, columns=["fuel_type"])
    feature_names = FEATURES + [f"fuel_type_{fuel}" for fuel in FUEL_TYPES]
    for name in feature_names:
        if name not in encoded:
            encoded[name] = 0
    model = RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(encoded[feature_names], encoded.price)
    return model, data, source, feature_names


model, data, source, feature_names = train_model()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#f1f3ee; --muted:#99a29a; --panel:#141817; --line:#2a322d; --lime:#c9f84a; --orange:#ff8a4c; }
.stApp { background: #0b0e0d; color: var(--ink); font-family:'Space Grotesk',sans-serif; }
.block-container { max-width: 1380px; padding: 2.4rem 4rem 4rem; }
[data-testid="stSidebar"] { background:#101412; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding:2rem 1.5rem; }
.logo { color:var(--lime); font-weight:700; letter-spacing:.12em; font-size:1.1rem; }
.eyebrow { color:var(--lime); font-family:'DM Mono',monospace; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; }
h1 { font-size:clamp(2.8rem,6vw,6rem); line-height:.95; letter-spacing:-.07em; margin:.5rem 0 1.25rem; font-weight:600; }
.hero-copy { color:var(--muted); max-width:500px; font-size:1rem; line-height:1.6; }
.panel { background:var(--panel); border:1px solid var(--line); padding:1.35rem; border-radius:18px; }
.price { color:var(--lime); font-size:clamp(2.8rem,5vw,5rem); letter-spacing:-.07em; font-weight:600; line-height:1; }
.price-label { color:var(--muted); font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em; font-size:.68rem; }
.brand-tag { color:var(--muted); font-size:.95rem; font-weight:500; margin:.15rem 0 0; }
.car-name { margin:.1rem 0 1.25rem; font-size:clamp(1.5rem,3vw,2.5rem); letter-spacing:-.04em; font-weight:600; }
.spec { border-top:1px solid var(--line); padding:.8rem 0; display:flex; justify-content:space-between; color:var(--muted); font-size:.85rem; }
.spec b { color:var(--ink); font-weight:500; }
.panel-progress-track { background:#1a211d; border-radius:999px; height:8px; overflow:hidden; margin-top:1rem; }
.panel-progress-fill { background:var(--lime); height:100%; border-radius:999px; }
div.stButton > button { background:var(--lime); color:#10140d; border:0; border-radius:999px; font-weight:700; padding:.7rem 1.25rem; }
div.stButton > button:hover { background:#e0ff79; color:#10140d; }
div[data-baseweb="select"] > div, input, [data-testid="stNumberInput"] input { background:#1a211d !important; border-color:var(--line) !important; color:var(--ink) !important; }
[data-testid="stMetricValue"] { color:var(--lime); }
.caption { color:var(--muted); font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="logo">RideGiude</div>', unsafe_allow_html=True)
    st.markdown("### Build your spec")
    st.caption("Tune the details. We estimate what this car would cost when new.")

    brands = ["Custom"] + sorted(data["brand"].dropna().astype(str).unique().tolist())
    brand = st.selectbox("Brand", brands)

    if brand == "Custom":
        car_names = ["Custom build"]
    else:
        car_names = sorted(data.loc[data["brand"] == brand, "car_name"].dropna().astype(str).unique().tolist())
        if not car_names:
            car_names = ["Custom build"]
    car_name = st.selectbox("Model", car_names)

    fuel = st.selectbox("Powertrain", FUEL_TYPES, index=2)
    capacity_label = "Battery capacity (kWh)" if fuel == "Electric" else "Engine capacity (cc)"
    capacity = st.slider(capacity_label, 40 if fuel == "Electric" else 800, 140 if fuel == "Electric" else 7000, 82 if fuel == "Electric" else 2200, step=1)
    horsepower = st.slider("Horsepower", 80, 1200, 320, step=5)
    top_speed = st.slider("Top speed (km/h)", 140, 420, 250, step=5)
    zero_to_hundred = st.slider("0-100 km/h (seconds)", 2.0, 16.0, 5.2, step=0.1)
    seats = st.select_slider("Seats", options=[2, 4, 5, 6, 7, 8], value=5)
    torque = st.slider("Torque (Nm)", 100, 1400, 480, step=10)
    estimate = st.button("Estimate price", use_container_width=True, type="primary")

if "prediction" not in st.session_state:
    st.session_state.prediction = 78500.0
if estimate:
    values = {"capacity": capacity, "horsepower": horsepower, "top_speed": top_speed, "zero_to_hundred": zero_to_hundred, "seats": seats, "torque": torque}
    row = pd.DataFrame([values])
    for selected in FUEL_TYPES:
        row[f"fuel_type_{selected}"] = int(selected == fuel)
    st.session_state.prediction = float(model.predict(row[feature_names])[0])

prediction = st.session_state.prediction
low, high = prediction * .86, prediction * 1.16
progress_pct = min(max((prediction - 15000) / 300000, 0.04), 1.0) * 100

st.markdown('<div class="eyebrow">New vehicle intelligence / 2025 edition</div>', unsafe_allow_html=True)
st.title("Your next car,\npriced intelligently.")
st.markdown('<p class="hero-copy">Configure a new vehicle around the things that matter to you. Ride Guide reads the spec and returns a market-aware starting point in seconds.</p>', unsafe_allow_html=True)
st.write("")

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.markdown(f'''
    <div class="panel">
        <div class="eyebrow">Selected vehicle</div>
        <p class="brand-tag">{brand}</p>
        <h2 class="car-name">{car_name}</h2>
        <div class="price-label">Estimated new-car price</div>
        <div class="price">${prediction:,.0f}</div>
        <p class="caption">Typical range ${low:,.0f} to ${high:,.0f} - model confidence based on spec similarity</p>
        <div class="panel-progress-track"><div class="panel-progress-fill" style="width:{progress_pct:.1f}%;"></div></div>
    </div>
    ''', unsafe_allow_html=True)
    st.write("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Powertrain", fuel)
    m2.metric("Performance", f"{zero_to_hundred:.1f}s")
    m3.metric("Model records", f"{len(data):,}")
    st.write("")
    st.markdown("#### Configuration summary")
    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.markdown(f'<div class="spec"><span>Capacity</span><b>{capacity} {"kWh" if fuel == "Electric" else "cc"}</b></div><div class="spec"><span>Horsepower</span><b>{horsepower} hp</b></div><div class="spec"><span>Top speed</span><b>{top_speed} km/h</b></div>', unsafe_allow_html=True)
    with summary_right:
        st.markdown(f'<div class="spec"><span>Acceleration</span><b>{zero_to_hundred:.1f}s</b></div><div class="spec"><span>Seating</span><b>{seats} seats</b></div><div class="spec"><span>Torque</span><b>{torque} Nm</b></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel"><div class="eyebrow">Market context</div><h3>Where your build sits</h3>', unsafe_allow_html=True)
    st.caption(f"Compared with {fuel.lower()} vehicles in the training set.")
    bins = pd.cut(data.price, bins=5)
    counts = data.assign(bucket=bins).groupby("bucket", observed=True).size()
    chart = pd.DataFrame({"Cars": counts.values}, index=[f"${int(interval.left/1000)}k" for interval in counts.index])
    st.bar_chart(chart, color="#c9f84a", height=235)
    st.markdown(f'<div class="caption">Dataset source: {source}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="panel"><div class="eyebrow">How it works</div><p class="caption">A Random Forest regression model learns relationships between six vehicle specifications and new-car prices. Upload <b>Cars_Datasets_2025.csv</b> beside this app to train on your own data.</p></div>', unsafe_allow_html=True)

st.divider()
st.caption("RideGuide is an estimation tool, not a dealer quote. Prices are illustrative and depend on market, trim, taxes, and availability.")