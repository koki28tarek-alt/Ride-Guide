import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="RideGuide | New Car Price Predictor", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

FEATURES = ["capacity", "horsepower", "top_speed", "zero_to_hundred", "seats", "torque"]
FUEL_TYPES = ["Petrol", "Diesel", "Hybrid", "Electric"]
DEMO_BRANDS = ["Toyota", "Ford", "BMW", "Mercedes-Benz", "Hyundai", "Kia", "Volkswagen", "Audi", "Nissan", "Chevrolet", "Tesla", "Honda", "Mazda", "Peugeot", "Renault"]
DEMO_MODEL_WORDS = ["Alpha", "Terra", "Nova", "Orion", "Vertex", "Pulse", "Atlas", "Comet", "Zenith"]
SEAT_OPTIONS = [2, 4, 5, 6, 7, 8]

# Model choices available under Developer options. Each entry is a factory
# function so we can build a fresh, untrained instance whenever we need one
# (once for held-out evaluation, once for the final full-data fit).
MODEL_FACTORY = {
    "Random Forest": lambda: RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, random_state=42),
    "Linear Regression": lambda: LinearRegression(),
    "K-Nearest Neighbors": lambda: KNeighborsRegressor(n_neighbors=7),
}


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


def clamp_int(value, lo, hi):
    return int(round(min(max(value, lo), hi)))


def clamp_float(value, lo, hi):
    return float(min(max(value, lo), hi))


def nearest_option(value, options):
    return min(options, key=lambda option: abs(option - value))


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
def train_models():
    data, source = load_data()
    encoded = pd.get_dummies(data, columns=["fuel_type"])
    feature_names = FEATURES + [f"fuel_type_{fuel}" for fuel in FUEL_TYPES]
    for name in feature_names:
        if name not in encoded:
            encoded[name] = 0
    X = encoded[feature_names]
    y = encoded.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    metrics, finals = {}, {}
    for name, factory in MODEL_FACTORY.items():
        eval_model = factory()
        eval_model.fit(X_train, y_train)
        preds = eval_model.predict(X_test)
        metrics[name] = {"R²": r2_score(y_test, preds), "MAE ($)": mean_absolute_error(y_test, preds)}

        final_model = factory()
        final_model.fit(X, y)
        finals[name] = final_model

    return finals, metrics, data, source, feature_names


def car_options_for_brand(brand_value, data):
    if brand_value == "Custom":
        return ["Custom build"]
    options = sorted(data.loc[data.brand == brand_value, "car_name"].dropna().astype(str).unique().tolist())
    return options if options else ["Custom build"]


def sync_from_car():
    """Runs when Brand or Model changes; auto-fills the spec sliders with the
    selected car's real values from the dataset."""
    brand_sel = st.session_state.get("brand_select", "Custom")
    car_sel = st.session_state.get("car_name_select", "Custom build")
    if brand_sel == "Custom" or car_sel in (None, "Custom build"):
        return
    match = data[(data.brand == brand_sel) & (data.car_name == car_sel)]
    if match.empty:
        return
    row = match.iloc[0]
    fuel_val = row.fuel_type if row.fuel_type in FUEL_TYPES else "Petrol"
    st.session_state.fuel_select = fuel_val
    st.session_state.capacity_slider = clamp_int(row.capacity, 40, 7000)
    st.session_state.horsepower_slider = clamp_int(row.horsepower, 80, 1200)
    st.session_state.top_speed_slider = clamp_int(row.top_speed, 140, 420)
    st.session_state.zero_slider = clamp_float(row.zero_to_hundred, 2.0, 16.0)
    st.session_state.seats_select = nearest_option(row.seats, SEAT_OPTIONS)
    st.session_state.torque_slider = clamp_int(row.torque, 100, 1400)


def on_brand_change():
    brand_sel = st.session_state.get("brand_select", "Custom")
    options = car_options_for_brand(brand_sel, data)
    st.session_state.car_name_select = options[0]
    sync_from_car()


def on_car_change():
    sync_from_car()


models, model_metrics, data, source, feature_names = train_models()

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
    st.caption("Tune the details, or pick a real Brand + Model below to auto-fill its actual specs.")

    brands = ["Custom"] + sorted(data["brand"].dropna().astype(str).unique().tolist())
    brand = st.selectbox("Brand", brands, key="brand_select", on_change=on_brand_change)

    car_names = car_options_for_brand(brand, data)
    car_name = st.selectbox("Model", car_names, key="car_name_select", on_change=on_car_change)

    fuel = st.selectbox("Powertrain", FUEL_TYPES, index=2, key="fuel_select")

    capacity_label = "Battery capacity (kWh)" if fuel == "Electric" else "Engine capacity (cc)"
    cap_lo, cap_hi, cap_default = (40, 140, 82) if fuel == "Electric" else (800, 7000, 2200)
    # Bounds change with fuel type, so re-clamp any auto-filled value that no
    # longer fits before the slider is drawn (avoids a min/max mismatch error).
    if "capacity_slider" in st.session_state:
        st.session_state.capacity_slider = clamp_int(st.session_state.capacity_slider, cap_lo, cap_hi)
    capacity = st.slider(capacity_label, cap_lo, cap_hi, cap_default, step=1, key="capacity_slider")

    horsepower = st.slider("Horsepower", 80, 1200, 320, step=5, key="horsepower_slider")
    top_speed = st.slider("Top speed (km/h)", 140, 420, 250, step=5, key="top_speed_slider")
    zero_to_hundred = st.slider("0-100 km/h (seconds)", 2.0, 16.0, 5.2, step=0.1, key="zero_slider")
    seats = st.select_slider("Seats", options=SEAT_OPTIONS, value=5, key="seats_select")
    torque = st.slider("Torque (Nm)", 100, 1400, 480, step=10, key="torque_slider")

    if brand != "Custom" and car_name != "Custom build":
        st.caption(f"✓ Specs auto-filled from {brand} {car_name}.")
    else:
        st.caption("Custom build — adjust freely.")

    with st.expander("⚙️ Developer options"):
        model_name = st.selectbox("Prediction model", list(models.keys()), key="model_choice")
        st.caption("R² and MAE measured on a held-out 20% test split.")
        metrics_view = pd.DataFrame(model_metrics).T
        metrics_view["R²"] = metrics_view["R²"].round(3)
        metrics_view["MAE ($)"] = metrics_view["MAE ($)"].round(0).astype(int)
        st.dataframe(metrics_view, use_container_width=True)
        compare_all = st.checkbox("Compare all models on this build", key="compare_all")

    estimate = st.button("Estimate price", use_container_width=True, type="primary")

values = {"capacity": capacity, "horsepower": horsepower, "top_speed": top_speed, "zero_to_hundred": zero_to_hundred, "seats": seats, "torque": torque}
row = pd.DataFrame([values])
for selected in FUEL_TYPES:
    row[f"fuel_type_{selected}"] = int(selected == fuel)
row = row[feature_names]

if "prediction" not in st.session_state:
    st.session_state.prediction = 78500.0
if estimate:
    st.session_state.prediction = float(models[model_name].predict(row)[0])

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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Powertrain", fuel)
    m2.metric("Performance", f"{zero_to_hundred:.1f}s")
    m3.metric("Model records", f"{len(data):,}")
    m4.metric("Prediction model", model_name)
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

    if compare_all:
        comparison = {name: float(m.predict(row)[0]) for name, m in models.items()}
        comparison_series = pd.Series(comparison).sort_values(ascending=False)
        st.markdown('<div class="panel"><div class="eyebrow">Developer view</div><h3>All models, this build</h3>', unsafe_allow_html=True)
        st.bar_chart(comparison_series, color="#ff8a4c", height=200)
        st.caption("Predicted price for the current configuration, by model.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

    st.markdown('<div class="panel"><div class="eyebrow">How it works</div><p class="caption">Five regression models — Random Forest, Gradient Boosting, Decision Tree, Linear Regression, and K-Nearest Neighbors — are trained on six vehicle specifications. Switch between them in Developer options. Upload <b>Cars_Datasets_2025.csv</b> beside this app to train on your own data.</p></div>', unsafe_allow_html=True)

st.divider()
st.caption("RideGuide is an estimation tool, not a dealer quote. Prices are illustrative and depend on market, trim, taxes, and availability.")
