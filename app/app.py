import re
import streamlit as st
import geopandas as gpd
import pydeck as pdk
import numpy as np
import pandas as pd
import zipfile, tempfile, shutil
from pathlib import Path

st.set_page_config(page_title="Charleston Airport Avg Speeds — Compare ToD", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    # gdf = gpd.read_file(r"../processed/charleston_airport_combined_timesets.geojson")
    zip_path = Path("../processed/charleston_airport_combined_timesets.zip")
    inner = "charleston_airport_combined_timesets.geojson"

    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            with z.open(inner) as fsrc, tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as fdst:
                shutil.copyfileobj(fsrc, fdst)
                tmp_path = fdst.name
        gdf = gpd.read_file(tmp_path)
    else:
        gdf = gpd.read_file("../processed/charleston_airport_combined_timesets.geojson")
    # include frc for filtering
    keep_cols = ["avg_speed", "timeSet_name", "dow", "streetName", "frc", "geometry"]
    have = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[have].copy()

    # numeric cleanup
    if "avg_speed" in gdf.columns:
        gdf["avg_speed"] = pd.to_numeric(gdf["avg_speed"], errors="coerce")
    if "frc" in gdf.columns:
        gdf["frc"] = pd.to_numeric(gdf["frc"], errors="coerce").astype("Int64")

    # geometry → path for pydeck
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["path"] = gdf.geometry.apply(
        lambda geom: [[x, y] for x, y in zip(*geom.xy)] if geom.geom_type == "LineString" else None
    )
    gdf = gdf[gdf["path"].notna()].copy()

    # integer tooltip
    gdf["avg_speed_int"] = np.rint(gdf["avg_speed"]).astype("Int64")
    return gdf

gdf = load_data()

# --- Helper: sort ToD labels chronologically ---
def tod_sort_key(label: str):
    m = re.match(r"^\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)\s*$", str(label))
    if not m:
        return 99999
    h, m1, _, _ = map(int, m.groups())
    return h * 60 + m1

# --- Sidebar filters ---
st.sidebar.header("Filters")
dow = st.sidebar.selectbox("Day of Week", sorted(gdf["dow"].dropna().unique()))

tod_options = sorted(gdf["timeSet_name"].dropna().unique(), key=tod_sort_key)
left_default  = 0
right_default = min(1, len(tod_options)-1)
tod_left  = st.sidebar.selectbox("Time of Day (Left map)", tod_options, index=left_default)
tod_right = st.sidebar.selectbox("Time of Day (Right map)", tod_options, index=right_default)

# FRC multiselect (if available)
if "frc" in gdf.columns:
    all_frcs = sorted([int(v) for v in gdf["frc"].dropna().unique()])
    frc_selected = st.sidebar.multiselect(
        "Functional Class (FRC)",
        options=all_frcs,
        default=all_frcs,
        help="Select one or more roadway functional classes"
    )
else:
    frc_selected = None

# --- Apply filters ---
def apply_filters(df, dow_val, tod_val, frc_vals):
    q = (df["dow"] == dow_val) & (df["timeSet_name"] == tod_val)
    if frc_vals is not None and len(frc_vals) > 0 and "frc" in df.columns:
        q &= df["frc"].isin(frc_vals)
    out = df[q].copy()
    return out

left_df  = apply_filters(gdf, dow, tod_left, frc_selected)
right_df = apply_filters(gdf, dow, tod_right, frc_selected)

# --- Shared center ---
if len(left_df) or len(right_df):
    combo = left_df if len(left_df) else right_df
    center = [combo.geometry.centroid.x.mean(), combo.geometry.centroid.y.mean()]
else:
    center = [-80.037, 32.898]
view_state = pdk.ViewState(latitude=center[1], longitude=center[0], zoom=12, pitch=0)

# --- Fixed discrete color bins for avg_speed ---
BINS = [0, 10, 20, 30, 40, 50, np.inf]  # 6 bins
BIN_LABELS = ["0–10", "10–20", "20–30", "30–40", "40–50", "50+"]
PALETTE = [  # red → green
    [178, 24, 43],   # 0–10
    [239, 59, 44],   # 10–20
    [253, 141, 60],  # 20–30
    [255, 217, 47],  # 30–40
    [144, 238, 144], # 40–50
    [0, 128, 0],     # 50+
]

def colorize(df):
    if df.empty:
        df["color"] = []
        return df
    cats = pd.cut(df["avg_speed"], bins=BINS, right=False, labels=False)
    cols = []
    for i in cats:
        if pd.isna(i):
            cols.append([180, 180, 180])
        else:
            idx = int(i)
            idx = max(0, min(idx, len(PALETTE)-1))
            cols.append(PALETTE[idx])
    df["color"] = cols
    return df

left_df  = colorize(left_df)
right_df = colorize(right_df)

def make_deck(df):
    if df.empty:
        return None
    layer = pdk.Layer(
        "PathLayer",
        data=df,
        get_path="path",
        get_color="color",
        get_width=3,            # meters
        width_min_pixels=2,     # fixed 2px thickness
        pickable=True,
    )
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Street: {streetName}\nAvg Speed: {avg_speed_int} mph"},
        map_provider="carto",
        map_style="light",
    )

st.write(f"### {dow} — Compare {tod_left} vs {tod_right}"
         + (f" • FRC: {', '.join(map(str, frc_selected))}" if frc_selected else ""))

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(f"**{tod_left}** — {len(left_df)} segments")
    deck_left = make_deck(left_df)
    if deck_left:
        st.pydeck_chart(deck_left)
    else:
        st.info("No segments for this selection.")

with col2:
    st.markdown(f"**{tod_right}** — {len(right_df)} segments")
    deck_right = make_deck(right_df)
    if deck_right:
        st.pydeck_chart(deck_right)
    else:
        st.info("No segments for this selection.")

# --- Legend for fixed bins (no leading spaces so Markdown won't make code blocks) ---
legend_items = []
for label, col in zip(BIN_LABELS, PALETTE):
    legend_items.append(
        f"<div style='display:flex;align-items:center;gap:6px;'>"
        f"<div style='width:36px;height:10px;background:rgb({col[0]},{col[1]},{col[2]});border:1px solid #999;'></div>"
        f"<span style='font-size:12px'>{label} mph</span>"
        f"</div>"
    )

legend_html = (
    "<div style='display:flex;gap:16px;align-items:center;flex-wrap:wrap;'>"
    + "".join(legend_items) +
    "</div>"
)

st.markdown("**Legend (avg_speed mph — fixed bins)**", unsafe_allow_html=True)
st.markdown(legend_html, unsafe_allow_html=True)

