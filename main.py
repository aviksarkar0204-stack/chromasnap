import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Page config
st.set_page_config(
    page_title="Color Palette Extractor",
    page_icon="🎨",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0e0e10;
    color: #f0ede8;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    letter-spacing: -0.03em;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #1a1a1f;
    border: 1.5px dashed #3a3a45;
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7c6fff;
}

/* Slider */
.stSlider > div > div > div {
    background: #7c6fff !important;
}

/* Metric cards */
.color-card {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
.color-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.6);
}
.color-swatch {
    width: 100%;
    height: 120px;
}
.color-info {
    background: #1a1a1f;
    padding: 10px 12px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    line-height: 1.7;
    color: #c0bdb8;
}
.color-info .pct {
    font-size: 22px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    color: #f0ede8;
    display: block;
    margin-bottom: 2px;
}
.color-info .hex {
    color: #7c6fff;
    font-weight: 400;
}

/* Progress bar strip */
.palette-strip {
    display: flex;
    border-radius: 12px;
    overflow: hidden;
    height: 24px;
    margin: 1.5rem 0 0.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.strip-seg {
    height: 100%;
    transition: flex 0.4s;
}

/* Section title */
.section-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #606070;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("## Color Palette Extractor")
st.markdown("<p style='color:#606070; margin-top:-12px; font-size:15px;'>Upload any image to extract its dominant color palette using K-Means clustering.</p>", unsafe_allow_html=True)
st.divider()

# Controls
n_colors = st.slider("Number of colors to extract", min_value=2, max_value=10, value=5)

uploaded_file = st.file_uploader("Drop an image here", type=["png", "jpg", "jpeg", "webp", "bmp"])

# Processing
def rgb_to_hex(r, g, b):
    return f"#{r:02X}{g:02X}{b:02X}"

def extract_palette(image_array, n_clusters):
    pixels = image_array.reshape(-1, 3).astype(np.float32)
    # Downsample for speed on large images
    if len(pixels) > 50000:
        idx = np.random.choice(len(pixels), 50000, replace=False)
        pixels = pixels[idx]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    total = len(labels)
    percentages = [np.sum(labels == i) / total * 100 for i in range(n_clusters)]
    # Sort by percentage descending
    order = np.argsort(percentages)[::-1]
    return colors[order], np.array(percentages)[order]

if uploaded_file:
    # Load image
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)

    col_img, col_info = st.columns([1, 1], gap="large")
    with col_img:
        st.markdown("<p class='section-label'>Original Image</p>", unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)

    with st.spinner("Analysing colors…"):
        colors, percentages = extract_palette(image_rgb, n_colors)

    # Palette strip
    st.markdown("<p class='section-label' style='margin-top:1.2rem;'>Palette Distribution</p>", unsafe_allow_html=True)
    strip_html = '<div class="palette-strip">'
    for color, pct in zip(colors, percentages):
        hex_c = rgb_to_hex(*color)
        strip_html += f'<div class="strip-seg" style="flex:{pct:.2f}; background:{hex_c};"></div>'
    strip_html += "</div>"
    st.markdown(strip_html, unsafe_allow_html=True)

    # Color cards
    st.markdown("<p class='section-label' style='margin-top:1.4rem;'>Dominant Colors</p>", unsafe_allow_html=True)
    cols = st.columns(min(n_colors, 5))
    for i, (color, pct) in enumerate(zip(colors, percentages)):
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        hex_c = rgb_to_hex(r, g, b)
        with cols[i % min(n_colors, 5)]:
            st.markdown(f"""
            <div class="color-card">
                <div class="color-swatch" style="background:{hex_c};"></div>
                <div class="color-info">
                    <span class="pct">{pct:.1f}%</span>
                    <span class="hex">{hex_c}</span><br>
                    rgb({r}, {g}, {b})
                </div>
            </div>
            """, unsafe_allow_html=True)

    #Data table
    with st.expander("📋 Raw data"):
        import pandas as pd
        df = pd.DataFrame({
            "Rank": range(1, len(colors)+1),
            "Hex": [rgb_to_hex(*c) for c in colors],
            "R": [int(c[0]) for c in colors],
            "G": [int(c[1]) for c in colors],
            "B": [int(c[2]) for c in colors],
            "Coverage %": [f"{p:.2f}%" for p in percentages],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style='text-align:center; padding: 3rem 1rem; color:#404050;'>
        <div style='font-size:48px; margin-bottom:1rem;'>🖼️</div>
        <p style='font-family: DM Mono, monospace; font-size:13px;'>Upload a PNG, JPG, or WEBP to get started</p>
    </div>
    """, unsafe_allow_html=True)