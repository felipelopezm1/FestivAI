import streamlit as st
from pathlib import Path
from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser
import streamlit.components.v1 as components

import time
# UI
# Page setup
st.set_page_config(page_title="ClimaSign Prod", layout="wide", initial_sidebar_state="expanded")
st.title("Climate and Sign Language Deducer")
st.markdown("#### Show a sign to the camera:")

# Layout with 2 columns: left for camera input, right for animation display
col1, col2 = st.columns([1, 1])

with col1:
    # Camera input section
    st.markdown("##### 📸 Camera Input")
    camera_image = st.camera_input("Capture your Sign")

with col2:
    st.markdown("##### 🌍 Weather Map")

    # Embed the map HTML directly into col2
    map_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
        <style>
            html, body {
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
            }
            #map {
                width: 100%;
                height: 400px;
            }
        </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css" />
        <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([51.5074, -0.1276], 10);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; OpenStreetMap contributors'
                }).addTo(map);

            var marker = L.marker([51.5074, -0.1276]).addTo(map);
            var icon = L.icon({
                iconUrl: 'https://cdn.weatherapi.com/weather/64x64/day/113.png',
                iconSize: [50, 50]
            });
            marker.setIcon(icon);
            marker.bindPopup("Weather: Sunny, 14.2°C").openPopup();
        </script>
    </body>
    </html>
    """

    # Inject map into col2
    components.html(map_html, height=400) #size of the embed


# Process the image if captured
if camera_image is not None:
    with st.status(":red[Processing captured image.]", expanded=True) as status:
        st.write(":orange[Analyzing Captured Image...]") #take out

        # Save to temp file and analyze
        temp_file_path = create_temp_file(camera_image)

        # Example call to your image analysis function (update model/prompt as needed)
        stream = analyze_image_file(temp_file_path, model='llava:7b', user_prompt="What sign is this?")

        parsed = stream_parser(stream)

        out_text = ''
        for chunk in parsed:
            out_text += chunk

        # Replace the placeholder box with actual analysis result
        col2.markdown(
            f"<div style='width:100%; height:400px; background-color:white; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;'>"
            f"<p style='text-align:center;'>{out_text}</p>"
            "</div>", unsafe_allow_html=True
        )

        status.update(label=":green[Image Processed Successfully!]", state="complete", expanded=False)

