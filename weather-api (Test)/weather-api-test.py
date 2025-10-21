import requests
import folium
from folium import CustomIcon

# 🔑 Your API keys
MAPTILER_API_KEY = "wnjhmnWHMYXVQRVYj8rN"   # Already provided by you
WEATHERAPI_KEY = "0a90412fb5a549d6b34141436250503"      # Replace with your WeatherAPI.com key

# 🌍 Location - Example: London
latitude = 51.5074
longitude = -0.1276

# === 1. Fetch Weather Data from WeatherAPI.com ===
weather_url = (
    f"http://api.weatherapi.com/v1/current.json"
    f"?key={WEATHERAPI_KEY}&q={latitude},{longitude}&aqi=no"
)

response = requests.get(weather_url)
weather_data = response.json()

# Extract weather data
condition_text = weather_data['current']['condition']['text']
temperature = weather_data['current']['temp_c']
icon_url = "https:" + weather_data['current']['condition']['icon']

# === 2. Create Map using MapTiler (contours style) ===
tiles_url = f"https://api.maptiler.com/maps/contours/256/{{z}}/{{x}}/{{y}}.png?key=wnjhmnWHMYXVQRVYj8rN"

# Create map centered at the chosen location
m = folium.Map(location=[latitude, longitude], zoom_start=10, tiles=tiles_url, attr='MapTiler')

# === 3. Add Weather Marker ===
popup_text = f"Weather: {condition_text}, {temperature}°C"

# Use WeatherAPI's weather icon
weather_icon = CustomIcon(icon_url, icon_size=(50, 50))

folium.Marker(
    [latitude, longitude],
    icon=weather_icon,
    popup=popup_text
).add_to(m)

# === 4. Save Map to HTML ===
m.save("map_with_weatherapi.html")

print("✅ Map with weather saved as 'map_with_weatherapi.html'")
