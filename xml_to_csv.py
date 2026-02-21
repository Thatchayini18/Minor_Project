import xml.etree.ElementTree as ET
import pandas as pd

# Load XML
tree = ET.parse("smartcity.xml")
root = tree.getroot()

data = []

# -------------------------
# WEATHER DATA
# -------------------------
weather = root.find(".//environment/weather/current")

temperature = weather.findtext("temperature")
humidity = weather.findtext("humidity")

# -------------------------
# AIR QUALITY DATA (MULTIPLE SENSORS)
# -------------------------
for sensor in root.findall(".//environment/airQuality/sensor"):
    pm25 = sensor.findtext("pm25")
    pm10 = sensor.findtext("pm10")
    no2 = sensor.findtext("no2")
    zone = sensor.get("zone")
    quality = sensor.findtext("qualityIndex")

    data.append([
        zone,
        temperature,
        humidity,
        pm25,
        pm10,
        no2,
        quality
    ])

# Create DataFrame
columns = [
    "zone",
    "temperature",
    "humidity",
    "pm2_5",
    "pm10",
    "no2",
    "air_quality_level"
]

df = pd.DataFrame(data, columns=columns)

# Save CSV
df.to_csv("data/raw/environment_data.csv", index=False)

print("✅ XML converted to CSV successfully!")
