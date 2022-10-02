import config
import json
import urllib.request
import requests

# Replace with your API key
api_key = config.api_key

# Ger randon location (in JP only for now)
resp = requests.get("https://api.3geonames.org/?randomland=JP&json=1")
location = resp.json()

# Construct urls, filenames
lat, lng = location['nearest']['inlatt'], location['nearest']['inlongt']
country = 'JP'
filename = f"pics/{country}_{lat}_{lng}.jpg"
pic_url = f"https://maps.googleapis.com/maps/api/streetview?size=300x300&source=outdoor&location={lat},{lng}&key={api_key}"
meta_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?source=outdoor&location={lat},{lng}&key={api_key}"

# Get Google image metadata
with urllib.request.urlopen(meta_url) as response:
  metadata = json.loads(response.read())

# Save if pic exists
if metadata['status'] == 'OK':
  with urllib.request.urlopen(pic_url) as response, open(filename, "wb") as file:
    file.write(response.read())