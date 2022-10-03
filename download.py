import config
import json
import urllib.request
import requests
import os
from pathlib import Path

# Replace with your API key
api_key = config.api_key

country = "JP"
# Path(os.path.join("pics/train", country)).mkdir(parents=True, exist_ok=True)
# Path(os.path.join("pics/val", country)).mkdir(parents=True, exist_ok=True)
Path(os.path.join("pics/test", country)).mkdir(parents=True, exist_ok=True)

n_pics = 10
while n_pics > 0:
  # Get random location
  resp = requests.get(f"https://api.3geonames.org/?randomland={country}&json=1")
  location = resp.json()
  print([ location['major'][data] for data in ['city', 'prov'] ])

  # Construct urls, filenames
  lat, lng = location['nearest']['latt'], location['nearest']['longt']
  pic_url = f"https://maps.googleapis.com/maps/api/streetview?size=300x300&source=outdoor&radius=1000&location={lat},{lng}&key={api_key}"
  meta_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?source=outdoor&radius=1000&location={lat},{lng}&key={api_key}"

  # Get Google image metadata
  with urllib.request.urlopen(meta_url) as response:
    metadata = json.loads(response.read())
  print(metadata['status'])

  # Save if pic exists
  if metadata['status'] == 'OK':
    # filename = os.path.join("pics/train", country, f"{country}_{metadata['location']['lat']}_{metadata['location']['lng']}.jpg")
    # filename = os.path.join("pics/val", country, f"{country}_{metadata['location']['lat']}_{metadata['location']['lng']}.jpg")
    filename = os.path.join("pics/test", country, f"{country}_{metadata['location']['lat']}_{metadata['location']['lng']}.jpg")
    with urllib.request.urlopen(pic_url) as response, open(filename, "wb") as file:
      file.write(response.read())
    n_pics-=1

  print()