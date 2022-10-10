import config
import json
import urllib.request
import requests
import os
from pathlib import Path

# Replace with your API key
api_key = config.api_key

countries = ["FR", "FI", "JP", "US", "BR"]
phases = ["pics120/train", "pics120/val", "pics120/test"]
sizes = [100, 30, 20]

for phase, size in zip(phases,sizes):
  print(f"{phase}, {size}")
  for country in countries:
    Path(os.path.join(phase, country)).mkdir(parents=True, exist_ok=True)
    n_pics = size
    while n_pics > 0:
      # Get random location
      resp = requests.get(f"https://api.3geonames.org/?randomland={country}&json=1")
      location = resp.json()
      print([ location['major'][data] for data in ['city', 'prov'] ])

      # Construct urls, filenames
      lat, lng = location['nearest']['latt'], location['nearest']['longt']
      hdg = (360*torch.rand(1)).round().item()
      pic_url = f"https://maps.googleapis.com/maps/api/streetview?size=600x600&source=outdoor&radius=1000&fov=120&location={lat},{lng}&heading={hdg}&key={api_key}"
      meta_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?source=outdoor&radius=1000&location={lat},{lng}&key={api_key}"

      # Get Google image metadata
      with urllib.request.urlopen(meta_url) as response:
        metadata = json.loads(response.read())
      print(metadata['status'])

      # Save if pic exists
      if metadata['status'] == 'OK':
        filename = os.path.join(phase, country, f"{country}_{metadata['pano_id']}.jpg")
        with urllib.request.urlopen(pic_url) as response, open(filename, "wb") as file:
          file.write(response.read())
          n_pics-=1

      print()