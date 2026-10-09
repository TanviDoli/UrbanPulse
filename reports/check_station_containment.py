import json
from shapely.geometry import shape, Point

with open('data/raw/chennai_zones_2025.geojson', 'r') as f:
    gj = json.load(f)

zones = []
for feat in gj['features']:
    zones.append({
        'name': feat['properties'].get('name_of_th'),
        'geom': shape(feat['geometry'])
    })

# Known official CAAQMS coordinates for Chennai
stations = [
    {'name': 'Alandur Bus Depot', 'file': 'alandur-2024-25.csv', 'lat': 12.9972, 'lon': 80.2005},
    {'name': 'Arumbakkam', 'file': 'arumbakkam-2024-25.csv', 'lat': 13.0640, 'lon': 80.2078},
    {'name': 'Gandhi Nagar, Ennore', 'file': 'gandhi-nagar-2024-25.csv', 'lat': 13.2081, 'lon': 80.3204},
    {'name': 'Kodungaiyur', 'file': 'kodungaiyur-2024-25.csv', 'lat': 13.1367, 'lon': 80.2458},
    {'name': 'Manali', 'file': 'manali-2024-25.csv', 'lat': 13.1656, 'lon': 80.2588},
    {'name': 'Manali Village', 'file': 'manali-village-2024-25.csv', 'lat': 13.1709, 'lon': 80.2642},
    {'name': 'Perungudi', 'file': 'perungudi-2024-25.csv', 'lat': 12.9660, 'lon': 80.2452},
    {'name': 'Royapuram', 'file': 'royapuram-2024-25.csv', 'lat': 13.1118, 'lon': 80.2941},
    {'name': 'Velachery Res. Area', 'file': 'velachery-2024-25.csv', 'lat': 12.9786, 'lon': 80.2185}
]

for st in stations:
    pt = Point(st['lon'], st['lat'])
    matched_zone = None
    min_dist = float('inf')
    closest_zone = None
    for z in zones:
        if z['geom'].contains(pt):
            matched_zone = z['name']
            break
        d = z['geom'].distance(pt)
        if d < min_dist:
            min_dist = d
            closest_zone = z['name']
            
    print(f"Station: {st['name']:25s} -> Point: ({st['lat']:.4f}, {st['lon']:.4f}) | In Zone: {matched_zone} (Closest: {closest_zone}, dist={min_dist:.5f})")
