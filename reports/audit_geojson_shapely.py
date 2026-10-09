import json
from shapely.geometry import shape, Polygon, MultiPolygon

with open('data/raw/chennai_zones_2025.geojson', 'r') as f:
    gj = json.load(f)

zones = []
for feat in gj['features']:
    geom = shape(feat['geometry'])
    name = feat['properties'].get('name_of_th')
    if not geom.is_valid:
        geom = geom.buffer(0)
    zones.append({'name': name, 'geom': geom, 'props': feat['properties']})

print(f"Loaded {len(zones)} zones:")
for i, z in enumerate(zones):
    bounds = z['geom'].bounds
    print(f"{i+1:2d}. {z['name']:18s} | Valid: {z['geom'].is_valid} | Area: {z['geom'].area:.6f} deg^2 | Bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f}]")

# Check adjacency
print("\nAdjacency matrix check (touches or intersects):")
adj_count = 0
for i in range(len(zones)):
    for j in range(i + 1, len(zones)):
        inter = zones[i]['geom'].intersection(zones[j]['geom'])
        # shared boundary has length > 0
        if inter.length > 0:
            adj_count += 1
            print(f"  {zones[i]['name']:18s} <---> {zones[j]['name']:18s} (shared len: {inter.length:.5f} deg)")

print(f"\nTotal neighboring pairs with shared boundary: {adj_count}")
