import json

with open('data/raw/chennai_zones_2025.geojson') as f:
    gj = json.load(f)

print(f"Total features: {len(gj['features'])}")
for i, feat in enumerate(gj['features']):
    props = feat['properties']
    geom = feat['geometry']
    coords = geom['coordinates']
    # Polygon or MultiPolygon
    if geom['type'] == 'Polygon':
        poly = coords[0]
    elif geom['type'] == 'MultiPolygon':
        poly = coords[0][0]
    lons = [pt[0] for pt in poly]
    lats = [pt[1] for pt in poly]
    c_lon = sum(lons) / len(lons)
    c_lat = sum(lats) / len(lats)
    print(f"Zone {i+1:02d}: Name = {props.get('name_of_th'):20s} | Centroid: ({c_lat:.4f} N, {c_lon:.4f} E) | Vertices: {len(poly)}")
