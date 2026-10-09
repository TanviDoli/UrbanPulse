"""
Spatial Graph Construction for UrbanPulse.
Builds the 15-zone Chennai adjacency graph using GCC 2025 GeoJSON boundaries.
Outputs data/processed/zone_adjacency.csv with shared boundary lengths.
"""

import json
import os
import pandas as pd
from shapely.geometry import shape

# Official GCC Zone numbering convention
GCC_ZONE_MAPPING = {
    'THIRUVOTTRIYUR': (1, 'Thiruvottiyur'),
    'MANALI': (2, 'Manali'),
    'MADHAVARAM': (3, 'Madhavaram'),
    'TONDIARPET': (4, 'Tondiarpet'),
    'ROYAPURAM': (5, 'Royapuram'),
    'THIRU-VIKA-NAGAR': (6, 'Thiru-Vi-Ka Nagar'),
    'AMBATTUR': (7, 'Ambattur'),
    'ANNA NAGAR': (8, 'Anna Nagar'),
    'TEYNAMPET': (9, 'Teynampet'),
    'KODAMBAKKAM': (10, 'Kodambakkam'),
    'VALASARVAKKAM': (11, 'Valasaravakkam'),
    'ALANDUR': (12, 'Alandur'),
    'ADYAR': (13, 'Adyar'),
    'PERUNKUDI': (14, 'Perungudi'),
    'SHOZHANGANALLUR': (15, 'Sholinganallur')
}

def build_zone_adjacency(geojson_path: str = 'data/raw/chennai_zones_2025.geojson',
                         output_csv: str = 'data/processed/zone_adjacency.csv') -> pd.DataFrame:
    with open(geojson_path, 'r') as f:
        gj = json.load(f)
        
    zones = []
    for feat in gj['features']:
        raw_name = feat['properties'].get('name_of_th')
        geom = shape(feat['geometry'])
        if not geom.is_valid:
            geom = geom.buffer(0)
            
        zone_id, clean_label = GCC_ZONE_MAPPING.get(raw_name, (None, raw_name))
        zones.append({
            'zone_id': zone_id,
            'zone_label': clean_label,
            'raw_name': raw_name,
            'geom': geom
        })
        
    zones.sort(key=lambda z: z['zone_id'])
    
    rows = []
    n = len(zones)
    
    for i in range(n):
        for j in range(i + 1, n):
            z1 = zones[i]
            z2 = zones[j]
            
            inter = z1['geom'].intersection(z2['geom'])
            # Shared boundary length in degrees
            boundary_len = float(inter.length)
            
            is_adj = 1 if boundary_len > 1e-6 else 0
            
            rows.append({
                'zone_id_1': z1['zone_id'],
                'zone_label_1': z1['zone_label'],
                'zone_id_2': z2['zone_id'],
                'zone_label_2': z2['zone_label'],
                'shared_boundary_length': round(boundary_len, 6),
                'adjacent': is_adj
            })
            
    df_adj = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_adj.to_csv(output_csv, index=False)
    
    adj_edges = df_adj[df_adj['adjacent'] == 1]
    print(f"Generated {output_csv} with {len(df_adj)} total pairs, {len(adj_edges)} adjacent neighbors.")
    return df_adj

if __name__ == '__main__':
    build_zone_adjacency()
