import networkx as nx
import osmnx as ox
import osmnx_func as osmfunc

cities = [#('bike', {'city': 'Oslo', 'country': 'Norway'}),
          #('bike', {'city': 'Bergen', 'country': 'Norway'}),
          #('bike', {'city': 'Trondheim', 'country': 'Norway'}),
          #('bike', {'city': 'Helsinki', 'country': 'Finland'}),
          #('bike', {'city': 'Vancouver', 'country': 'Canada'}),
          ('bike', {"city": "New York City", "state": "New York", "country": "USA"}),
          #('bike', {"city": "Portland", "state": "Oregon", "country": "USA"}),
          ('bike', {"city": "Washington, D.C.", "country": "USA"}),
          ]

for typ, place in cities:
    print(f"Downloading: {place['city']} {typ}")
    G = osmfunc.download_city_graph(where= place, network_type= typ)
    print(f"Saving: {place['city']} {typ}")
    osmfunc.city_to_files(G = G,
                          city = place['city'], 
                          osm_type = typ,
                          nx_type = 'MultiDiGraph'.lower())


for typ, place in cities:
    print(f"Downloading amenities: {place['city']} {typ}")
    osmfunc.download_city_amenities(where = place)
