import networkx as nx
import osmnx as ox
import osmnx_func as osmfunc

cities = [#"Oslo", 
          #"Bergen", 
          #"Trondheim",
          # "Helsinki",
          # "Vancouver", 
            "New York", 
          # "Portland", 
            "Washington"]

for city in cities:
    print(f"Downloading: {city}")
    osmfunc.get_plublic_transport(city)