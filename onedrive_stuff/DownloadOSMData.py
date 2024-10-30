import os
import re
import json
import math
import requests
import osmnx as ox
import pandas as pd
import networkx as nx
from tqdm import tqdm
from time import sleep
import geopandas as gpd
from shapely import get_coordinates
from shapely.geometry import Point, LineString, Polygon, MultiPolygon


class DownloadOSMData():
    def __init__(self, where: dict, osm_type:str, nx_type:str, elevation = True) -> None:
        self.G = None
        self.city = where['city']
        self.where = where
        self.osm_type = osm_type
        self.elevation = elevation
        self.nx_type = nx_type
        print(self.where, self.osm_type)
    
    def download(self):
        self.G = self.download_city_graph()

    def save_network(self):
        self.city_to_files()

    def download_city_graph(self):
        """
        where: dict -> dict containing the placem fx. {"city": "Portland", "state": "Oregon", "country": "USA"} 
        osm_type:str -> type of network, e.g "bike"
        _______________
        Returns networkx graph
        """
        #Get the network
        G = ox.graph_from_place(query = self.where, network_type = self.osm_type)
        #Add edge speeds
        print('Getting edge speeds')
        G = ox.routing.add_edge_speeds(G)
        #Add travel time based on speed limits
        print('Getting edge traveltime')
        G = ox.routing.add_edge_travel_times(G)
        if self.elevation:
            #Add elevation
            print('Getting elevation')
            G = self.get_elevation(G)
        return G

    def get_elevation(self, G:nx.graph):
        """
        Iterrates the graph nodes and adds the elevation for every node.
        G:nx.graph -> networkx graph
        __________
        returns networkx graph
        """
        locations = [(data['y'], data['x']) for node, data in G.nodes(data=True)]
        node_list = [node for node in G.nodes()]

        end = len(locations)
        step = 100
        for i in tqdm(range(0, end, step)):
            #Download the next x
            x = i
            test = "|".join([f"{lat},{lon}" for lat, lon in locations[x:x+step]])
            url_template=f"https://api.opentopodata.org/v1/eudem25m?locations={test}" #lat, lon
            response = requests.get(url_template)
            elevation_list = response.json()['results']

            #Slide node_list to get corresponding nodes
            temp_nodes = node_list[x:x+step]
            #Add elevation to nodes
            for idx in range(len(temp_nodes)):
                node = node_list[idx]
                G.nodes()[node]['elevation'] = elevation_list[idx]['elevation']
            sleep(1)
            
        return G

    def city_to_files(self):
        """
        Saves the graph as three files:
            1. One containing edge geometries (geojson)
            2. One for the edges and other attributes (txt)
            3. One for the nodes and their attributes (txt)
        __________
        G: nx.graph -> graph to save
        city: str -> city of the graph
        osm_type:str -> type of network, e.g "bike"
        nx_type:str -> networkx tsype, 
        """
        city = self.city.lower().replace(',', '').replace('.','').replace(' ', '_')
        
        #Check if folder exists
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(f"data/{city}"):
            os.chdir(f"data")
            os.makedirs(city)
            os.chdir('..')
        
        self.city_edgelist_to_txt(self.G, city, self.osm_type, self.nx_type)
        self.city_node_att_to_txt(self.G, city, self.osm_type)

    def city_edgelist_to_txt(self, G: nx.graph, city: str, osm_type:str, nx_type:str):
        """
        Saves the edge data as two files:
            1. One containing edge geometries (geojson)
            2. One for the edges and other attributes (txt)
        __________
        G: nx.graph -> graph to save
        city: str -> city of the graph
        osm_type:str -> type of network, e.g "bike"
        nx_type:str -> networkx type, 
        """
        lines = []
        for e in G.edges(data = True):
            att = e[2]
            if 'geometry' in att.keys():
                #Make geometry to a list
                att['geometry'], att['geom_type'] = self.make_geom_to_string(att['geometry'])

            lines.append(f"{e[0]} {e[1]} {json.dumps(att, ensure_ascii=False)}\n")
        
        with open(f'data/{city}/{city}_{osm_type}_{nx_type}_edge.txt', "w", encoding= 'utf8') as file:
            file.writelines(lines)
            file.close()
    
    def city_node_att_to_txt(self, G: nx.graph, city: str, osm_type:str):
        """
        Saves the nodes as one files:
            3. One for the nodes and their attributes (txt)
        __________
        G: nx.graph -> graph to save
        city: str -> city of the graph
        osm_type:str -> type of network, e.g "bike"
        """

        with open(f'data/{city}/{city}_{osm_type}_node_att.txt', "w", encoding= 'utf8') as file:
            lines = [f"{n} {G.nodes()[n]}\n".replace("'", '"') for n in G.nodes()]
            file.writelines(lines)
            file.close()

    def load_city_graph(self):
        """
        Loads the graph into the original downloaded form
        __________
        city: str -> city of the graph
        osm_type:str -> type of network, e.g "bike"
        geometry:bool -> to include geometries or not
        __________
        retrun networkx graph
        """
        city = self.city
        city = city.lower().replace(',', '').replace('.','').replace(' ', '_')
        G = nx.MultiDiGraph()

        #Add the nodes
        file = open(f"data/{city}/{city}_{self.osm_type}_node_att.txt", "r")
        while True:
            content=file.readline()
            if not content:
                break
            d = content.split(' ', 1)
            attr = re.sub("((?<!{)(?<!,\s)(?<!: ))\"((?!:)(?!,\s))", "", d[1]).replace(': None', ': ""')
            G.add_node(int(d[0]), **json.loads(attr))
        file.close()
            
        #Add the edges
        file = open(f'data/{city}/{city}_{self.osm_type}_multidigraph_edge.txt', "r")
        while True:
            content=file.readline()
            if not content:
                break
            d = content.split(' ', 2)
            attr = json.loads(d[2])
            if 'geometry' in attr.keys():
                attr['geometry'] = get_coordinates
            G.add_edge(int(d[0]), int(d[1]), **attr)
        file.close()

        return G

    def download_city_amenities(self):
        """
        Downloads amenities of a city and saves it as a geojson
        _________
        where:dict -> where: dict -> dict containing the placem fx. {"city": "Portland", "state": "Oregon", "country": "USA"}
        _________
        return geopandas dataframe
        """
        city = self.city.lower().replace(',', '').replace('.','').replace(' ', '_')
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(f"data/{city}"):
            os.chdir(f"data")
            os.makedirs(city)
            os.chdir('..')
    
        amenities = ox.features.features_from_place(self.where, tags = {'amenity':True})
        print(amenities.shape)
        #Make into dict
        amenities_dict = amenities.to_dict(orient= 'index')

        #Make data compact
        compact_ameniti_data = []
        for idx in amenities_dict.keys():
            new_dict = {}
            for key, value in amenities_dict[idx].items():
                if value != None:
                    if type(value) == float and math.isnan(value):
                        pass
                    else:
                        if key == 'geometry':
                            value, geom_type = self.make_geom_to_string(value)
                            new_dict['geom_type'] = geom_type
                        new_dict[key] = value
            compact_ameniti_data.append(new_dict)

        with open(f"data/{city}/{city}_amenities.json", "w", encoding= 'utf8') as final:
            json.dump(compact_ameniti_data, final)
    
    def make_geom_to_string(self, geom):
        geom_type = geom.geom_type
        
        if geom_type != 'MultiPolygon':
            geom = get_coordinates(geom).tolist()
        else:
            multi = []
            for i in range(len(geom.geoms)):
                multi.append(get_coordinates(geom.geoms[i]).tolist())
            geom = multi
        
        return geom, geom_type