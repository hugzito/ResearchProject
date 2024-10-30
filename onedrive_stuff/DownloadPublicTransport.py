import os
import re
import json
import math
import pyproj
import requests
import overpass
import osmnx as ox
import pandas as pd
import networkx as nx
from tqdm import tqdm
from time import sleep
import geopandas as gpd
from pyproj import Transformer
from shapely.ops import transform
import xml.etree.ElementTree as ET
from shapely import get_coordinates
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

class DownloadPublicTransport():
    def __init__(self, where: dict, nx_type:str) -> None:
        self.G = None
        self.city_download = self.check_city_name_for_download(where['city'])
        self.city_save = self.check_city_name_for_save(self.city_download)
        self.where = where
        self.nx_type = nx_type
        self.osm_type = 'public_transport'

    def check_city_name_for_download(self, city_name):
        if city_name == 'New York City':
            self.city = 'New York'
        elif city_name == "Washington, D.C.":
            self.city = 'Washington'
        else:
            self.city = city_name
    
    def check_city_name_for_save(self, city_name):
        if city_name == 'New York':
            self.city_save = 'New York City'
        elif city_name == 'Washington':
            self.city_save = "Washington, D.C."
    
    def download(self):
        network = nx.MultiDiGraph()

        for tran in ['subway', 'light_rail', 'bus', 'tram']:
            tree = self.download_osm_transit_data(tran, self.city)
            if len(tree.findall('relation')) > 0:
                network = self.create_network(tree = tree, network = network)
            print(f"After adding {tran} to the network, we have {network.number_of_nodes()} nodes")
        self.G = network

    def save_network(self):
        self.city_to_files()

    def download_osm_transit_data(self, transport, city):
        api = overpass.API()

        # fetch all ways and nodes
        if transport == 'light_rail':
            result = api.get(f"""
                            area["name"="{city}"] -> .a;
                            (
                            rel [type=route][route=light_rail][railway!=platform](area.a);
                            );
                            (._;>>;);
                            out geom;
                            >;
                            """, responseformat="xml")

        if transport == 'subway':
            result = api.get(f"""
                            area["name"="{city}"] -> .a;
                            (
                            rel [type=route][route=subway][railway!=platform](area.a);
                            );
                            (._;>>;);
                            out geom;
                            >;
                            """, responseformat="xml")

        if transport == 'bus':
            result = api.get(f"""
                            area["name"="{city}"]["boundary"="administrative"] -> .a;
                            (
                            rel(area.a)[route=bus](area.a);
                            );
                            out geom;
                            >;
                            """, responseformat="xml")

        if transport == 'tram':
            result = api.get(f"""
                            area["name"="{city}"] -> .a;
                            (
                            rel [type=route][route=tram][railway!=platform](area.a);
                            );
                            (._;>>;);
                            out geom;
                            >;
                            """, responseformat="xml")

        tree = ET.ElementTree(ET.fromstring(result))

        return tree

    def get_meta_from_tree(self, tree, osm_type):
        """
        Get the meta data from nodes and ways
        in the element tree, returns a list
        of dicts with the meta_data
        """
        dicts = []
        for element in tree.findall(osm_type):
            tags = element.findall('tag')
            temp_dict = {}
            temp_dict['id'] = int(element.get('id'))
            temp_dict['osm_type'] = osm_type
            if osm_type == 'node':
                temp_dict['lat'] = float(element.get('lat'))
                temp_dict['lon'] = float(element.get('lon'))
            for tag in tags:
                temp_dict[tag.get('k')] = tag.get('v')
            dicts.append(temp_dict)
        return dicts

    def get_nodes(self, tree):
        """
        Get all the stations of a relation,
        returns a dict with list, with a 
        stations (point, osm_id)
        """
        node_order = {} #key = rel_id, value = stations nodes
        for rel in tree.findall('relation'):
            nodes = []
            relation_id = int(rel.attrib['id'])

            #Get members of relations
            for mem in rel.findall('member'):
                #Get node ids
                if mem.attrib['type'] == 'node':
                    lon = float(mem.attrib['lon'])
                    lat = float(mem.attrib['lat'])
                    nodes.append([(lon,lat), int(mem.attrib['ref'])])
            node_order[relation_id] = nodes
        return node_order

    def get_way_order(self, tree):
        """
        Get all the ways of a relation,
        creates a network, where each node 
        is a point of the ways.
        Returns a dict with a nx.Graph of the
        ways.
        """
        rel_graph_dict = {} 
        for rel in tree.findall('relation'):
            relation_id = int(rel.attrib['id'])
            G = nx.Graph()
            #Get members of relations
            for mem in rel.findall('member'):
                #Check if it is a way
                if mem.attrib['type'] == 'way':
                    osm_id = int(mem.attrib['ref'])
                    previous_point = None
                    #Add edge in the graph
                    for point in mem.findall('nd'):
                        lon = float(point.attrib['lon'])
                        lat = float(point.attrib['lat'])
                        if previous_point == None:
                            previous_point = (lon, lat)
                        else:
                            G.add_edge(u_of_edge = previous_point, v_of_edge = (lon, lat), attr={'osm_id':osm_id})
                            #print(f"{previous_point} -> {(lon, lat)}")
                            previous_point = (lon, lat)
            rel_graph_dict[relation_id] = G
        
        return rel_graph_dict

    def check_stations(self, node_order, way_graphs):
        """
        Checks if the stations are in the graph, if 
        not it match them to the point on the line
        closest to the station
        """
        new_node_order = {}
        for rel_id in tqdm(way_graphs.keys(), desc= "Cheking stations"):
            nodes = node_order[rel_id]
            G = way_graphs[rel_id]
            graph_nodes = list(G.nodes())
            nodes_in_graph = []
            for n in nodes:
                if G.has_node(n):
                    nodes_in_graph.append(n)       
                else: #Match to point on the ways
                    closest = 1_000_000
                    close = None
                    p = Point(n[0][0], n[0][1])
                    for i in graph_nodes:
                        dist = p.distance(Point(i[0], i[1]))
                        if dist < closest:
                            closest = dist
                            close = i
                    if close != None:
                        nodes_in_graph.append((close, n[1]))
            new_node_order[rel_id] = nodes_in_graph
        return new_node_order

    def get_meta(self, tree):
        nodes_meta = self.get_meta_from_tree(tree = tree, osm_type='node')
        way_meta = self.get_meta_from_tree(tree = tree, osm_type='way')
        meta_data = pd.DataFrame(nodes_meta+way_meta)
        meta_data = meta_data.drop_duplicates('id')
        meta_data.index = meta_data.id
        meta_data = meta_data.to_dict(orient = 'index')
        return meta_data

    def clean_meta_dict(self, d, keep_coord = True):
        """
        Takes dict and removes nan values
        """
        clean_meta = d.copy()
        for key,value in d.items():
            if key == 'lat' and keep_coord:
                del clean_meta[key]
                clean_meta['y'] = value
            elif key == 'lon'and keep_coord:
                del clean_meta[key]
                clean_meta['x'] = value
            elif type(value) != str and math.isnan(value):
                del clean_meta[key]
            else:
                None 
        return clean_meta

    def get_osmid_from_shortest_path(self, G, path):
        ids = []
        for idx in range(1, len(path)-2): #to exclude the stations
            att = G[path[idx]][path[idx+1]]['attr']
            ids.append(att['osm_id'])
        return list(set(ids))

    def line_lenght(self, line):
        # Transformer to convert from WGS84 to EPSG:3857 (meters)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        projected_line = LineString([transformer.transform(*coord) for coord in line.coords])
        length_in_meters = projected_line.length
        return length_in_meters

    def get_components_ends(self, component):
        source = list(component.nodes())[0]
        dfs_tree = list(nx.dfs_tree(component, source = source).edges())
        paths = []

        while dfs_tree != []:
            path = []
            used_edges = []
            previous = None
            for i in dfs_tree:
                if previous == None or previous == i[0]:
                    used_edges.append(i)
                    previous = i[1] #Set new previous node
                    path.append(i[0]) #Add to path
                    path.append(i[1]) #Add to path
            #Remove edges from dfs_tree
            for e in used_edges:
                dfs_tree.remove(e)
            #Add path to paths
            paths.append(path)

        if len(paths) == 2:
            return paths[0][-1], paths[1][-1]
        elif len(paths) == 1:
            return paths[0][0], paths[0][-1]
        else:
            #Find the two longest comonents!!!!!!!!!!
            l = [[i, len(paths[i])] for i in range(len(paths))] #[index, len of path]
            l.sort(key= lambda x: x[1], reverse= True) 
            first = paths[l[0][0]]
            second = paths[l[1][0]]
            return first[0], second[-1]

    def create_network(self, tree, network = nx.MultiDiGraph()):
        node_order = self.get_nodes(tree)
        way_graphs = self.get_way_order(tree)
        meta_data = self.get_meta(tree)

        node_order = self.check_stations(node_order, way_graphs)

        great_graph = network
        #Plot graph
        for rel_id in tqdm(way_graphs.keys(), desc= 'Buiding Lines'):
            G = way_graphs[rel_id]
            nodes = node_order[rel_id]
            for n_idx in range(len(nodes)-1):
                u = nodes[n_idx]
                u_meta = self.clean_meta_dict(meta_data[u[1]])

                v = nodes[n_idx+1]
                v_meta = self.clean_meta_dict(meta_data[v[1]])

                great_graph.add_nodes_from([(u[1], u_meta)])
                great_graph.add_nodes_from([(v[1], v_meta)])
                
                try:
                    if u[0] != v[0]:
                        shortest_path = nx.shortest_path(G, u[0], v[0])
                        osm_ids = self.get_osmid_from_shortest_path(G, shortest_path)
                        attr_dict = {}
                        for i in osm_ids:
                            attr_dict[int(i)] = self.clean_meta_dict(meta_data[i], keep_coord= False)
                            del attr_dict[i]['id']
                        line = LineString(shortest_path)
                        great_graph.add_edge(u_for_edge = u[1], v_for_edge = v[1], osmid = osm_ids, geometry = line, 
                                                                                            length= self.line_lenght(line),
                                                                                            attr_dict = attr_dict
                                                                                            )
                except:
                    ...
                    #print(f'Failed to find a way from {u} to {v} in relation {rel_id}')
                        
        return great_graph
    
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
        city = self.city
        city = city.lower().replace(',', '').replace('.','').replace(' ', '_')
        
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
