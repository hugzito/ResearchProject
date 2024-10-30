import os
import re
import json
import pyproj
import overpass
import networkx as nx
from tqdm import tqdm
from pyproj import Transformer
from shapely.ops import transform
from shapely import get_coordinates
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

class CleanGraph():
    def __init__(self, where: dict, osm_type:str, nx_type:str) -> None:
        self.G = None
        self.city = where['city']
        self.osm_type = osm_type
        self.nx_type = nx_type
    
    def clean_graph(self):
        self.load_city_graph()
        print(len(self.G.nodes))
        self.network_pruning(self.city, self.G)
        self.city_to_files()


    def get_box(self):
        """
        Downloads a bounding box for the city
        """
        api = overpass.API()

        if self.city.lower() == 'oslo':
            box = api.get("""rel[admin_level=4][name="Oslo"]; out geom;""", responseformat="json")
        elif self.city.lower() == 'bergen':
            box = api.get("""rel[admin_level=7][name="Bergen"]; out geom;""", responseformat="json")
        elif self.city.lower() == 'helsinki':
            box = api.get("""rel[admin_level=8][name="Helsinki"]; out geom;""", responseformat="json")
        elif 'new york' in self.city.lower():
            box = api.get("""rel[admin_level=5][name="City of New York"]; out geom;""", responseformat="json")
        elif 'washington' in self.city.lower():
            box = api.get("""rel[admin_level=6][name="Washington"]; out geom;""", responseformat="json")
        elif self.city.lower() == 'portland':
            box = api.get("""rel[admin_level=8][name="Portland"]["is_in:state" = Oregon]; out geom;""", responseformat="json")
        elif self.city.lower() == 'vancouver':
            box = api.get("""rel[admin_level=8][name="Vancouver"]; out geom;""", responseformat="json") 
        elif self.city.lower() == 'trondheim':
            box = api.get("""rel[admin_level=7][name="Trondheim"]; out geom;""", responseformat="json") 

        coord = box['elements'][0]['bounds']

        box = Polygon([(coord['maxlon'],coord['minlat']),
                    (coord['maxlon'],coord['maxlat']),
                    (coord['minlon'], coord['maxlat']),
                    (coord['minlon'], coord['minlat']),
                    ])
        return box

    def project_coords(self, element):
        """
        Reprojects the geometry
        """
        frm = pyproj.CRS('EPSG:4326')
        to = pyproj.CRS('EPSG:3857')

        project = pyproj.Transformer.from_crs(frm, to, always_xy=True).transform
        return transform(project, element)

    def clean_edge_attr(self, attr):
        """
        This function cleans the edge attributes
        """
        new_attr = {}

        #Get way attributes
        if 'attr_dict' in attr.keys():
            way_attrs = attr['attr_dict']
            way_ids = attr['osmid']
            way_ids = [str(i) for i in way_ids]
            
            #Get all attributes for the individual ways
            all_keys = []
            for i in way_ids:
                all_keys += list(way_attrs[i].keys())
            all_keys = set(all_keys)

            #Merge keys
            for key in all_keys:
                values = set()
                for way in way_ids:
                    if key in list(way_attrs[way].keys()):
                        values.add(way_attrs[way][key])
                values = list(values)
                
                if len(values) == 1:
                    new_attr[key] = values[0]
                else:
                    new_attr[key] = values
        else:
            for key, values in attr.items():
                new_attr['osmid'] = attr['osmid']
                new_attr['length'] = attr['length']
        
        #Project geometries
        if 'geometry' in attr.keys():
            line = attr['geometry']
            new_attr['geometry'] = self.project_coords(line) 
        
        return new_attr

    def network_pruning(self, city: str, G: nx.graph):
        """
        This function cleans the network nodes, 
        edges and their attributes
        """
        box = self.get_box()
        box = self.project_coords(box)

        pruned_G = nx.MultiDiGraph()

        nodes_to_keep = []
        nodes_data = []
        for node in tqdm(G.nodes(data= True)):
            node_id = node[0]
            node_attr = node[1]
            #Create point
            node_point = Point(node_attr['x'],node_attr['y'])
            node_point = self.project_coords(node_point)
            node_attr['geometry'] = node_point
            #Check if within box
            if box.contains(node_point):
                nodes_data.append((node_id, node_attr))
                nodes_to_keep.append(node_id)
                pruned_G.add_node(node_id, **node_attr)
        
        #Remove edges
        for edges in tqdm(G.edges(data = True)):
            frm = edges[0]
            to = edges[1]
            if frm in nodes_to_keep and to in nodes_to_keep:
                attr = edges[2]
                new_attr = self.clean_edge_attr(attr)
                pruned_G.add_edge(u_for_edge = frm, v_for_edge = to, **new_attr)

        self.G = pruned_G

    def make_geom_to_string(self, geom):
        geom_type = geom.geom_type
        #geom = self.project_coords(geom)

        if geom_type != 'MultiPolygon':
            geom = get_coordinates(geom).tolist()
        else:
            multi = []
            for i in range(len(geom.geoms)):
                multi.append(get_coordinates(geom.geoms[i]).tolist())
            geom = multi
        
        return geom, geom_type

    def list_coord_to_geo(self, coord, geom_type):
        if geom_type == 'Point':
            return Point(coord)
        elif geom_type == 'LineString':
            return LineString(coord)
        elif geom_type == 'Polygon':
            return Polygon(coord)
        elif geom_type == 'MultiPolygon':
            polys = [Polygon(i) for i in coord]
            return MultiPolygon(polys)
        else:
            print(geom_type, 'is unknown')

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
        city = self.city.lower().replace(',', '').replace('.','').replace(' ', '_')
        G = nx.MultiDiGraph()

        #Add the nodes
        file = open(f"data/{city}/{city}_{self.osm_type}_node_att.txt", "r")
        while True:
            content=file.readline()
            if not content:
                break
            d = content.split(' ', 1)
            attr = re.sub("((?<!{)(?<!,\s)(?<!: ))\"((?!:)(?!,\s))", "", d[1]).replace(': None', ': ""').replace("\\", '')
            attr = json.loads(attr)
            if 'geometry' in attr.keys():
                attr['geometry'] = self.list_coord_to_geo(attr['geometry'], attr['geom_type'])
            G.add_node(int(d[0]), **attr)
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
                attr['geometry'] = self.list_coord_to_geo(attr['geometry'], attr['geom_type'])
            G.add_edge(int(d[0]), int(d[1]), **attr)
        file.close()

        self.G = G

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
        if not os.path.exists(f"data/{city}/clean"):
            os.chdir(f"data/{city}")
            os.makedirs("clean")
            os.chdir('..')
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
        for e in tqdm(G.edges(data = True)):
            att = e[2]
            if 'geometry' in att.keys():
                #Make geometry to a list
                att['geometry'], att['geom_type'] = self.make_geom_to_string(att['geometry'])
            else: 
                continue #Skip id no geometry

            lines.append(f"{e[0]} {e[1]} {json.dumps(att, ensure_ascii=False)}\n")
        
        with open(f'data/{city}/clean/{city}_{osm_type}_{nx_type}_edge.txt', "w", encoding= 'utf8') as file:
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
        lines = []

        for n in tqdm(G.nodes(data = True)):
            attr = n[1]
            attr['geometry'], attr['geom_type'] = self.make_geom_to_string(attr['geometry'])
            lines.append(f"{n[0]} {attr}\n".replace("'", '"'))
        
        with open(f'data/{city}/clean/{city}_{osm_type}_node_att.txt', "w", encoding= 'utf8') as file:
            file.writelines(lines)
            file.close()

