import os
import json
import math
import requests
import overpass
import osmnx as ox
import pandas as pd
import networkx as nx
from tqdm import tqdm
from time import sleep
import geopandas as gpd
from pyproj import Transformer
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString


def download_city_graph(where: dict, network_type:str):
    """
    where: dict -> dict containing the placem fx. {"city": "Portland", "state": "Oregon", "country": "USA"} 
    network_type:str -> type of network, e.g "bike"
    _______________
    Returns networkx graph
    """
    #Get the network
    G = ox.graph_from_place(where, network_type=network_type)
    #Add edge speeds
    print('Getting edge speeds')
    G = ox.routing.add_edge_speeds(G)
    #Add travel time based on speed limits
    print('Getting edge traveltime')
    G = ox.routing.add_edge_travel_times(G)
    #Add elevation
    print('Getting elevation')
    G = get_elevation(G)

    return G

def get_elevation(G:nx.graph):
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

def city_to_files(G: nx.graph, city: str, osm_type:str, nx_type:str):
    """
    Saves the graph as three files:
        1. One containing edge geometries (geojson)
        2. One for the edges and other attributes (txt)
        3. One for the nodes and their attributes (txt)
    __________
    G: nx.graph -> graph to save
    city: str -> city of the graph
    osm_type:str -> type of network, e.g "bike"
    nx_type:str -> networkx type, 
    """
    city = city.lower().replace(',', '').replace('.','').replace(' ', '_')
    #Check if folder exists
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(f"data/{city}"):
        os.chdir(f"data")
        os.makedirs(city)
        os.chdir('..')
    
    city_edgelist_to_txt(G, city, osm_type, nx_type)
    city_node_att_to_txt(G, city, osm_type)

def city_edgelist_to_txt(G: nx.graph, city: str, osm_type:str, nx_type:str):
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
    geometries = []
    lines = []
    for edge in G.edges():
        att = G[edge[0]][edge[1]][0]
        if 'geometry' in att.keys():
            geometries.append([edge[0], edge[1], att["osmid"], att['geometry']])
            del att['geometry']
        lines.append(f"{edge[0]} {edge[1]} {json.dumps(att,ensure_ascii=False)}\n")
    
    with open(f'data/{city}/{city}_{osm_type}_{nx_type}_edge.txt', "w", encoding= 'utf8') as file:
        file.writelines(lines)
        file.close()
    
    #Save geomitries 
    geometries = pd.DataFrame(data = geometries, columns = ['from', 'to', 'osmid', 'geometry'])
    geometries = gpd.GeoDataFrame(data = geometries, geometry="geometry")
    geometries.to_file(f'data/{city}/{city}_{osm_type}_edge_geometry.geojson', driver="GeoJSON")  

def city_node_att_to_txt(G: nx.graph, city: str, osm_type:str):
    """
    Saves the nodes as one files:
        3. One for the nodes and their attributes (txt)
    __________
    G: nx.graph -> graph to save
    city: str -> city of the graph
    osm_type:str -> type of network, e.g "bike"
    """
    with open(f'data/{city}/{city}_{osm_type}_node_att.txt', "w", encoding= 'utf8') as file:
        lines = [f"{str(n)} {G.nodes()[n]}\n".replace("'", '"') for n in G.nodes()]
        file.writelines(lines)
        file.close()

def load_city_graph(city: str, osm_type:str, geometry:bool = False):
    """
    Loads the graph into the original downloaded form
    __________
    city: str -> city of the graph
    osm_type:str -> type of network, e.g "bike"
    geometry:bool -> to include geometries or not
    __________
    retrun networkx graph
    """
    city = city.lower().replace(',', '').replace('.','').replace(' ', '_')
    G = nx.MultiDiGraph()

    #Add the nodes
    file = open(f"data/{city}/{city}_{osm_type}_node_att.txt", "r")
    while True:
        content=file.readline()
        if not content:
            break
        d = content.split(' ', 1)
        G.add_nodes_from([(int(d[0]), json.loads(d[1]))])
    file.close()

    #Add the edges
    file = open(f'data/{city}/{city}_{osm_type}_multidigraph_edge.txt', "r")
    while True:
        content=file.readline()
        if not content:
            break
        d = content.split(' ', 2)
        G.add_edges_from([(int(d[0]), int(d[1]), json.loads(d[2]))])
    file.close()

    #Add geometries
    if geometry:
        geo_data = gpd.read_file(f'data/{city}/{city}_{osm_type}_edge_geometry.geojson')
        for idx, row in geo_data.iterrows():
            G[row['from']][row['to']][0]['geometry'] = row['geometry']

    return G

def download_city_amenities(where: dict):
    """
    Downloads amenities of a city and saves it as a geojson
    _________
    where:dict -> where: dict -> dict containing the placem fx. {"city": "Portland", "state": "Oregon", "country": "USA"}
    _________
    return geopandas dataframe
    """
    city = where['city'].lower().replace(',', '').replace('.','').replace(' ', '_')
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(f"data/{city}"):
        os.chdir(f"data")
        os.makedirs(city)
        os.chdir('..')   
    amenities = ox.features.features_from_place(where, tags = {'amenity':True})
    amenities.to_file(f'data/{city}/{city}_amenities.geojson', driver="GeoJSON")
    return amenities


######### Download Public Transport ###############

def download_osm_transit_data(transport, city):
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

def get_meta_from_tree(tree, osm_type):
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

def get_nodes(tree):
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

def get_way_order(tree):
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

def check_stations(node_order, way_graphs):
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

def get_meta(tree):
    nodes_meta = get_meta_from_tree(tree = tree, osm_type='node')
    way_meta = get_meta_from_tree(tree = tree, osm_type='way')
    meta_data = pd.DataFrame(nodes_meta+way_meta)
    meta_data = meta_data.drop_duplicates('id')
    meta_data.index = meta_data.id
    meta_data = meta_data.to_dict(orient = 'index')
    return meta_data

def clean_meta_dict(d, keep_coord = True):
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

def get_osmid_from_shortest_path(G, path):
    ids = []
    for idx in range(1, len(path)-2): #to exclude the stations
        att = G[path[idx]][path[idx+1]]['attr']
        ids.append(att['osm_id'])
    return list(set(ids))

def line_lenght(line):
    # Transformer to convert from WGS84 to EPSG:3857 (meters)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    projected_line = LineString([transformer.transform(*coord) for coord in line.coords])
    length_in_meters = projected_line.length
    return length_in_meters

def get_components_ends(component):
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



def create_network(tree, network = nx.MultiDiGraph()):
    node_order = get_nodes(tree)
    way_graphs = get_way_order(tree)
    meta_data = get_meta(tree)

    node_order = check_stations(node_order, way_graphs)
    data = []

    great_graph = network
    #Plot graph
    for rel_id in tqdm(way_graphs.keys(), desc= 'Buiding Lines'):
        G = way_graphs[rel_id]

        #Check if multiple components and fix if is
        """
        if nx.is_connected(G) != True:
            components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
            components_points = {}
            for i in range(len(components)): 
                start, end = get_components_ends(component = components[i])
                components_points[start] = i
                components_points[end] = i
            
            #Matches
            matches = []
            #Used nodes
            used = []
            #Match the points to the closets one from another component

            for key_one in components_points.keys():
                if key_one not in used:
                    dist = 1_000_000
                    temp_match = None
                    #Point 1
                    p = Point(key_one[0], key_one[1]) 
                    p_com = components_points[key_one]
                    for key_two in components_points.keys():
                        if key_two not in used and components_points[key_two] != p_com:
                            p_two = Point(key_two[0], key_two[1])
                            if p.distance(p_two) < dist:
                                temp_match = key_two
                                dist = p.distance(p_two)
                
                    if temp_match != None:
                        used.append(key_one) #No need to look at these anymore
                        used.append(temp_match) #No need to look at these anymore
                        matches.append([key_one, temp_match])

            G.add_edges_from(matches)

        if nx.is_connected(G) != True:
            print(f'Relation {rel_id} is connected')
        else:
            print(f'Relation {rel_id} it not connected with {nx.number_connected_components(G)}, however these edges were added: \n {matches}')
        """
        nodes = node_order[rel_id]
        for n_idx in range(len(nodes)-1):
            u = nodes[n_idx]
            u_meta = clean_meta_dict(meta_data[u[1]])

            v = nodes[n_idx+1]
            v_meta = clean_meta_dict(meta_data[v[1]])

            great_graph.add_nodes_from([(u[1], u_meta)])
            great_graph.add_nodes_from([(v[1], v_meta)])
            
            data.append([u, None, Point(u[0][0], u[0][1])])
            data.append([v, None, Point(u[0][0], u[0][1])])
            
            try:
                if u[0] != v[0]:
                    shortest_path = nx.shortest_path(G, u[0], v[0])
                    osm_ids = get_osmid_from_shortest_path(G, shortest_path)
                    attr_dict = {}
                    for i in osm_ids:
                        attr_dict[int(i)] = clean_meta_dict(meta_data[i], keep_coord= False)
                        del attr_dict[i]['id']
                    line = LineString(shortest_path)
                    data.append([u, v, line])
                    great_graph.add_edge(u_for_edge = u[1], v_for_edge = v[1], osmid = osm_ids, geometry = line, 
                                                                                        length= line_lenght(line),
                                                                                        attr_dict = attr_dict
                                                                                        )
            except:
                ...
                #print(f'Failed to find a way from {u} to {v} in relation {rel_id}')
                     
    return great_graph, gpd.GeoDataFrame(data, columns = ['from', 'to', 'geometry'])

def get_plublic_transport(city):
    network = nx.MultiDiGraph()

    for tran in ['subway', 'light_rail', 'bus', 'tram']:
        tree = download_osm_transit_data(tran, city)
        if len(tree.findall('relation')) > 0:
            network, dataframe = create_network(tree = tree, network = network)
        print(f"After adding {tran} to the network, we have {network.number_of_nodes()} nodes")
    
    if city == 'New York':
        city = 'New York City'
    elif city == 'Washington':
        city = "Washington, D.C."
        
    city_to_files(G = network, city = city, osm_type = 'public_transport', nx_type = 'multidigraph')