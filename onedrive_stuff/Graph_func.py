import json
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely import get_coordinates
from shapely.geometry import Point, LineString, Polygon, MultiPolygon


def load_city_graph(city: str, osm_type:str, clean = False, deep_clean = False):
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

    base_path = f"data/{city}/"
    if clean:
        base_path = f"data/{city}/clean/"
    elif deep_clean:
        base_path = f"data/{city}/deep_clean/"
    
    #Add the nodes
    file = open(base_path + f"{city}_{osm_type}_node_att.txt", "r")
    print(base_path + f"{city}_{osm_type}_node_att.txt")
    while True:
        content=file.readline()
        if not content:
            break
        d = content.split(' ', 1)
        attr = json.loads(d[1])
        attr['geometry'] = Point(attr['geometry'])
        G.add_node(int(d[0]), **attr)
    file.close()
        
    #Add the edges
    file = open(base_path + f'{city}_{osm_type}_multidigraph_edge.txt', "r")
    while True:
        content=file.readline()
        if not content:
            break
        d = content.split(' ', 2)
        attr = json.loads(d[2])
        if 'geometry' in attr.keys():
            attr['geometry'] = list_coord_to_geo(attr['geometry'], attr['geom_type'])
        G.add_edge(int(d[0]), int(d[1]), **attr)
    file.close()

    return G

def load_city_amenities(city, clean = False):
    """
    Loads either the raw data or the compresed data
    """
    city = city.lower()
    if not clean:
        return gpd.read_file(f"data/{city}/{city}_amenities.geojson")
    else:
        data = pd.read_json(f'data/{city}/clean/clean_{city}_amenities.json')
        data = gpd.GeoDataFrame(data)

        data['geometry'] = data.apply(lambda row: list_coord_to_geo(row['geometry'], row['geom_type']), axis = 1)
        data = data.set_geometry('geometry')
        data = data.set_crs('EPSG:3857')
        return data

def make_geom_to_string(geom):
    geom_type = geom.geom_type

    if geom_type != 'MultiPolygon':
        geom = get_coordinates(geom).tolist()
    else:
        multi = []
        for i in range(len(geom.geoms)):
            multi.append(get_coordinates(geom.geoms[i]).tolist())
        geom = multi
    
    return geom, geom_type

def list_coord_to_geo(coord, geom_type):
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

################ OTHER FUNCTIONS #####################

def graph_to_dataframe(G:nx.Graph):
    data = []

    #Get nodes
    for node in G.nodes(data = True):  
        node_data = {}
        if node[1] != {}:
            node_data['from'] = node[0]
            node_data['geometry'] = node[1]['geometry']
            data.append(node_data)

    #Get edges
    for edge in G.edges(data = True):
        edge_data = edge[2]
        edge_data['from'] = edge[0]
        edge_data['to'] = edge[1]
        data.append(edge_data)

    data = gpd.GeoDataFrame(data, geometry= 'geometry', crs = "EPSG:3857")
    return data