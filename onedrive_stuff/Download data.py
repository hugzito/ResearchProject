import networkx as nx
import osmnx as ox
from DownloadOSMData import DownloadOSMData
from DownloadPublicTransport import DownloadPublicTransport
from CleanGraph import CleanGraph
cities = [{'city': 'Bergen', 'country': 'Norway'},
          {'city': 'Oslo', 'country': 'Norway'},
          {'city': 'Bergen', 'country': 'Norway'},
          {'city': 'Trondheim', 'country': 'Norway'},
          {'city': 'Helsinki', 'country': 'Finland'},
          {'city': 'Vancouver', 'country': 'Canada'},
          {"city": "New York City", "state": "New York", "country": "USA"},
          {"city": "Portland", "state": "Oregon", "country": "USA"},
          {"city": "Washington, D.C.", "country": "USA"},
          ]

for place in cities:
    
    """ #Get bikelane
    print('Getting bike')
    try:
        bike_net = DownloadOSMData(where = place, osm_type = 'bike', nx_type = 'multidigraph', elevation = True)
        bike_net.download()
        bike_net.download_city_amenities()
        bike_net.save_network()
    except:
        print(f"ERROR {place} FAILED: BIKE")
    
    #Get public streets
    print('Getting public streets')
    try:
        all_public_net = DownloadOSMData(where = place, osm_type = 'all_public', nx_type = 'multidigraph', elevation = False)
        all_public_net.download()
        all_public_net.save_network()
    except:
        print(f"ERROR {place} FAILED: ALL PUBLIC STREETS")
    
    #Get publictransport
    print('Getting public transport')
    try:
        public_net = DownloadPublicTransport(where = place, nx_type = 'multidigraph')
        public_net.download()
        public_net.save_network()
    except:
        print(f"ERROR {place} FAILED: PUBLIC TRANSPORT")
    """
    #Clean graphs
    print('Clean bike')
    try:
        clean_bike = CleanGraph(where = place, osm_type = 'bike', nx_type = 'multidigraph')
        clean_bike.clean_graph()
    except:
        print(f"ERROR {place} FAILED CLEANING: BIKE")
        
    print('Clean public transport', place)
    try:
        clean_public_transport = CleanGraph(where = place, osm_type = 'public_transport', nx_type = 'multidigraph')
        clean_public_transport.clean_graph()
    except:
        print(f"ERROR {place} FAILED CLEANING: PUBLIC TRANSPORT")

    print('Clean public streets')
    try:
         clean_all_public = CleanGraph(where = place, osm_type = 'all_public', nx_type = 'multidigraph')
         clean_all_public.clean_graph()
    except:
        print(f"ERROR {place} FAILED CLEANING: ALL PUBLIC STREETS")
    

