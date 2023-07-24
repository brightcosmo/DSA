__author__ = "Amirul Azizol"
__version__ = "1.4"

"""
The file was created for Assignment 1, FIT2004 and submitted on 28th April 2023.

This file should contain the following:
- Functions: optimalRoute (Q1), selectSections (Q2), reverseList function
- Classes: Vertex, Edge, and Graph
"""

import heapq

def optimalRoute(start, end, passengers, roads):
    """
    This is the function outlined in question 1. The goal is to find the optimal route from one location to another.
    These locations are represented by numbers, with roads connecting the locations. However, these roads also have carpool
    lanes which may be faster to take, but these can only be taken if we have a passenger, which can be picked up at certain 
    locations. The main goal is to determine the fastest route, and whether picking up passengers can reduce the total time taken.

    Approach:
    This function first processes the input, then creates a graph. I used the "multiverse" graph approach which means creating 
    one large graph which consists of two duplicate "inner" graphs, which have the same vertices and edges, but the weight 
    for each edge is different. This represents travelling alone vs carpooling, as we will be travelling between the same
    locations but take different amounts of time. Edges are also added at passenger locations which connect these two inner graphs,
    which represent picking up a passenger.

    We can traverse from the "alone" graph to the "carpool" graph by arriving at a passenger location. This is because picking up a 
    passenger takes 0 time, and gives us access to carpool lanes. Because carpooling always is either the same or faster than travelling 
    alone for any given road, this approach will always pick up a passenger if a passenger location is passed. This is consistent
    with my real-life understanding of the problem, which is that passengers are always worth picking up if they happen to be on the way.
    
    Precondition: There is a set of locations {0, 1... |L|-1}, where start and end are elements of the set, and passengers is a subset.
    Postcondition: The list of locations returned represents the most efficient route from the source to the destination.
    
    Input:
        start: the id of the source vertex
        end: the id of the destination vertex
        passengers: a list of locations where 
        roads: A list of tuples, where each tuple represents one road. Each tuple has 4 values.
            In this order, it contains: 
                - the location where the road begins
                - the location where the road ends
                - the time taken if this road was travelled alone
                - the time taken if this road was travelled with a passenger

    Returns:
        A list of integers, which represents each location visited from the source to the destination.
        This should be the most efficient route.

    Time complexity: 
        Best case: O(R), where R is the number of roads
            - this happens when the source has its closest edge pointing to the destination, so dijkstra terminates early
        Worst case: O(R log L), where R is the number of roads and L is the total number of locations
    
    Space complexity: 
        Input: O(P + L + R)
        Aux: O(L + R)
        - L is the number of locations
        - R is the number of roads
        - P is the number of passengers
    """

    # find the highest location, from every location given
    max_location = 0
    for road in roads:
        max_location = max(road[0], road[1]) if max(road[0], road[1]) > max_location else max_location
    
    # graph will create more than max_location vertices to account for carpooling and alone times
    # offset each vertex by the size, i.e carpool vertices start directly after the last alone vertex
    graph = Graph(max_location)    
    offset = max_location+1        

    # O(R) to loop through every road (start, end, time_alone, time_carpool) and add edges
    # offset the location for carpool vertices
    for road in roads:
        graph.add_edge(road[0], road[1], road[2])
        graph.add_edge(road[0]+offset, road[1]+offset, road[3])

    # O(P) to add an edge at each passenger location (alone v -> carpool v) which takes 0 time
    for num in passengers:
        graph.add_edge(num, num+offset, 0)

    # run dijkstra on the start and end locations (this saves the two vertices in graph object)   
    graph.dijkstra(start, end)

    # process the vertices to get the final list of roads
    return graph.process_roads()


def select_sections(input):
    """
    This is the function outlined in question 2. The input to this function is a matrix of n rows and m columns. 
    Each area has a value which represents the occupancy in that area. For each row, one area must be chosen, 
    but they must be adjacent either vertically or diagonally. This function attempts to find a suitable selection 
    for every row which minimizes the final occupancy. 

    Approach:
    The approach used here is a bottom-up approach using a memory array and a decision array. For every given location after
    the first row, we will find the minimum possible occupancy in the previous row that this location can access 
    (i.e 2-3 values in the same column and adjacent columns in the previous row). After identifying the minimum, we 
    increment the value of the occupancy in this position by adding the minimum to the memory array, and saving the 
    index in the decision array. The memory array will show a list of possibilities for the minimum selection in any given row.
    
    This will slowly build up to a full solution, as each row builds upon the results of the last. Once the final row is
    reached, we can simply find the minimum and its index. From here, we find the corresponding index in the decision array.
    Since the decision array saves the adjacent/same column's minimum in the previous row, we can follow this to continuously 
    get the correct result for the previous row, until we reach the first row. This will allow us to obtain the full list 
    of locations. 

    Precondition: A matrix of n*m values is input, where n > m. Each value is an integer between 0 and 100 inclusive.
    Postcondition: The minimum total occupancy based on the selection criteria is returned, as well as a list of locations
                    that lead to this minimum.

    Input:
        occupancy_probability: A matrix of size n*m, where each row n has m elements showing the current 
                            occupancy in that area.
    Return:
        A list with two items:
            minimum_total_occupancy: an integer representing the minimum total occupancy, i.e the sum
                                    of the locations selected from each row
            sections_location: a list of (n, m) coordinates, showing all the locations selected

    Time complexity: 
        Best and worst case: O(nm), where n is the rows and m is the columns of the matrix input
    
    Space complexity: 
        Input and Aux: O(nm), where n is the rows and m is the columns of the matrix input
    """
    # get number of rows (n) and columns (m)
    n = len(input)
    m = len(input[0])

    # initialize a matrix of None based on n and m
    memo = [[None for _ in range(m)] for _ in range(n)]
    decision_array = [[None for _ in range(m)] for _ in range(n)]

    # copy over the first row of the input matrix to memory
    memo[0] = (input[0])

    # O(n) to go through every row  
    for i in range(1, n):
        # O(m) to go through every element
        for j in range(m):
            
            # get adjacent columns (2-3 values between start and end)
            start = max(0, j-1)
            end = min(m, j+2)

            # find minimum value and its corresponding index among adjacent/same column
            min_value = float('inf')
            min_index = None
            for k in range(start, end):
                if memo[i-1][k] < min_value:
                    min_value = memo[i-1][k]
                    min_index = k
            
            # add the previous row's adjacent/same column minimum to the input's number
            # save this value and the index we chose for the previous row's adjacent minimum
            memo[i][j] = input[i][j] + min_value
            decision_array[i][j] = min_index

    # O(m) to go through the final row and obtain the minimum and its index within this row
    minimum_total_occupancy = float('inf')
    row_index = None
    for i in range(m):
        if memo[-1][i] < minimum_total_occupancy:
            minimum_total_occupancy = memo[-1][i]
            row_index = i
    
    # start from the index of the minimum obtained, going upwards
    # O(n) to go through each row of the decision array
    sections_location = []
    for i in range(n-1, -1, -1):
        sections_location.append((i, row_index))
        current_row = decision_array[i]
        row_index = current_row[row_index]

    # reverse the list to get the order from top to bottom
    reverseList(sections_location)

    return [minimum_total_occupancy, sections_location]


def reverseList(lst):
    """
    A helper function which uses symmetry of indices to reverse any given list (Complexity O(n))
    """
    for i in range(len(lst) // 2):
        lst[i], lst[-i-1] = lst[-i-1], lst[i]


class Vertex:
    """
    This is a vertex, used to represent a location in the graph.
    
    Attributes:
    id (int): A unique identifier.
    edges (List[int]): A list of the edges that contain the destination and weight for each edge,
    distance (float | int): The distance between this vertex and a given source v.
    previous (Vertex): The previous vertex in the dijkstra's shortest path.

    Reference: This implementation is based on the approach outlined in the FIT2004 Week 4 lectures.

    discovered (bool): True if the vertex is seen in another vertex's edge in dijkstra
    visited (bool): True if dijkstra reaches this edge and is currently checking its edges.
    carpool (bool): True if this is an alternate vertex for carpooling.
    """
    def __init__(self, id):
        self.id = id
        self.edges = []
        self.distance = float('inf')
        self.previous = None 
        
        self.discovered = False
        self.visited = False 
        self.carpool = False 

    def add_edge(self, edge):
        self.edges.append(edge)
    
    def visit(self):
        self.visited = True
    
    def discover(self):
        self.discovered = True
    
    def get_parent(self):
        return self.previous
    
    def reset(self):
        self.discovered = False
        self.visited = False
        self.distance = float('inf')
        self.previous = None

    def set_carpool(self):
        self.carpool = True

    # magic methods; vertices are compared by distance, then by id if distance is equal
    # these are used when the vertices are added to the heapq
    def __gt__(self, other):
        if self.distance == other.distance:
            return self.id > other.id
        return self.distance > other.distance

    def __lt__(self, other):
        if self.distance == other.distance:
            return self.id < other.id
        return self.distance < other.distance

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __str__(self):
        return f"{self.id} -> {[str(edge) for edge in self.edges]}"


class Edge:
    """
    This is an edge which points from a source vertex (u), to a destination vertex (v), with a weight (w).
    This is used for a directed graph, so it will be stored in the edge list of vertex u.

    Reference: This implementation is based on the approach outlined in the FIT2004 Week 4 lectures.

    Attributes:
    v (int): The vertex ID that the edge is pointing to.
    w (int): The weight of this edge.
    """
    def __init__(self, v, w):
        self.v = v
        self.w = w

    def __str__(self):
        return f"{self.v} ({self.w})"


class Graph:
    """
    The graph class stores all vertices using an adjacency list. This approach is used as the graph is assumed to be 
    sparse in most cases. 
    
    This graph also has a special property, in that it creates more vertices than the actual number of locations passed
    to it (from optimalRoute). This is done because of the distinction made between vertices for travelling alone and 
    carpooling - their vertex edges will have different weights, so an alternate version of the graph is created and
    we can traverse from the "alone" graph to the "carpool" graph by arriving at any passenger location.

    Reference: This implementation is based on the approach outlined in the FIT2004 Week 4 lectures.

    Attributes:
    max_v (int): The maximum vertex id in the graph.
    size (int): The actual number of vertices in the graph, which is roughly double the maximum vertex number.

    vertices (List[Vertex]): A list of length self.size which contains the vertices in the graph.
    destination (Vertex): The current destination vertex after running dijkstra.
    Source (Vertex): The current source vertex after running dijkstra.
    """
    def __init__(self, max_v):
        # size will account for double the maximum vertex number
        # first set of vertices for travelling alone, second set for carpool
        self.max_vertex = max_v
        self.size = (max_v+1)*2

        # create the adjacency list
        self.vertices = [None] * (self.size)

        # vertex index corresponds to ID (i.e vertex id 0 is at self.vertices[0])
        for i in range(self.size):
            self.vertices[i] = Vertex(i)

            # vertices above the maximum are carpool vertices
            if i > max_v:
                self.vertices[i].set_carpool()
        
        # store which vertices to process as output later on
        self.destination = None
        self.source = None

    def get_vertex(self, i):
        """
        Given a location number, return the vertex in the graph.
        """
        return self.vertices[i]
    
    def get_carpool_vertex(self, i):
        """
        Given a location number, return the carpool vertex (which is offset).
        """
        return self.vertices[i + self.max_vertex + 1]

    def reset(self):
        """
        Reset the vertex attributes (this is mainly used for testing).
        """
        for vertex in self.vertices:
            vertex.reset()

    def add_edge(self, u, v, w):
        """
        Given a tuple (u, v, w), create an edge containing (v, w) and add it to the edge list of u.
        """
        self.get_vertex(u).add_edge(Edge(v, w))

    def dijkstra(self, source, destination):
        """
        A method that runs dijkstra's algorithm to find the shortest path. This uses a minheap to start from the source and
        continuously "discover" the nearest vertices that the source's edges lead to. After discovering all vertices, 
        "visit" the nearest discovered vertex so we can "discover" vertices in their edges, and so on. Continue until the 
        distance from the source is determined for all vertices in the graph.

        This algorithm is a modification of dijkstra, as it only needs to identify the shortest path to a particular 
        destination. There are two vertices that represent this destination, which are the "alone" and "carpool" vertices.
        Whichever vertex is reached first is considered to have the lower time, and based on this it can be determined 
        whether carpooling actually saves time or not. This also means that the the program will end as soon as the 
        distance for one of the distance vertices is found.
        
        Precondition: Source and destination are distinct, valid vertices; destination can be reached from source via edges.
        Postcondition: The minimum distance from the source to the destination (and the vertices between them) will be known.
        
        Input:
            source: the location number of the vertex to start from
            destination: the location number of the vertex to end at. Note this is a singular location number, 
                        but in the graph, this corresponds to two destination vertices.

        Returns:
            None (distance of the graph vertices is updated)

        Time complexity: 
            Best case: O(E), where E is the number of edges in the source vertex's edge list
                        - this happens when the source has its closest edge pointing to the destination
            Worst case: O(E log V), where V is the total number of vertices and E is the total number of edges

        Space complexity: 
            Input and Aux: O(E), where E is the number of edges
        """
        # get the source vertex from this graph, and set its distance to 0
        source_v = self.get_vertex(source)
        source_v.distance = 0

        # get the destination vertices if from travelling alone and from carpooling
        dest_alone = self.get_vertex(destination)
        dest_carpool = self.get_carpool_vertex(destination)
        
        # create a minimum heap for all discovered vertices
        discovered = []
        
        # add to the vertex to the discovered minheap
        # vertices are sorted by distance, then if distance is equal, by their unique id (magic method for vertex)
        source_v.discover()
        heapq.heappush(discovered, source_v)
        
        # O(V) as this will eventually go through every vertex in the graph at least once
        while len(discovered):
            
            # O(1) to get minimum BUT O(logn) to sink
            u = heapq.heappop(discovered)
            if u == dest_alone or u == dest_carpool:
                # whichever one is found first has the shorter distance (discard the other)
                # set the destination and source values to be processed later, then end processing
                self.destination = u
                self.source = source_v
                return
            
            # the vertex is considered visited and its distance is finalized
            u.visit()

            # O(E) to perform edge relaxation on every edge
            # initialize the distance, or update it if a lower one is found
            for edge in u.edges:
                v = self.get_vertex(edge.v)
                
                # if undiscovered, discover it, set its initial distance and previous, and add it to the minheap
                if not v.discovered:
                    v.discover()
                    v.distance = u.distance + edge.w
                    v.previous = u
                    heapq.heappush(discovered, v)

                # if the vertex is not visited, and a shorter distance is found
                # update the distance, the previous, and the key in the heap
                elif not v.visited and v.distance > u.distance + edge.w:
                    v.distance = u.distance + edge.w
                    v.previous = u
                    heapq.heappush(discovered, v)

    def process_roads(self):
        """
        A method that processes and returns a list of roads from a given source to a destination.

        Precondition: dijkstra has been run at least once, so self.source and self.destination are not None
        Postcondition: The output will represent the most efficient route from source to destination.

        Returns:
            A list of location numbers visited, which shows the fastest route from the source to the destination.

        Time complexity: 
            Best and worst case: O(V), where V is the number of vertices in the most efficient route from source to destination.

        Space complexity: 
            Input and Aux: O(V), where V is the number of vertices in the most efficient route from source to destination.
        """
        vertex = self.destination
        source = self.source
        
        # keep appending the previous destination until source is reached
        previous_ids = []
        previous_ids.append(vertex.id)
        
        # O(V) to go through all previous vertices
        while vertex is not source:
            previous_ids.append(vertex.previous.id)
            vertex = vertex.previous
        
        # reverse to get source -> destination output
        reverseList(previous_ids)
        
        output = []
        carpool = False
        
        # O(V) to go through every ID again to process output
        for id in previous_ids:
            if not carpool:
                # the first time a vertex is carpool, it means we are picking up a passenger (location unchanged)
                # from now on, always use carpool lanes as their distance <= non-carpool lanes
                if self.get_vertex(id).carpool:
                    carpool = True
                
                # not carpooling and non-carpool vertex; append location as usual
                else:
                    output.append(id)
            
            # carpool vertex, so get the location by subtracting the offset
            else:
                output.append(id - self.max_vertex - 1)
        
        return output
