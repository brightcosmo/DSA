from collections import deque

__author__ = "Amirul Azizol"

"""
The file was created for Assignment 2, FIT2004 and submitted on 26th May 2023.
All code is written by Amirul Azizol (32619898).

This file should contain these classes in the following order:
- Q1 Function: maxThroughput
- Q1 Classes: Vertex, Edge, ResidualNetwork
- Q2 Classes: TrieNode, CatsTrie

Time and space complexity of all functions are generally O(1) unless specified.
"""

def maxThroughput(connections, maxIn, maxOut, origin, targets):
    """
    The maxThroughput function, which should return the maximum flow for a given scenario. The scenario involves an
    origin where all data comes from, a list of targets that can recieve data, and intermediary data centres with
    connections between them. These connections have a maximum capacity, but each data centre also has its own capacity 
    for incoming and outgoing data.

    Explanation of apprach:
    My approach builds a residual network and runs Ford-Fulkerson's algorithm with BFS to continuously find a path to
    augment, update the graph, and repeat until we identify the maximum flow value. Each connection is represented by
    an Edge and each data centre is represented by a Vertex.

    To deal with the concept of maxIn and maxOut for vertices, I represent one data centre with three vertices (v1, v2, v3). 
    This is to allow for "internal edges" of capacity maxIn between v1 and v2, and capacity maxOut between v2 and v3. 
    External edges exclusively interact with v1 and v3 depending on whether the data centre is the source or destination.
    
    To deal with targets and origin, I created a proper "source" and "sink" and stored these as distinct vertices. I then add
    additional edges that link the origin data centre to the source (since the origin sometimes has incoming edges). Additionally, 
    all targets have an edge directly linking them to the sink in the end. This final sink allows me to identify the maximum flow
    as I only need to run bfs from source to sink rather than source to any targets T.

    Precondition: Origin and targets represent valid locations, and there is a way to travel from origin to targets.
    Postcondition: There are no more augmenting paths that can be found, thus the maximum flow was obtained.
    
    Input:
        connections (List[Tuple[int, int, int]]): A list of tuples where each tuple represents a connection
            between data centres.
        maxIn (List[int]): A list of integers where maxIn[i] corresponds to the maximum incoming data for data centre i.
        maxOut (List[int]): A list of integers where maxOut[i] corresponds to the maximum outgoing data for data centre i.
        origin: The data centre where all the data originates from.
        targets: A list of data centres that can store data from the origin.

    Returns:
        The maximum amount of data that can be backed up per second from the origin to any of the targets. 

    Time complexity: 
        Best and Worst case: O(D * C^2), where:
                                - D is the number of data centres
                                - C is the number of connections
    
    Space complexity: 
        Input and Aux: O(C + D), where:
                                - C is the number of connections
                                - D is the number of data centres
    """
    # create the residual network using the initial number of data centres
    # O(D) to create the graph
    network = ResidualNetwork(len(maxIn))

    # add edge from max_in to vertex, and from vertex to max_out
    for id in range(len(maxIn)):        
        input_v = network.get_max_in(id)
        vertex = network.get_vertex(id)
        output_v = network.get_max_out(id)

        network.add_edge(input_v, vertex, maxIn[id])
        network.add_edge(vertex, output_v, maxOut[id])
    
    # add edge from target to the sink
    for id in targets:
        target_v = network.get_vertex(id)
        network.add_edge(target_v, network.sink, maxIn[id])

    # add an edge for each connection
    for a, b, t in connections:
        u = network.get_max_out(a)
        v = network.get_max_in(b)
        network.add_edge(u, v, t)

    # add an edge connecting the source to the origin
    origin_v = network.get_vertex(origin)
    network.add_edge(network.source, origin_v, maxOut[origin])

    # O(FE) to run ford fulkerson
    return network.ford_fulkerson() 

class Vertex:
    """
    This is a vertex, used to represent a data centre in the residual network.
    
    Attributes:
    id (int): A unique identifier.
    edges (List[Edge]): A list of the edges that start with this vertex.
    discovered (bool): True if BFS reaches this vertex.
    previous_edge (Edge): The edge that was used to reach this vertex.
    """
    def __init__(self, id):
        self.id = id
        self.edges = []
        self.discovered = False
        self.previous_edge = None

    def add_edge(self, edge):
        """Add an edge to the list of edges"""
        self.edges.append(edge)
    
    def visit(self):
        """Mark the vertex as visited"""
        self.visited = True
    
    def discover(self):
        """Mark the vertex as discovered"""
        self.discovered = True
    
    def reset(self):
        """Reset the vertex's attributes, called after every BFS run."""
        self.discovered = False 
        self.previous_edge = None

class Edge:
    """
    This is a directed edge between two vertices in a residual network.

    Attributes:
    u (Vertex): The vertex which flow comes from.
    v (Vertex): The vertex which receives the flow.
    capacity (int): The maximum flow that can be transferred. 
    flow (int): The current flow for this edge.
    reverse_edge (Edge): The corresponding reverse edge.
    """
    def __init__(self, u, v, capacity, inverse):
        """Flow is initialized to the same value as capacity for inverse edges"""
        self.u = u
        self.v = v
        self.capacity = capacity
        self.flow = capacity if inverse else 0
        self.reverse_edge = None
    
    def set_reverse(self, reverse_edge):
        """Set the reverse edge"""
        self.reverse_edge = reverse_edge

    def add_flow(self, flow):
        """Add flow to the edge, and subtract from the corresponding reverse edge's flow."""
        self.flow += flow
        self.reverse_edge.flow -= flow

    def get_residual(self):
        """Get the remaining capacity that can be augmented for this edge."""
        return self.capacity - self.flow

class ResidualNetwork:
    """
    This is a residual network used to solve the maximm flow problem with a given flow network.
    
    Note that the number of intended vertices ultimately corresponds to 3 vertices. The reason for this 
    is explained in more detail in the maxThroughput function below. 
    
    Two additional vertices are also created; these represent the source and the sink.

    Attributes:
    size (int): The vertex which receives the flow.
    vertices (int): The maximum flow that can be transferred.
    sink (Vertex): The vertex that all flow leads to.
    source (Vertex): The vertex that all flow comes from.
    """
    def __init__(self, vertex_count):
        self.size = (3*vertex_count) + 2
        self.vertices = [Vertex(id) for id in range(self.size)]
        self.sink = self.vertices[-2]
        self.source = self.vertices[-1]

    def get_max_in(self, i):
        """Given a vertex ID, get the vertex that receives flow on its behalf."""
        return self.vertices[3*i]

    def get_vertex(self, i):
        """Given a vertex ID, return the vertex itself."""
        return self.vertices[3*i+1]

    def get_max_out(self, i):
        """Given a vertex ID, get the vertex that produces flow for other vertices."""
        return self.vertices[3*i+2]
    
    def add_edge(self, u, v, capacity):
        """
        Add an edge between two vertices in the residual network, along with its reverse edge.
        Each edge starts with 0 flow while its reverse starts with the maximum flow.
        These edges are added to the respective edge lists and saved as each other's reverse edge.
        This is done so that adding flow to one edge will subtract the same value from its reverse.

        Input:
            u (Vertex): The vertex which flow comes from.
            v (Vertex): The vertex which receives the flow.
            capacity (int): The maximum flow of the edge.
        """
        edge = Edge(u, v, capacity, False)
        reverse_edge = Edge(v, u, capacity, True)

        edge.set_reverse(reverse_edge)
        reverse_edge.set_reverse(edge)

        u.add_edge(edge)
        v.add_edge(reverse_edge)
    
    def ford_fulkerson(self):
        """
        Runs the ford fulkerson algorithm to solve the maximum flow problem.
        This method repeatedly finds a path and augments it, then uses
        then adds the flow calculated. This continues until another path cannot be found.

        Time complexity: 
            Best and worst case: O(V * E^2), where:
                                - V is the number of vertices
                                - E is the number of edges
        """
        max_flow = 0
        while self.find_and_augment(): max_flow += self.residual
        return max_flow

    def find_and_augment(self):
        """
        A breadth-first search function, which continuously finds a path from the source to the sink.
        Once a path is found, it augments the path updating the flow of edges accordingly.
        The exact value of flow used is the minimum residual across every edge; this value is also saved in 
        self.residual so it can be accessed in other methods.

        Returns:
            True if a path was found, False otherwise.

        Time complexity: 
            Best and Worst case: O(V + E), where:
                                - V is the number of vertices
                                - E is the number of edges
        """
        self.residual = float('inf')
        
        # reset all vertices (discovered / previous edge)
        for v in self.vertices: v.reset()
        
        # create a queue, start with source and run BFS
        discovered = deque()
        discovered.append(self.source)
        
        # O(V) where V is the number of vertices
        while len(discovered) > 0:
            u = discovered.popleft()

            # if we reached the sink, augment the path using the minimum residual found
            if u == self.sink:
                current_edge = u.previous_edge
                while current_edge.u is not self.source: # O(E) where E is the length of the path found
                    current_edge.add_flow(self.residual)
                    current_edge = current_edge.u.previous_edge
                return True
            
            # O(E) where E is the number of edges for each vertex
            for edge in u.edges:
                v = edge.v
                if not v.discovered and edge.get_residual(): # discover vertices if we can add more flow (residual)
                    discovered.append(v)
                    v.discover()
                    v.previous_edge = edge
                    self.residual = edge.get_residual() if edge.get_residual() < self.residual else self.residual
        
        return False # if we terminate without visiting sink

class TrieNode:
    def __init__(self, char, parent):
        """
        This is a node used in the CatsTrie data structure, to efficiently store and search for strings.
        
        Attributes:
            char (str): The character stored in the current node.
            link (List[None | TrieNode]): A list of child TrieNodes which represent the next character in the sequence of strings
                                        - The index within this list corresponds to the position of the character in the alphabet.
                                        - If None, this indicates that the child for that character doesn't exist yet.
            terminal (TrieNode): An additional node which acts as a terminal, i.e marking that a word can end after this character.
            parent (TrieNode): The parent node, which has this node as one of its links.
            best_node (TrieNode): The child TrieNode which will lead to the end of a word with the highest frequency (from this node).
            frequency (int): The frequency of the word that best_node leads to.
        """
        self.char = char
        self.link = [None] * 26
        self.parent = parent
        self.terminal = None
        self.best_node = None
        self.frequency = 0
    
    def index_of(self, char):
        """Get the index in the links array for any given character"""
        return ord(char) - 97
    
    def add_link(self, char):
        """Create a new link for a given character"""
        self.link[self.index_of(char)] = TrieNode(char, self)
        return self.link[self.index_of(char)]

    def get_terminal(self):
        """Get the existing ending node, or create one if it doesn't exist, and increment."""
        if self.terminal is None: self.terminal = TrieNode("", self)
        self.terminal.frequency += 1
        return self.terminal
    
    def update_best(self, new_best):
        """Update the best node with its frequency"""
        self.best_node = new_best
        self.frequency = new_best.frequency

    ################################### Magic methods ##################################
    # Compare freqencies of nodes
    def __gt__(self, other): return self.frequency > other.frequency
    def __lt__(self, other): return self.frequency < other.frequency
    def __eq__(self, other): return self.frequency == other.frequency
    
    # Get the child node corresponding to the given character
    def __getitem__(self, char): return self.link[self.index_of(char)]

    # Get the character of a node
    def __str__(self): return self.char

class CatsTrie:
    """
    This is a variant of the Trie data structure which stores "sentences" (words) consisting of "words" (characters).
    Each character is represented by its own node, and words can be formed by getting subsequent child nodes from the 
    root. 

    Explanation of approach:
    For question 2, my solution involves processing each word during their insert, since the complexity of initializing
    the CatsTrie is O(N * M) regardless. Processing entails getting the frequency of the word we just inserted, and checking 
    if this is the most frequent word from each parent node. We recursively make this check, and update as necessary. 
    
    By processing and updating during the insert, we are able to more efficiently run the autoComplete function. Each
    node in the CatsTrie will know in O(1) time the "best node" which will lead to the word with the highest frequency 
    of the Trie, allowing us to achieve O(Y) complexity (length of the most frequent word) in the worst case, and O(X)
    complexity (length of the prompt) in the best case.

    Attributes:
        self.root (TrieNode): The root node. Every other node can be traversed to from here.
    """
    def __init__(self, sentences):
        """
        Initializes the CatsTrie. This inserts every sentence and, after every insertion, updates the best node of
        each parent node. This "best node" represents the node leading to the word with the highest frequency.

        Input: 
            sentences (List[String]): A list of sentences that need to be stored, where each sentence is a string.
        
        Time complexity: 
            Best and worst case: O(N * M), where:
                                - N is the number of sentences
                                - M is the length of the longest sentence
    
        Space complexity: 
            Input and Aux: O(N * M), where:
                                - N is the number of sentences
                                - M is the length of the longest sentence
        """
        self.root = TrieNode("", None)
        for sentence in sentences: 
            self.insert(sentence)
    
    def insert(self, sentence):
        """
        A function that inserts a single sentence into the CatsTrie, by looping through every character
        and adding child nodes if they don't exist yet. After insertion, we create/get a terminal node at the
        location, increment the frequency, and recursively update the frequencies of all the parent nodes.

        Input: 
            sentence: A string of characters to insert into the CatsTrie.
        
        Time complexity:
            Best and worst case: O(M), where M is the length of the sentence given.
        
        Space complexity: 
            Input and Aux: O(M), where M is the length of the sentence given.
        """
        current = self.root
        for char in sentence: 
            current = current[char] if current[char] is not None else current.add_link(char)
        terminal = current.get_terminal()
        self.update_max(terminal)

    def update_max(self, current):
        """
        A recursive function that continuously compares the current node's frequency with its parent, 
        updating the parent's "best node" if current frequency is higher. If the frequencies are equal, 
        a tiebreaker is used where the lexicographic character is compared.

        Recursion ends when the current frequency is lower than the parent (no need to check the rest) or 
        the root is reached.

        Input:
            current (TrieNode): The current node. This is initially the terminal and traverses upwards by getting the parent.
        
        Time complexity:
            Best case: O(1)
                - When the terminal's frequency is lower than its parent node, so nothing needs to be updated.
            
            Worst case: O(M) where M is the length of the longest word
                - When we insert the longest word in the CatsTrie, and we have to update every node up to the root.
        
        Space complexity: 
            Input and Aux: O(1), this is an in-place function and simply updates existing data stored.
        """
        parent = current.parent

        # stop updating the node
        if parent is None or current < parent:
            return

        # update and continue
        elif current > parent or str(current) < str(parent.best_node): 
            parent.update_best(current)
            self.update_max(parent)

    def autoComplete(self, prompt):
        """
        Given a string prompt, find the most common word in this sentence that starts with the prompt.

        Input:
            prompt (String): The prompt, which will autocomplete to the longest word
        
        Time complexity:
            Best case: O(X), where X is the length of the prompt
                - When the prompt doesn't exist in the CatsTrie and None is returned.
            
            Worst case: O(Y) where m is the length of the longest word starting with the prompt
                - When the word is found.
        Space complexity: 
            Input and Aux: O(X + Y), where
                - X is the length of the prompt
                - Y is the length of the longest sentence beginning with the prompt
        """
        current = self.root
        sentence = []

        # O(X) to traverse to the node according to the prompt given 
        for char in prompt:
            sentence.append(char)
            current = current[char]
            if current is None: return None # early exit when prompt doesn't exist

        # O(Y) to traverse to the most frequent word that continues the end of the prompt
        current = current.best_node
        while current is not None:
            sentence.append(str(current))
            current = current.best_node
        
        # O(Y) to join the list together as a string
        return "".join(sentence)