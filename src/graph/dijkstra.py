# Priority dictionary using binary heaps
# David Eppstein, UC Irvine, 8 Mar 2002

# Implements a data structure that acts almost like a dictionary, with two modifications:
# (1) D.smallest() returns the value x minimizing D[x].
#     For this to work correctly, all values D[x] stored in the
#     dictionary must be comparable.
# (2) iterating "for x in D" finds and removes the items from D
#     in sorted order. Each item is not removed until the next item is
#     requested, so D[x] will still return a useful value until the
#     next iteration of the for-loop.
# Each operation takes logarithmic amortized time.

from __future__ import generators


class priorityDictionary(dict):
    def __init__(self):
        '''Initialize priorityDictionary by creating binary heap of
        pairs (value,key). Note that changing or removing a dict entry
        will not remove the old pair from the heap until it is found by
        smallest() or until the heap is rebuilt.'''
        self.__heap = []
        dict.__init__(self)

    def smallest(self):
        '''Find smallest item after removing deleted items from front of
        heap.'''
        if len(self) == 0:
            raise IndexError("smallest of empty priorityDictionary")
        heap = self.__heap
        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            lastItem = heap.pop()
            insertionPoint = 0
            while 1:
                smallChild = 2 * insertionPoint + 1
                if smallChild + 1 < len(heap) and \
                        heap[smallChild] > heap[smallChild + 1]:
                    smallChild += 1
                if smallChild >= len(heap) or lastItem <= heap[smallChild]:
                    heap[insertionPoint] = lastItem
                    break
                heap[insertionPoint] = heap[smallChild]
                insertionPoint = smallChild
        return heap[0][1]

    def __iter__(self):
        '''Create destructive sorted iterator of priorityDictionary.'''
        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iterfn()

    def __setitem__(self, key, val):
        '''Change value stored in dictionary and add corresponding pair
        to heap. Rebuilds the heap if the number of deleted items gets
        large, to avoid memory leakage.'''
        dict.__setitem__(self, key, val)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v, k) for k, v in self.items()]
            self.__heap.sort()
            # builtin sort probably faster than O(n)-time heapify
        else:
            newPair = (val, key)
            insertionPoint = len(heap)
            heap.append(None)
            while insertionPoint > 0 and \
                    newPair < heap[(insertionPoint - 1) // 2]:
                heap[insertionPoint] = heap[(insertionPoint - 1) // 2]
                insertionPoint = (insertionPoint - 1) // 2
            heap[insertionPoint] = newPair

    def setdefault(self, key, val):
        '''Reimplement setdefault to pass through our customized __setitem__.'''
        if key not in self:
            self[key] = val
        return self[key]


############################################################################

# Dijkstra's algorithm for shortest paths
# David Eppstein, UC Irvine, 4 April 2002

# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228

def Dijkstra(G,start,end=None):
  """
  Find shortest paths from the start vertex to all
  vertices nearer than or equal to the end.

  The input graph G is assumed to have the following
  representation: A vertex can be any object that can
  be used as an index into a dictionary.  G is a
  dictionary, indexed by vertices.  For any vertex v,
  G[v] is itself a dictionary, indexed by the neighbors
  of v.  For any edge v->w, G[v][w] is the length of
  the edge.  This is related to the representation in
  <http://www.python.org/doc/essays/graphs.html>
  where Guido van Rossum suggests representing graphs
  as dictionaries mapping vertices to lists of neighbors,
  however dictionaries of edges have many advantages
  over lists: they can store extra information (here,
  the lengths), they support fast existence tests,
  and they allow easy modification of the graph by edge
  insertion and removal.  Such modifications are not
  needed here but are important in other graph algorithms.
  Since dictionaries obey iterator protocol, a graph
  represented as described here could be handed without
  modification to an algorithm using Guido's representation.

  Of course, G and G[v] need not be Python dict objects;
  they can be any other object that obeys dict protocol,
  for instance a wrapper in which vertices are URLs
  and a call to G[v] loads the web page and finds its links.

  The output is a pair (D,P) where D[v] is the distance
  from start to v and P[v] is the predecessor of v along
  the shortest path from s to v.

  Dijkstra's algorithm is only guaranteed to work correctly
  when all edge lengths are positive. This code does not
  verify this property for all edges (only the edges seen
  before the end vertex is reached), but will correctly
  compute shortest paths even for some graphs with negative
  edges, and will raise an exception if it discovers that
  a negative edge has caused it to make a mistake.
  """

  D = {}  # dictionary of final distances
  P = {}  # dictionary of predecessors
  Q = priorityDictionary()   # est.dist. of non-final vert.
  Q[start] = 0

  for v in Q:
    D[v] = Q[v]
    if v == end: break

    for w in G[v]:
      vwLength = D[v] + G[v][w]
      if w in D:
        if vwLength < D[w]:
          raise ValueError("Dijkstra: found better path to already-final vertex")
      elif w not in Q or vwLength < Q[w]:
        Q[w] = vwLength
        P[w] = v

  return (D,P)

def shortestPath(G,start,end):
  """
  Find a single shortest path from the given start vertex
  to the given end vertex.
  The input has the same conventions as Dijkstra().
  The output is a list of the vertices in order along
  the shortest path.
  """

  D,P = Dijkstra(G,start,end)
  Path = []
  while 1:
    Path.append(end)
    if end == start: break
    end = P[end]
  Path.reverse()
  return Path
