# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:35:26 2017

@author: carmonda
"""
import sys
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

PLOT = False
PIXEL_VALUES = 2
X_VALUES = np.zeros(PIXEL_VALUES)
X_VALUES[0] = -1
X_VALUES[1] = 1
ALPHA = 0.5  # probability that the pixel is the same
BETA = 0.3  # probability that the neighbors are alike
THRESHOLD = 0.01  # the maximum change in any message for the algorithm to converge


def phi_i(xi, yi):
    return np.exp(X_VALUES[xi]*yi*ALPHA)


def phi_ij(xi, xj):
    return np.exp(X_VALUES[xi]*X_VALUES[xj]*BETA)


class Vertex(object):
    def __init__(self, id, name='', y=None, neighs=None, in_msgs=None):
        self._id = id
        self._name = name
        self._y = y # original pixel
        if(neighs == None): neighs = set() # set of neighbour nodes
        if(in_msgs==None): in_msgs = {} # dictionary mapping neighbours to their messages
        self._neighs = neighs
        self._in_msgs = in_msgs
    def add_neigh(self,vertex):
        self._neighs.add(vertex)
    def rem_neigh(self,vertex):
        self._neighs.remove(vertex)
    def get_y_value(self):
        return self._y
    def get_id(self):
        return self._id
    def get_neighbors(self):
        return self._neighs
    def get_belief(self):
        return
    def snd_msg(self,neigh):
        """ Combines messages from all other neighbours
            to propagate a message to the neighbouring Vertex 'neigh'.
        """
        return

    def __str__(self):
        ret = "Name: "+self._name
        ret += "\nNeighbours:"
        neigh_list = ""
        for n in self._neighs:
            neigh_list += " "+n._name
        ret+= neigh_list
        return ret


class Graph(object):
    def __init__(self, number_of_vertices, graph_dict=None):
        """ initializes a graph object
            If no dictionary is given, an empty dict will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict
        # the i,j entry is the message from vertex i to vertex j for each of the values of xj
        self.prevMessages = np.zeros((number_of_vertices, number_of_vertices, PIXEL_VALUES))
        # the i,j entry is the message from vertex i to vertex j for each of the values of xj
        self.messages = np.zeros((number_of_vertices, number_of_vertices, PIXEL_VALUES))

    def vertices(self):
        """ returns the vertices of a graph"""
        return list(self._graph_dict.keys())
    def edges(self):
        """ returns the edges of a graph """
        return self.generate_edges()
    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex]=[]
    def add_edge(self,edge):
        """ assumes that edge is of type set, tuple, or list;
            between two vertices can be multiple edges.
        """
        edge = set(edge)
        (v1,v2) = tuple(edge)
        if v1 in self._graph_dict:
            self._graph_dict[v1].append(v2)
        else:
            self._graph_dict[v1] = [v2]
        # if using Vertex class, update data:
        if(type(v1)==Vertex and type(v2)==Vertex):
            v1.add_neigh(v2)
            v2.add_neigh(v1)
    def generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one or two vertices
        """
        e = []
        for v in self._graph_dict:
            for neigh in self._graph_dict[v]:
                if {neigh,v} not in e:
                    e.append({v,neigh})
        return e
    def __str__(self):
        res = "V: "
        for k in self._graph_dict:
            res+=str(k) + " "
        res+= "\nE: "
        for edge in self.generate_edges():
            res+= str(edge) + " "
        return res

    def init_messages(self):
        # initialize messages to uniform distribution
        for vertex_i in self.vertices():
            i = vertex_i.get_id()
            for vertex_j in vertex_i.get_neighbors():
                j = vertex_j.get_id()
                for val in range(PIXEL_VALUES):
                    self.prevMessages[i, j, val] = 1/PIXEL_VALUES

    def step(self):
        """
        This method performs an update of all outgoing messages based on the previous messages
        after this method the messages ndarray will be updated and prevMessages will remain the same
        """
        for vertex_i in self.vertices():
            # update message from i to i's neighbors
            i = vertex_i.get_id()
            neighbors = vertex_i.get_neighbors()
            psi_i = np.zeros(PIXEL_VALUES)
            for val in range(PIXEL_VALUES):
                y_i = vertex_i.get_y_value()
                psi_i[val] = phi_i(val, y_i)
            # now we have the singleton values

            for vertex_j in neighbors:
                j = vertex_j.get_id()
                # update m_ij(x_j)
                psi_ij = np.zeros((PIXEL_VALUES, PIXEL_VALUES))
                for val_i in range(PIXEL_VALUES):
                    for val_j in range(PIXEL_VALUES):
                        psi_ij[val_i, val_j] = phi_ij(val_i, val_j)
                # now we have for the neighbor j of i the smooth term

                # calculate the contribution of all i's neighbors except j
                message_k_i = np.ones(PIXEL_VALUES)
                for vertex_k in neighbors:
                    k = vertex_k.get_id()
                    if k == j:
                        continue
                    for val_i in range(PIXEL_VALUES):
                        message_k_i[val_i] *= self.prevMessages[k, i, val_i]
                # now we have the contribution of all i's neighbors for each of i's values

                message_i_j = np.zeros(PIXEL_VALUES)  # an entry for each x_j
                for val_j in range(PIXEL_VALUES):
                    current_message_i_j = np.zeros(PIXEL_VALUES)  # for the given xj for each xi
                    for val_i in range(PIXEL_VALUES):
                        current_message_i_j[val_i] = psi_i[val_i] * psi_ij[val_i, val_j] * message_k_i[val_i]
                    message_i_j[val_j] = np.max(current_message_i_j)
                # now message_i_j contains the message from i to j for each xj

                # normalize the message
                message_i_j = normalize(message_i_j)
                for val_j in range(PIXEL_VALUES):
                    self.messages[i, j, val_j] = message_i_j[val_j]

    def converged(self):
        max_change = 0
        for vertex_i in self.vertices():
            i = vertex_i.get_id()
            for vertex_j in self.vertices():
                j = vertex_j.get_id()
                for val in range(PIXEL_VALUES):
                    curr_change = np.abs(self.messages[i, j, val] - self.prevMessages[i, j, val])
                    if curr_change > max_change:
                        max_change = curr_change
                    self.prevMessages[i, j, val] = self.messages[i, j, val]
        return max_change < THRESHOLD  # true when finally converged

    def decide(self):
        decision = np.zeros(len(self.vertices()))
        for vertex_i in self.vertices():
            i = vertex_i.get_id()
            y = vertex_i.get_y_value()
            neighbors = vertex_i.get_neighbors()
            xi_vals = np.zeros(PIXEL_VALUES)
            psi_i = np.zeros(PIXEL_VALUES)
            message_k_i = np.ones(PIXEL_VALUES)

            for val_i in range(PIXEL_VALUES):
                psi_i[val_i] = phi_i(val_i, y)
                for vertex_k in neighbors:
                    k = vertex_k.get_id()
                    message_k_i[val_i] = message_k_i[val_i] * self.messages[k, i, val_i]
                xi_vals[val_i] = psi_i[val_i] * message_k_i[val_i]
            xi_star = np.argmax(xi_vals)
            decision[i] = X_VALUES[xi_star]
        return decision


def build_grid_graph(n,m,img_mat):
    """ Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
    n: num of rows
    m: num of columns
    img_mat = np.ndarray of shape (n,m) of pixel intensities
    
    returns the Graph object corresponding to the grid
    """
    V = []
    g = Graph(n*m)
    # add vertices:
    for i in range(n*m):
        row,col = (i//m,i%m)
        v = Vertex(i, name="v"+str(i), y=img_mat[row][col])
        g.add_vertex(v)
        if((i%m)!=0): # has left edge
            g.add_edge((v,V[i-1]))
        if(i>=m): # has up edge
            g.add_edge((v,V[i-m]))
        V += [v]
    return g


def grid2mat(grid,n,m, decision):
    """ convertes grid graph to a np.ndarray
    n: num of rows
    m: num of columns
    
    returns: np.ndarray of shape (n,m)
    """
    mat = np.zeros((n,m))
    l = grid.vertices() # list of vertices
    for v in l:
        i = int(v._name[1:])
        row,col = (i//m,i%m)
        mat[row][col] = decision[i] # you should change this of course
    return mat


def normalize(M):
    norm_M = np.zeros(PIXEL_VALUES)
    sumM = np.sum(M)
    for i in range(PIXEL_VALUES):
        norm_M[i] = M[i]/sumM
    return norm_M


def denoise_image(image):
    n, m = image.shape

    # build grid:
    g = build_grid_graph(n, m, image)

    # process grid:
    g.init_messages()  # initialize default value for messages

    while True:
        # propagate messages
        g.step()
        # check for convergence
        if g.converged():
            break

    # make the decision
    decision = g.decide()

    # convert grid to image:
    infered_img = grid2mat(g, n, m, decision)

    return infered_img


def main():
    # begin:
    if len(sys.argv) < 3:
        print( 'Please specify input and output file names.')
        exit(0)
    # load image:
    in_file_name = sys.argv[1]
    image = misc.imread(in_file_name + '.png')

    # binarize the image.
    image = image.astype(np.float32)
    image[image<128] = -1.
    image[image>127] = 1.
    if PLOT:
        plt.imshow(image)
        plt.show()

    infered_img = denoise_image(image)

    if PLOT:
        plt.imshow(infered_img)
        plt.show()

    # save result to output file
    out_file_name = sys.argv[2]
    misc.toimage(infered_img).save(out_file_name + '.png')


if __name__ == "__main__":
    main()
