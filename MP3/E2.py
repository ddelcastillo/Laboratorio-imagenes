# %% Packages
import numpy as np


# %%
# The problem is solved by constructing a simplified undirected unweighted graph
# which is coupled with an union-finder, through which edges are defined through the
# 4 or 8 connectivity which triggers a merge of components. The code is entirely of my
# authorship, the code may be found in https://github.com/ddelcastillo/Notebook/tree/master/Notebook/src/unionFinder
def MyConnComp_201630945_201622695(binary_image, conn):
    binary_image = np.asarray(binary_image, dtype=np.int_)
    if conn != 4 and conn != 8:
        raise Exception('Connectivity value must be 4 or 8.')
    b = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if conn == 4 else np.ones([3, 3], dtype=np.int_)
    n, m = np.size(binary_image, 0), np.size(binary_image, 1)
    v = np.sum(binary_image)  # Number of vertexes (pixels).
    coord_to_vertex = {}
    # Base UnionFinder
    par = [-1] * v  # All pixels are isolated to begin with (they're their own component).
    num_components = v  # Number of components.

    # The root determines which component does the pixel (box) belongs to.
    # Designed so that a vertex's root updates whenever it's called.
    def root(p_vertex):
        if par[p_vertex] < 0:
            return p_vertex
        else:
            par[p_vertex] = root(par[p_vertex])
            return par[p_vertex]

    # Finding and adding all vertexes.
    c = 0
    for i in range(n):
        for j in range(m):
            if binary_image[i, j]:
                coord_to_vertex[(i, j)] = c
                c += 1
    # After all vertexes are processed, their neighbors are checked based on the connectivity.
    for i, j in coord_to_vertex.keys():
        for index_x, x in enumerate(range(i - 1, i + 2)):
            for index_y, y in enumerate(range(j - 1, j + 2)):
                # Checking if (x,y) is not outside the image, that (i,j) != (x,y), and that they're connected.
                if 0 <= x < n and 0 <= y < m and not (x == i and y == j) and b[index_x, index_y]:
                    if binary_image[x, y]:
                        # What happens here is a merge of components when connected (same root).
                        vertex1 = root(coord_to_vertex[(i, j)])
                        vertex2 = root(coord_to_vertex[(x, y)])
                        if vertex1 != vertex2:
                            if par[vertex2] < par[vertex1]:
                                vertex1 += vertex2
                                vertex2 = vertex1 - vertex2
                                vertex1 -= vertex2
                            par[vertex1] += par[vertex2]
                            par[vertex2] = vertex1
                            num_components -= 1  # One fewer component since two merged.
    # Making each connected components based on the parents.
    labeled_image = np.zeros([n, m], dtype=np.int_)
    parents = np.zeros(v, dtype=np.int_)
    parent_to_index = {}
    c = 0
    for i in range(v):
        parents[i] = root(i)
        if parents[i] not in parent_to_index:
            parent_to_index[parents[i]] = c
            c += 1
    # Each pixel is re-drawn with an unique value based on it's respective connected component.
    pixel_labels = [[] for _ in range(num_components)]
    for (x, y), vertex in coord_to_vertex.items():
        pixel_labels[parent_to_index[parents[vertex]]].append(x * n + y)
        labeled_image[x, y] = parent_to_index[parents[vertex]] + 1
    # print(labeled_image, pixel_labels)
    return labeled_image, pixel_labels


# %%
test = np.asarray([[1, 0, 0], [0, 0, 1], [1, 1, 0]])
image, labels = MyConnComp_201630945_201622695(test, 4)