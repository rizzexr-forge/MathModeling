import numpy as np
import math
import random

class GraphMatrixRepresentation:
    """Класс для работы с матричными представлениями графов"""
    def __init__(self, vertices_count, directed=False):
        self.V = vertices_count
        self.directed = directed
        self.vertices_names = [f"V{i+1}" for i in range(vertices_count)]
        self.adj_matrix = np.zeros((vertices_count, vertices_count), dtype=float)
        self.edges = []  # (u, v, weight)
        self.coordinates = self._generate_coordinates()

    def _generate_coordinates(self):
        np.random.seed(42)
        return {i: (np.random.uniform(0, 100), np.random.uniform(0, 100)) for i in range(self.V)}

    def add_edge(self, u, v, weight=1):
        self.adj_matrix[u][v] = weight
        self.edges.append((u, v, weight))
        if not self.directed:
            self.adj_matrix[v][u] = weight
            self.edges.append((v, u, weight))

    def remove_edge(self, u, v):
        self.adj_matrix[u][v] = 0
        self.edges = [(a, b, w) for a, b, w in self.edges if not (a == u and b == v)]
        if not self.directed:
            self.adj_matrix[v][u] = 0
            self.edges = [(a, b, w) for a, b, w in self.edges if not (a == v and b == u)]

    def get_adjacency_matrix(self): return self.adj_matrix.copy()

    def get_incidence_matrix(self):
        if not self.edges: return np.zeros((self.V, 0), dtype=int)
        if not self.directed:
            inc = np.zeros((self.V, len(self.edges)), dtype=int)
            for idx, (u, v, _) in enumerate(self.edges):
                inc[u][idx] = 1; inc[v][idx] = 1
            return inc
        else:
            inc = np.zeros((self.V, len(self.edges)), dtype=int)
            for idx, (u, v, _) in enumerate(self.edges):
                inc[u][idx] = 1; inc[v][idx] = -1
            return inc

    def get_distance_matrix(self):
        dist = self.adj_matrix.copy()
        for i in range(self.V):
            for j in range(self.V):
                if i != j and dist[i][j] == 0: dist[i][j] = float('inf')
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def get_reachability_matrix(self):
        reach = (self.adj_matrix > 0).astype(int)
        for i in range(self.V): reach[i][i] = 1
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    if reach[i][k] and reach[k][j]: reach[i][j] = 1
        return reach

    def get_laplacian_matrix(self):
        if self.directed:
            deg = np.sum(self.adj_matrix > 0, axis=1) + np.sum(self.adj_matrix > 0, axis=0)
        else:
            deg = np.sum(self.adj_matrix > 0, axis=1)
        lap = -self.adj_matrix.copy()
        np.fill_diagonal(lap, deg)
        return lap

    def get_degree_matrix(self):
        if self.directed:
            return np.diag(np.sum(self.adj_matrix > 0, axis=1)), np.diag(np.sum(self.adj_matrix > 0, axis=0))
        return np.diag(np.sum(self.adj_matrix > 0, axis=1))

    def adjacency_to_incidence(self, adj_matrix=None):
        if adj_matrix is None: adj_matrix = self.adj_matrix
        edges = []
        for i in range(self.V):
            for j in range(self.V):
                if adj_matrix[i][j] != 0 and (self.directed or i <= j):
                    edges.append((i, j))
        inc = np.zeros((self.V, len(edges)), dtype=int)
        for idx, (u, v) in enumerate(edges):
            if self.directed: inc[u][idx] = 1; inc[v][idx] = -1
            else: inc[u][idx] = 1; inc[v][idx] = 1
        return inc

    def incidence_to_adjacency(self, inc_matrix):
        V, E = inc_matrix.shape
        adj = np.zeros((V, V))
        for j in range(E):
            verts = [i for i in range(V) if inc_matrix[i][j] != 0]
            if len(verts) >= 2:
                u, v = verts[0], verts[1]
                adj[u][v] = 1
                if not self.directed: adj[v][u] = 1
        return adj

    def print_matrix(self, matrix, title="Матрица", format_float=False):
        print(f"\n{title}:")
        print("  " + "".join([f"{name:>6}" for name in self.vertices_names]))
        for i in range(self.V):
            row = f"{self.vertices_names[i]:>3}:"
            cols = matrix.shape[1] if len(matrix.shape) > 1 else self.V
            for j in range(cols):
                val = matrix[i][j]
                if format_float and isinstance(val, (float, np.floating)):
                    row += "     ∞" if math.isinf(val) else f"{val:6.2f}"
                else:
                    row += f"{int(val):6d}" if val != 0 else "     0"
            print(row)

    def generate_random_graph(self, density=0.4, min_weight=1, max_weight=10):
        self.adj_matrix = np.zeros((self.V, self.V))
        self.edges = []
        for i in range(self.V):
            for j in range(self.V):
                if i != j and random.random() < density:
                    self.add_edge(i, j, random.randint(min_weight, max_weight))
        self._ensure_connectivity(min_weight, max_weight)
        return self

    def _ensure_connectivity(self, min_w, max_w):
        visited = [False] * self.V
        def dfs(v):
            visited[v] = True
            for u in range(self.V):
                if self.adj_matrix[v][u] > 0 and not visited[u]: dfs(u)
        dfs(0)
        for i in range(self.V):
            if not visited[i]: self.add_edge(0, i, random.randint(min_w, max_w))

    def print_statistics(self):
        print("\n" + "="*70)
        print("СТАТИСТИКА ГРАФА")
        print("="*70)
        num_edges = len(self.edges) // (2 if not self.directed else 1)
        print(f"Количество вершин: {self.V}\nКоличество рёбер: {num_edges}")
        print(f"Тип: {'Ориентированный' if self.directed else 'Неориентированный'}")
        if not self.directed:
            deg = np.sum(self.adj_matrix > 0, axis=1)
            print("\nСтепени вершин:")
            for i in range(self.V): print(f"  {self.vertices_names[i]}: {int(deg[i])}")
        max_e = self.V * (self.V - 1) // (1 if not self.directed else 2)
        dens = num_edges / max_e if max_e > 0 else 0
        print(f"\nПлотность: {dens:.3f} ({dens*100:.1f}%)")
        if not self.directed:
            print(f"Компонент связности: {len(self._find_connected_components())}")

    def _find_connected_components(self):
        visited, comps = [False] * self.V, []
        def dfs(v, c):
            visited[v] = True; c.append(v)
            for u in range(self.V):
                if self.adj_matrix[v][u] > 0 and not visited[u]: dfs(u, c)
        for i in range(self.V):
            if not visited[i]:
                c = []; dfs(i, c); comps.append(c)
        return comps

class MatrixOperations:
    @staticmethod
    def matrix_multiply(A, B): return np.dot(A, B)
    @staticmethod
    def matrix_power(A, k):
        res = np.eye(A.shape[0])
        for _ in range(k): res = MatrixOperations.matrix_multiply(res, A)
        return res
    @staticmethod
    def count_paths(adj_matrix, length): return MatrixOperations.matrix_power(adj_matrix, length)
    @staticmethod
    def is_connected(adj_matrix):
        V = adj_matrix.shape[0]
        reach = (adj_matrix > 0).astype(int)
        for i in range(V): reach[i][i] = 1
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if reach[i][k] and reach[k][j]: reach[i][j] = 1
        return np.all(reach > 0)
    @staticmethod
    def find_centers(graph):
        dist = graph.get_distance_matrix()
        ecc = [max(dist[i][j] for j in range(graph.V) if not math.isinf(dist[i][j])) for i in range(graph.V)]
        min_e = min(ecc)
        return [graph.vertices_names[i] for i, e in enumerate(ecc) if e == min_e], min_e
    @staticmethod
    def find_radius(graph):
        dist = graph.get_distance_matrix()
        return min(max(dist[i][j] for j in range(graph.V) if not math.isinf(dist[i][j])) for i in range(graph.V))
    @staticmethod
    def find_diameter(graph):
        dist = graph.get_distance_matrix()
        return max(dist[i][j] for i in range(graph.V) for j in range(graph.V) if not math.isinf(dist[i][j]))