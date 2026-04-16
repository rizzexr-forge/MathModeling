import numpy as np
import heapq
import random
from collections import defaultdict

class DisjointSet:
    """Система непересекающихся множеств (DSU) для алгоритма Краскала"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """Поиск корня множества с сжатием путей"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Объединение двух множеств"""
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return False

        # Объединяем по рангу
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
        return True

class Graph:
    """Класс для представления взвешенного неориентированного графа"""
    def __init__(self, vertices):
        self.V = vertices
        self.vertices_names = [f"V{i+1}" for i in range(vertices)]
        self.edges = []  # Список рёбер (вес, u, v)
        self.adj_matrix = np.zeros((vertices, vertices))

    def add_edge(self, u, v, weight):
        """Добавление ребра в граф"""
        if weight < 0:
            print(f"Предупреждение: отрицательный вес {weight} для ребра {u}-{v}")
        self.edges.append((weight, u, v))
        self.adj_matrix[u][v] = weight
        self.adj_matrix[v][u] = weight

    def add_edge_named(self, u_name, v_name, weight):
        """Добавление ребра по именам вершин (V1, V2...)"""
        try:
            u = int(u_name.replace('V', '')) - 1
            v = int(v_name.replace('V', '')) - 1
            self.add_edge(u, v, weight)
        except ValueError:
            print(f"Ошибка в именах вершин: {u_name}, {v_name}")

    def get_vertex_name(self, v):
        """Получение имени вершины по индексу"""
        return self.vertices_names[v]

    def kruskal_mst(self):
        """Алгоритм Краскала для нахождения MST"""
        sorted_edges = sorted(self.edges, key=lambda x: x[0])
        dsu = DisjointSet(self.V)
        mst_edges = []
        total_weight = 0

        print("\n" + "="*60)
        print("АЛГОРИТМ КРАСКАЛА")
        print("="*60)
        print("Шаги выполнения:")

        for weight, u, v in sorted_edges:
            if dsu.union(u, v):
                mst_edges.append((weight, u, v))
                total_weight += weight
                print(f"  Добавлено ребро {self.get_vertex_name(u)}-{self.get_vertex_name(v)} с весом {weight}")
        
        print(f"\nИтоговый вес MST: {total_weight}")
        print(f"Количество рёбер в MST: {len(mst_edges)}")
        return mst_edges, total_weight

    def prim_mst(self, start_vertex=0):
        """Алгоритм Прима для нахождения MST"""
        visited = [False] * self.V
        pq = []
        heapq.heappush(pq, (0, start_vertex, -1))
        mst_edges = []
        total_weight = 0

        print("\n" + "="*60)
        print("АЛГОРИТМ ПРИМА")
        print("="*60)
        print(f"Стартовая вершина: {self.get_vertex_name(start_vertex)}")
        print("Шаги выполнения:")

        while pq:
            weight, v, parent = heapq.heappop(pq)
            if visited[v]:
                continue
            
            visited[v] = True
            if parent != -1:
                mst_edges.append((weight, parent, v))
                total_weight += weight
                print(f"  Добавлено ребро {self.get_vertex_name(parent)}-{self.get_vertex_name(v)} с весом {weight}")

            for u in range(self.V):
                if not visited[u] and self.adj_matrix[v][u] > 0:
                    heapq.heappush(pq, (self.adj_matrix[v][u], u, v))

        print(f"\nИтоговый вес MST: {total_weight}")
        print(f"Количество рёбер в MST: {len(mst_edges)}")
        return mst_edges, total_weight

    def generate_random_graph(self, density=0.5, min_weight=1, max_weight=20):
        """Генерация случайного связного графа"""
        self.edges = []
        self.adj_matrix = np.zeros((self.V, self.V))
        
        for i in range(self.V):
            for j in range(i + 1, self.V):
                if random.random() < density:
                    weight = random.randint(min_weight, max_weight)
                    self.add_edge(i, j, weight)

        if not self.is_connected():
            print("Граф несвязный, добавляем дополнительные рёбра...")
            self.ensure_connectivity(min_weight, max_weight)
        return self

    def is_connected(self):
        """Проверка графа на связность с помощью BFS"""
        visited = [False] * self.V
        queue = [0]
        visited[0] = True
        
        while queue:
            v = queue.pop(0)
            for u in range(self.V):
                if self.adj_matrix[v][u] > 0 and not visited[u]:
                    visited[u] = True
                    queue.append(u)
        return all(visited)

    def ensure_connectivity(self, min_weight, max_weight):
        """Обеспечение связности графа путём соединения компонент"""
        components = []
        visited = [False] * self.V
        
        for v in range(self.V):
            if not visited[v]:
                component = []
                queue = [v]
                visited[v] = True
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    for u in range(self.V):
                        if self.adj_matrix[current][u] > 0 and not visited[u]:
                            visited[u] = True
                            queue.append(u)
                components.append(component)

        for i in range(len(components) - 1):
            v1 = components[i][0]
            v2 = components[i+1][0]
            weight = random.randint(min_weight, max_weight)
            self.add_edge(v1, v2, weight)

    def print_graph(self):
        """Вывод информации о графе"""
        print("\n" + "="*60)
        print("ИНФОРМАЦИЯ О ГРАФЕ")
        print("="*60)
        print(f"Количество вершин: {self.V}")
        print(f"Количество рёбер: {len(self.edges)}")
        print("\nСписок рёбер:")
        for weight, u, v in sorted(self.edges):
            print(f" {self.get_vertex_name(u)}--{self.get_vertex_name(v)}: {weight}")
        print("\nМатрица смежности:")
        header = "   " + "".join([f"{self.get_vertex_name(i):>3}" for i in range(self.V)])
        print(header)
        for i in range(self.V):
            row = f"{self.get_vertex_name(i):>3}:"
            for j in range(self.V):
                val = int(self.adj_matrix[i][j]) if self.adj_matrix[i][j] > 0 else 0
                row += f"{val:>3}" if val > 0 else "  0"
            print(row)

class MSTComparator:
    """Класс для сравнения алгоритмов"""
    @staticmethod
    def compare_algorithms(graph):
        print("\n" + "="*60)
        print("СРАВНЕНИЕ АЛГОРИТМОВ")
        print("="*60)

        kruskal_edges, kruskal_weight = graph.kruskal_mst()
        prim_edges, prim_weight = graph.prim_mst()

        print("\n" + "-"*40)
        print("РЕЗУЛЬТАТЫ:")
        print("-"*40)
        print(f"Алгоритм Краскала: вес = {kruskal_weight}")
        print(f"Алгоритм Прима:    вес = {prim_weight}")

        if abs(kruskal_weight - prim_weight) < 1e-6:
            print("\n✓ Алгоритмы дали одинаковый результат (вес совпадает)")
        else:
            diff = abs(kruskal_weight - prim_weight)
            print(f"\n⚠ Разница в весе: {diff}")
            if kruskal_weight < prim_weight:
                print("  Алгоритм Краскала дал лучшее решение")
            else:
                print("  Алгоритм Прима дал лучшее решение")

        print(f"\nКоличество рёбер в MST: {len(kruskal_edges)} (должно быть {graph.V - 1})")

        return {
            'kruskal': {'edges': kruskal_edges, 'weight': kruskal_weight},
            'prim': {'edges': prim_edges, 'weight': prim_weight}
        }