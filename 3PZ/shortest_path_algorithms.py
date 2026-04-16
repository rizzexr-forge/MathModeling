import numpy as np
import heapq
from collections import defaultdict
import math
import random
import time

class Graph:
    """Класс для представления взвешенного графа (ориентированного или нет)"""
    def __init__(self, vertices, directed=False):
        self.V = vertices
        self.directed = directed
        self.vertices_names = [f"V{i+1}" for i in range(vertices)]
        self.edges = []
        self.adj_list = defaultdict(list)
        self.adj_matrix = np.full((vertices, vertices), np.inf)
        for i in range(vertices):
            self.adj_matrix[i][i] = 0

    def add_edge(self, u, v, weight):
        """Добавление ребра в граф"""
        self.edges.append((u, v, weight))
        self.adj_list[u].append((v, weight))
        self.adj_matrix[u][v] = weight
        if not self.directed:
            self.adj_list[v].append((u, weight))
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
        return self.vertices_names[v]

    def dijkstra(self, start):
        """Алгоритм Дейкстры для поиска кратчайших путей от одной вершины"""
        if start < 0 or start >= self.V:
            raise ValueError("Неверная начальная вершина")
            
        dist = [math.inf] * self.V
        prev = [-1] * self.V
        visited = [False] * self.V
        dist[start] = 0
        pq = [(0, start)]
        
        print("\n" + "="*60)
        print(f"АЛГОРИТМ ДЕЙКСТРЫ (от вершины {self.get_vertex_name(start)})")
        print("="*60)
        print("Шаги выполнения:")
        step = 1
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            if visited[u]: continue
            visited[u] = True
            print(f"  Шаг {step}: Обрабатываем вершину {self.get_vertex_name(u)} (расстояние={current_dist})")
            
            for v, weight in self.adj_list[u]:
                if not visited[u] and dist[u] + weight < dist[v]:
                    old_dist = dist[v]
                    dist[v] = dist[u] + weight
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
                    print(f"    → Обновляем расстояние до {self.get_vertex_name(v)}: {old_dist if old_dist!=math.inf else '∞'} → {dist[v]}")
            step += 1
            
        print("\nИтоговые расстояния:")
        for i in range(self.V):
            if i != start:
                print(f"  {self.get_vertex_name(start)} → {self.get_vertex_name(i)}: {dist[i]}")
        return dist, prev

    def bellman_ford(self, start):
        """Алгоритм Беллмана-Форда (поддержка отрицательных весов)"""
        if start < 0 or start >= self.V:
            raise ValueError("Неверная начальная вершина")
            
        dist = [math.inf] * self.V
        prev = [-1] * self.V
        dist[start] = 0
        
        print("\n" + "="*60)
        print(f"АЛГОРИТМ БЕЛЛМАНА-ФОРДА (от вершины {self.get_vertex_name(start)})")
        print("="*60)
        
        for i in range(self.V - 1):
            relaxed = False
            print(f"\n  Итерация {i+1}:")
            for u, v, weight in self.edges:
                if dist[u] != math.inf and dist[u] + weight < dist[v]:
                    old_dist = dist[v]
                    dist[v] = dist[u] + weight
                    prev[v] = u
                    relaxed = True
                    print(f"    Ребро {self.get_vertex_name(u)}→{self.get_vertex_name(v)}: {old_dist if old_dist!=math.inf else '∞'} → {dist[v]}")
            if not relaxed:
                print("  Нет изменений, досрочное завершение")
                break
                
        # Проверка на отрицательные циклы
        for u, v, weight in self.edges:
            if dist[u] != math.inf and dist[u] + weight < dist[v]:
                print("\n⚠ ВНИМАНИЕ: Обнаружен отрицательный цикл!")
                return None, None
                
        print("\nИтоговые расстояния:")
        for i in range(self.V):
            if i != start:
                print(f"  {self.get_vertex_name(start)} → {self.get_vertex_name(i)}: {dist[i]}")
        return dist, prev

    def floyd_warshall(self):
        """Алгоритм Флойда-Уоршелла (все пары вершин)"""
        print("\n" + "="*60)
        print("АЛГОРИТМ ФЛОЙДА-УОРШЕЛЛА (все пары вершин)")
        print("="*60)
        
        dist = self.adj_matrix.copy()
        next_vertex = np.full((self.V, self.V), -1, dtype=int)
        for i in range(self.V):
            for j in range(self.V):
                if i != j and not np.isinf(self.adj_matrix[i][j]):
                    next_vertex[i][j] = j
                    
        print("\nНачальная матрица расстояний:")
        self.print_distance_matrix(dist)
        
        for k in range(self.V):
            print(f"\n  Итерация {k+1} (промежуточная вершина {self.get_vertex_name(k)}):")
            for i in range(self.V):
                for j in range(self.V):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        old_dist = dist[i][j]
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_vertex[i][j] = next_vertex[i][k]
                        print(f"    Путь {self.get_vertex_name(i)}→{self.get_vertex_name(j)} улучшен: {old_dist if old_dist!=math.inf else '∞'} → {dist[i][j]}")
                        
        # Проверка на отрицательные циклы
        for i in range(self.V):
            if dist[i][i] < 0:
                print("\n⚠ ВНИМАНИЕ: Обнаружен отрицательный цикл!")
                return None, None
                
        print("\nФинальная матрица расстояний:")
        self.print_distance_matrix(dist)
        return dist, next_vertex

    def print_distance_matrix(self, dist):
        print("\n     " + "".join([f"{self.get_vertex_name(i):>6}" for i in range(self.V)]))
        for i in range(self.V):
            row = f"{self.get_vertex_name(i):>3}:"
            for j in range(self.V):
                if np.isinf(dist[i][j]):
                    row += "     ∞"
                else:
                    row += f"{dist[i][j]:6.1f}"
            print(row)

    def get_path(self, prev, start, end):
        if prev[end] == -1 and start != end: return None
        path = []
        current = end
        while current != -1:
            path.append(current)
            current = prev[current]
        path.reverse()
        return [self.get_vertex_name(v) for v in path]

    def get_path_floyd(self, next_vertex, start, end):
        if next_vertex[start][end] == -1: return None
        path = [start]
        curr = start
        while curr != end:
            curr = next_vertex[curr][end]
            path.append(curr)
        return [self.get_vertex_name(v) for v in path]

    def generate_random_graph(self, density=0.4, min_weight=1, max_weight=20, allow_negative=False):
        self.edges = []
        self.adj_list = defaultdict(list)
        self.adj_matrix = np.full((self.V, self.V), np.inf)
        for i in range(self.V): self.adj_matrix[i][i] = 0
        
        for i in range(self.V):
            for j in range(self.V):
                if i != j and random.random() < density:
                    weight = random.randint(min_weight - 10 if allow_negative else min_weight, max_weight)
                    self.add_edge(i, j, weight)
                    
        if not self.directed and not self.is_connected():
            print("Граф несвязный, добавляем дополнительные рёбра...")
            self.ensure_connectivity(min_weight, max_weight)
        return self

    def is_connected(self):
        if self.directed: return True
        visited = [False] * self.V
        queue = [0]
        visited[0] = True
        while queue:
            v = queue.pop(0)
            for u, _ in self.adj_list[v]:
                if not visited[u]:
                    visited[u] = True
                    queue.append(u)
        return all(visited)

    def ensure_connectivity(self, min_weight, max_weight):
        components = []
        visited = [False] * self.V
        for v in range(self.V):
            if not visited[v]:
                comp = []
                q = [v]
                visited[v] = True
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for u, _ in self.adj_list[curr]:
                        if not visited[u]:
                            visited[u] = True
                            q.append(u)
                components.append(comp)
                
        for i in range(len(components) - 1):
            v1 = components[i][0]
            v2 = components[i+1][0]
            self.add_edge(v1, v2, random.randint(min_weight, max_weight))

    def print_graph(self):
        print("\n" + "="*60)
        print(f"ИНФОРМАЦИЯ О ГРАФЕ ({'ориентированный' if self.directed else 'неориентированный'})")
        print("="*60)
        print(f"Количество вершин: {self.V}")
        print(f"Количество рёбер: {len(self.edges)}")
        print("\nСписок рёбер:")
        seen = set()
        for u, v, w in self.edges:
            if not self.directed:
                edge_key = tuple(sorted([u, v]))
                if edge_key in seen: continue
                seen.add(edge_key)
            arrow = "→" if self.directed else "--"
            print(f"  {self.get_vertex_name(u)} {arrow} {self.get_vertex_name(v)} : {w}")
        print("\nМатрица смежности:")
        self.print_distance_matrix(self.adj_matrix)

class ShortestPathComparator:
    @staticmethod
    def compare_algorithms(graph, start=0):
        print("\n" + "="*60)
        print("СРАВНЕНИЕ АЛГОРИТМОВ ПОИСКА КРАТЧАЙШИХ ПУТЕЙ")
        print("="*60)
        results = {}
        
        # Дейкстра
        try:
            t0 = time.time()
            dist_d, prev_d = graph.dijkstra(start)
            results['dijkstra'] = {'distances': dist_d, 'predecessors': prev_d, 'time': time.time() - t0}
        except Exception as e:
            results['dijkstra'] = {'error': str(e)}
            
        # Беллман-Форд
        t0 = time.time()
        dist_bf, prev_bf = graph.bellman_ford(start)
        results['bellman_ford'] = {'distances': dist_bf, 'predecessors': prev_bf, 'time': time.time() - t0}
        
        # Флойд-Уоршелл
        t0 = time.time()
        dist_fw, next_fw = graph.floyd_warshall()
        results['floyd_warshall'] = {'distances': dist_fw, 'next': next_fw, 'time': time.time() - t0}
        
        print("\n" + "-"*40)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ:")
        print("-"*40)
        if 'distances' in results['bellman_ford'] and 'distances' in results.get('dijkstra', {}):
            match = all(abs(results['dijkstra']['distances'][i] - results['bellman_ford']['distances'][i]) < 1e-6 for i in range(graph.V) if not math.isinf(results['dijkstra']['distances'][i]))
            print(f"\nДейкстра и Беллман-Форд: {'✓ совпадают' if match else '✗ различаются'}")
        print(f"\nВремя выполнения:")
        print(f"  Дейкстра: {results.get('dijkstra', {}).get('time', 'N/A')} сек")
        print(f"  Беллман-Форд: {results['bellman_ford']['time']} сек")
        print(f"  Флойд-Уоршелл: {results['floyd_warshall']['time']} сек")
        return results