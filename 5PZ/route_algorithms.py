import numpy as np
import heapq
import math
import time
import random
from collections import defaultdict

class RouteFinder:
    """Класс для поиска кратчайших маршрутов в графе"""
    def __init__(self, vertices_count, directed=False):
        self.V = vertices_count
        self.directed = directed
        self.vertices_names = [f"Point_{i+1}" for i in range(vertices_count)]
        self.edges = []
        self.adj_list = defaultdict(list)
        self.adj_matrix = np.full((vertices_count, vertices_count), np.inf)
        for i in range(vertices_count):
            self.adj_matrix[i][i] = 0
        self.coordinates = self._generate_coordinates()

    def _generate_coordinates(self):
        """Генерация координат вершин для эвристики A*"""
        np.random.seed(42)
        return {i: (np.random.uniform(10, 90), np.random.uniform(10, 90)) for i in range(self.V)}

    def set_coordinates(self, coords_dict):
        """Установка пользовательских координат"""
        self.coordinates = coords_dict

    def add_edge(self, u, v, weight):
        """Добавление ребра"""
        self.edges.append((u, v, weight))
        self.adj_list[u].append((v, weight))
        self.adj_matrix[u][v] = weight
        if not self.directed:
            self.adj_list[v].append((u, weight))
            self.adj_matrix[v][u] = weight
            self.edges.append((v, u, weight))

    def add_edge_named(self, u_name, v_name, weight):
        u = int(u_name.replace('Point_', '')) - 1
        v = int(v_name.replace('Point_', '')) - 1
        self.add_edge(u, v, weight)

    def get_vertex_name(self, v):
        return self.vertices_names[v]

    def _heuristic(self, node, goal):
        """Евклидово расстояние для A*"""
        x1, y1 = self.coordinates[node]
        x2, y2 = self.coordinates[goal]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def dijkstra(self, start, end=None):
        """Алгоритм Дейкстры"""
        if not (0 <= start < self.V):
            raise ValueError("Неверная начальная вершина")
        dist = [math.inf] * self.V
        prev = [-1] * self.V
        visited = [False] * self.V
        dist[start] = 0
        pq = [(0, start)]

        print("\n" + "="*70)
        print("АЛГОРИТМ ДЕЙКСТРЫ")
        print("="*70)
        print(f"Старт: {self.get_vertex_name(start)}")
        if end is not None: print(f"Цель: {self.get_vertex_name(end)}")
        print("\nШаг | Обрабатываемая вершина | Расстояние")
        print("-"*50)
        step = 1
        while pq:
            current_dist, u = heapq.heappop(pq)
            if visited[u]: continue
            visited[u] = True
            print(f"{step:2} | {self.get_vertex_name(u):18} | {current_dist:10.2f}")
            if end is not None and u == end:
                print("\nЦелевая вершина достигнута!")
                break
            for v, weight in self.adj_list[u]:
                if not visited[v] and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
            step += 1
        return dist, prev

    def a_star(self, start, end):
        """Алгоритм A* (A-star)"""
        if not (0 <= start < self.V and 0 <= end < self.V):
            raise ValueError("Неверные вершины")
        g_score = [math.inf] * self.V
        g_score[start] = 0
        f_score = [math.inf] * self.V
        f_score[start] = self._heuristic(start, end)
        prev = [-1] * self.V
        open_set = [(f_score[start], start)]
        open_set_hash = {start}
        closed_set = set()

        print("\n" + "="*70)
        print("АЛГОРИТМ A* (A-STAR)")
        print("="*70)
        print(f"Старт: {self.get_vertex_name(start)}")
        print(f"Цель: {self.get_vertex_name(end)}")
        print("\nШаг | Текущая вершина | g(x) | h(x) | f(x)")
        print("-"*55)
        step = 1
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            if current in closed_set: continue
            closed_set.add(current)
            h_val = self._heuristic(current, end)
            print(f"{step:2} | {self.get_vertex_name(current):12} | {g_score[current]:6.2f} | {h_val:6.2f} | {g_score[current]+h_val:6.2f}")
            if current == end:
                print("\nЦелевая вершина достигнута!")
                break
            for neighbor, weight in self.adj_list[current]:
                if neighbor in closed_set: continue
                tentative_g = g_score[current] + weight
                if tentative_g < g_score[neighbor]:
                    prev[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
            step += 1

        if g_score[end] == math.inf:
            print("\nПуть не найден!")
            return None, None
        path = []
        curr = end
        while curr != -1:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        return path, g_score[end]

    def floyd_warshall(self):
        """Алгоритм Флойда-Уоршелла"""
        dist = self.adj_matrix.copy()
        next_vertex = np.full((self.V, self.V), -1, dtype=int)
        for i in range(self.V):
            for j in range(self.V):
                if i != j and not np.isinf(self.adj_matrix[i][j]):
                    next_vertex[i][j] = j

        print("\n" + "="*70)
        print("АЛГОРИТМ ФЛОЙДА-УОРШЕЛЛА (Все пары вершин)")
        print("="*70)
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_vertex[i][j] = next_vertex[i][k]

        for i in range(self.V):
            if dist[i][i] < 0:
                print("\n⚠ Обнаружен отрицательный цикл!")
                return None, None
        return dist, next_vertex

    def get_path(self, prev, start, end):
        if prev[end] == -1: return None
        path = []
        curr = end
        while curr != -1:
            path.append(curr)
            curr = prev[curr]
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

    def compare_algorithms(self, start, end):
        """Сравнение производительности"""
        print("\n" + "="*70)
        print("СРАВНЕНИЕ АЛГОРИТМОВ ПОИСКА КРАТЧАЙШЕГО МАРШРУТА")
        print("="*70)
        results = {}
        t0 = time.time()
        dist_d, prev_d = self.dijkstra(start, end)
        results['dijkstra'] = {'time': time.time()-t0, 'dist': dist_d[end], 'path': self.get_path(prev_d, start, end)}
        t0 = time.time()
        path_a, dist_a = self.a_star(start, end)
        results['a_star'] = {'time': time.time()-t0, 'dist': dist_a, 'path': [self.get_vertex_name(v) for v in path_a] if path_a else None}
        t0 = time.time()
        dist_f, next_f = self.floyd_warshall()
        results['floyd'] = {'time': time.time()-t0, 'dist': dist_f[start][end] if dist_f is not None else math.inf, 'path': self.get_path_floyd(next_f, start, end) if next_f is not None else None}

        print("\n" + "-"*50)
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
        print("-"*50)
        for algo, data in results.items():
            print(f"\n{algo.upper()}:")
            print(f"  • Время: {data['time']*1000:.3f} мс")
            print(f"  • Расстояние: {data['dist']:.2f if data['dist']!=math.inf else '∞'}")
            print(f"  • Путь: {' → '.join(data['path']) if data['path'] else 'Не найден'}")
        return results

    def generate_random_graph(self, density=0.4, min_weight=5, max_weight=50):
        self.edges = []
        self.adj_list = defaultdict(list)
        self.adj_matrix = np.full((self.V, self.V), np.inf)
        for i in range(self.V): self.adj_matrix[i][i] = 0
        for i in range(self.V):
            for j in range(i+1, self.V):
                if random.random() < density:
                    w = random.randint(min_weight, max_weight)
                    self.add_edge(i, j, w)
        self._ensure_connectivity(min_weight, max_weight)
        return self

    def _ensure_connectivity(self, min_w, max_w):
        visited = [False]*self.V
        def dfs(v):
            visited[v] = True
            for u, _ in self.adj_list[v]:
                if not visited[u]: dfs(u)
        dfs(0)
        for i in range(self.V):
            if not visited[i]:
                self.add_edge(0, i, random.randint(min_w, max_w))

    def print_graph(self):
        print("\n" + "="*70)
        print("ИНФОРМАЦИЯ О ТРАНСПОРТНОЙ СЕТИ")
        print("="*70)
        print(f"Узлов: {self.V} | Маршрутов: {len(self.edges)//2 if not self.directed else len(self.edges)}")
        print(f"Тип: {'Ориентированная' if self.directed else 'Неориентированная'}")
        print("\nМаршруты:")
        seen = set()
        for u, v, w in self.edges:
            edge = (min(u,v), max(u,v)) if not self.directed else (u,v)
            if edge in seen: continue
            seen.add(edge)
            print(f"  {self.get_vertex_name(u)} {'→' if self.directed else '--'} {self.get_vertex_name(v)}: {w}")
        print("\nКоординаты:")
        for i in range(self.V):
            print(f"  {self.get_vertex_name(i)}: ({self.coordinates[i][0]:.1f}, {self.coordinates[i][1]:.1f})")

class MapCreator:
    @staticmethod
    def create_city_map():
        g = RouteFinder(7, directed=False)
        g.vertices_names = ["Минск","Брест","Витебск","Гомель","Гродно","Могилёв","Бобруйск"]
        coords = {0:(53.9,27.56), 1:(52.09,23.73), 2:(55.19,30.20), 3:(52.43,30.97), 4:(53.67,23.82), 5:(53.89,30.33), 6:(53.13,29.22)}
        g.set_coordinates(coords)
        edges = [(0,1,350),(0,2,280),(0,3,310),(0,4,270),(0,5,200),(0,6,140),(1,4,200),(2,5,150),(3,5,160),(3,6,120)]
        for u,v,w in edges: g.add_edge(u,v,w)
        return g

    @staticmethod
    def create_metro_map():
        g = RouteFinder(12, directed=False)
        g.vertices_names = ["Каменная горка","Кунцевщина","Спортивная","Пушкинская","Молодёжная","Октябрьская","Фрунзенская","Купаловская","Немига","Первомайская","Автозаводская","Могилёвская"]
        coords = {0:(0,0),1:(1,0),2:(2,0),3:(3,0),4:(3,1),5:(2,1),6:(1,1),7:(0,1),8:(0,2),9:(1,2),10:(2,2),11:(3,2)}
        g.set_coordinates(coords)
        edges = [(0,1,2),(1,2,2),(2,3,2),(3,4,2),(4,5,2),(5,6,2),(6,7,2),(7,8,2),(8,9,2),(9,10,2),(10,11,2),(0,7,3),(1,6,3),(2,5,3),(7,8,3),(6,9,3),(5,10,3),(4,11,3)]
        for u,v,w in edges: g.add_edge(u,v,w)
        return g