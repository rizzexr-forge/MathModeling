import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
import sys
from io import StringIO
from route_algorithms import RouteFinder, MapCreator

class RouteGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Кратчайший маршрут в транспортной сети")
        self.root.geometry("1300x800")
        self.graph = None
        self.pos = None
        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Вкладки
        self.map_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.map_frame, text="Карта маршрутов")
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="Поиск маршрута")
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Визуализация")
        self.compare_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.compare_frame, text="Сравнение алгоритмов")

        self._setup_map_tab()
        self._setup_search_tab()
        self._setup_viz_tab()
        self._setup_compare_tab()

    def _setup_map_tab(self):
        ctrl = ttk.LabelFrame(self.map_frame, text="Выбор карты", padding=10)
        ctrl.pack(fill='x', padx=10, pady=5)
        self.map_type = tk.StringVar(value="belarus")
        ttk.Radiobutton(ctrl, text="Города Беларуси", variable=self.map_type, value="belarus").pack(side='left', padx=5)
        ttk.Radiobutton(ctrl, text="Метро", variable=self.map_type, value="metro").pack(side='left', padx=5)
        ttk.Radiobutton(ctrl, text="Случайная сеть", variable=self.map_type, value="random").pack(side='left', padx=5)

        rnd = ttk.Frame(ctrl)
        rnd.pack(side='left', padx=10)
        ttk.Label(rnd, text="Вершин:").pack(side='left')
        self.rnd_v = tk.StringVar(value="8")
        ttk.Entry(rnd, textvariable=self.rnd_v, width=4).pack(side='left', padx=3)
        ttk.Label(rnd, text="Плотность:").pack(side='left', padx=(10,0))
        self.rnd_d = tk.StringVar(value="0.35")
        ttk.Entry(rnd, textvariable=self.rnd_d, width=4).pack(side='left', padx=3)

        ttk.Button(ctrl, text="Загрузить карту", command=self.load_map).pack(side='left', padx=15)
        self.map_info = scrolledtext.ScrolledText(self.map_frame, height=6, font=('Courier', 9))
        self.map_info.pack(fill='x', padx=10, pady=5)
        self.map_plot = ttk.Frame(self.map_frame)
        self.map_plot.pack(fill='both', expand=True, padx=10, pady=5)

    def _setup_search_tab(self):
        p = ttk.LabelFrame(self.search_frame, text="Маршрут", padding=10)
        p.pack(fill='x', padx=10, pady=5)
        self.start_cb = ttk.Combobox(p, state='readonly', width=18)
        ttk.Label(p, text="От:").grid(row=0, column=0, padx=5)
        self.start_cb.grid(row=0, column=1, padx=5)
        self.end_cb = ttk.Combobox(p, state='readonly', width=18)
        ttk.Label(p, text="До:").grid(row=0, column=2, padx=5)
        self.end_cb.grid(row=0, column=3, padx=5)
        self.algo_cb = ttk.Combobox(p, values=["dijkstra","a_star","floyd"], state='readonly', width=10)
        self.algo_cb.set("a_star")
        ttk.Label(p, text="Алгоритм:").grid(row=0, column=4, padx=5)
        self.algo_cb.grid(row=0, column=5, padx=5)
        ttk.Button(p, text="Найти", command=self.find_route).grid(row=0, column=6, padx=15)

        self.res_txt = scrolledtext.ScrolledText(self.search_frame, height=10, font=('Courier', 9))
        self.res_txt.pack(fill='both', expand=True, padx=10, pady=5)
        self.res_plot = ttk.Frame(self.search_frame)
        self.res_plot.pack(fill='both', expand=True, padx=10, pady=5)

    def _setup_viz_tab(self):
        ctrl = ttk.Frame(self.viz_frame)
        ctrl.pack(fill='x', padx=10, pady=5)
        self.viz_mode = tk.StringVar(value="graph")
        ttk.Label(ctrl, text="Показать:").pack(side='left', padx=5)
        ttk.Radiobutton(ctrl, text="Сеть", variable=self.viz_mode, value="graph").pack(side='left')
        ttk.Radiobutton(ctrl, text="Дерево путей", variable=self.viz_mode, value="tree").pack(side='left')
        ttk.Radiobutton(ctrl, text="Матрица", variable=self.viz_mode, value="matrix").pack(side='left')
        self.viz_start_cb = ttk.Combobox(ctrl, state='readonly', width=15)
        ttk.Label(ctrl, text="Старт:").pack(side='left', padx=(15,5))
        self.viz_start_cb.pack(side='left', padx=5)
        ttk.Button(ctrl, text="Обновить", command=self.update_viz).pack(side='left', padx=15)
        self.viz_plot = ttk.Frame(self.viz_frame)
        self.viz_plot.pack(fill='both', expand=True, padx=10, pady=5)

    def _setup_compare_tab(self):
        p = ttk.LabelFrame(self.compare_frame, text="Сравнение", padding=10)
        p.pack(fill='x', padx=10, pady=5)
        self.c_start = ttk.Combobox(p, state='readonly', width=15)
        ttk.Label(p, text="От:").grid(row=0, column=0, padx=5)
        self.c_start.grid(row=0, column=1, padx=5)
        self.c_end = ttk.Combobox(p, state='readonly', width=15)
        ttk.Label(p, text="До:").grid(row=0, column=2, padx=5)
        self.c_end.grid(row=0, column=3, padx=5)
        ttk.Button(p, text="Сравнить", command=self.compare).grid(row=0, column=4, padx=15)
        self.c_res = scrolledtext.ScrolledText(self.compare_frame, height=12, font=('Courier', 9))
        self.c_res.pack(fill='both', expand=True, padx=10, pady=5)
        self.c_plot = ttk.Frame(self.compare_frame)
        self.c_plot.pack(fill='both', expand=True, padx=10, pady=5)

    def _update_combos(self):
        names = self.graph.vertices_names if self.graph else []
        for cb in [self.start_cb, self.end_cb, self.viz_start_cb, self.c_start, self.c_end]:
            cb['values'] = names
            if names: cb.set(names[0])

    def load_map(self):
        mtype = self.map_type.get()
        if mtype == "belarus": self.graph = MapCreator.create_city_map()
        elif mtype == "metro": self.graph = MapCreator.create_metro_map()
        else:
            try: self.graph = RouteFinder(int(self.rnd_v.get()), False).generate_random_graph(float(self.rnd_d.get()))
            except: return messagebox.showerror("Ошибка", "Неверные параметры")
        self._update_combos()
        old, sys.stdout = sys.stdout, StringIO()
        self.graph.print_graph()
        self.map_info.delete(1.0, tk.END); self.map_info.insert(tk.END, sys.stdout.getvalue()); sys.stdout = old
        self.viz_mode.set("graph"); self.update_viz()
        messagebox.showinfo("Успех", f"Карта загружена: {self.graph.V} узлов")

    def _get_coords_pos(self):
        return {i: self.graph.coordinates[i] for i in range(self.graph.V)}

    def _draw_base_graph(self, ax, highlight_edges=None):
        G = nx.Graph()
        for i in range(self.graph.V): G.add_node(i, label=self.graph.get_vertex_name(i))
        seen = set()
        for u,v,w in self.graph.edges:
            e = (u,v) if self.graph.directed else tuple(sorted([u,v]))
            if e not in seen: seen.add(e); G.add_edge(u,v,weight=w)
        pos = self._get_coords_pos()
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=400, ax=ax)
        nx.draw_networkx_labels(G, pos, {i: self.graph.get_vertex_name(i) for i in range(self.graph.V)}, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        if highlight_edges:
            nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=3, edge_color='red', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):d['weight'] for u,v,d in G.edges(data=True)}, font_size=8, ax=ax)
        return pos

    def find_route(self):
        if not self.graph: return messagebox.showerror("Ошибка", "Загрузите карту")
        try:
            s = self.graph.vertices_names.index(self.start_cb.get())
            e = self.graph.vertices_names.index(self.end_cb.get())
        except: return messagebox.showerror("Ошибка", "Выберите точки")
        algo = self.algo_cb.get()
        self.res_txt.delete(1.0, tk.END)
        edges_hl = []
        if algo == "dijkstra":
            dist, prev = self.graph.dijkstra(s, e)
            path = self.graph.get_path(prev, s, e)
            self.res_txt.insert(tk.END, f"ДЕЙКСТРА\nПуть: {' → '.join(path) if path else 'Не найден'}\nРасстояние: {dist[e]:.2f}\n")
            if path: edges_hl = [(self.graph.vertices_names.index(a), self.graph.vertices_names.index(b)) for a,b in zip(path, path[1:])]
        elif algo == "a_star":
            path, dist = self.graph.a_star(s, e)
            names = [self.graph.get_vertex_name(v) for v in path] if path else []
            self.res_txt.insert(tk.END, f"A*\nПуть: {' → '.join(names)}\nРасстояние: {dist:.2f if dist else '∞'}\n")
            if path: edges_hl = [(path[i], path[i+1]) for i in range(len(path)-1)]
        else:
            dist, nxt = self.graph.floyd_warshall()
            if dist is None: self.res_txt.insert(tk.END, "Обнаружен отрицательный цикл!")
            else:
                path = self.graph.get_path_floyd(nxt, s, e)
                self.res_txt.insert(tk.END, f"ФЛОЙД-УОРШЕЛЛ\nПуть: {' → '.join(path) if path else 'Не найден'}\nРасстояние: {dist[s][e]:.2f}\n")
                if path: edges_hl = [(self.graph.vertices_names.index(a), self.graph.vertices_names.index(b)) for a,b in zip(path, path[1:])]

        for w in self.res_plot.winfo_children(): w.destroy()
        fig, ax = plt.subplots(figsize=(9,5))
        self._draw_base_graph(ax, highlight_edges=edges_hl)
        ax.set_title(f"Маршрут: {self.start_cb.get()} → {self.end_cb.get()} ({algo.upper()})")
        ax.axis('off')
        FigureCanvasTkAgg(fig, self.res_plot).get_tk_widget().pack(fill='both', expand=True)

    def update_viz(self):
        if not self.graph: return
        for w in self.viz_plot.winfo_children(): w.destroy()
        fig, ax = plt.subplots(figsize=(9,6))
        mode = self.viz_mode.get()
        if mode == "graph":
            self._draw_base_graph(ax)
            ax.set_title("Транспортная сеть")
        elif mode == "tree":
            try: s = self.graph.vertices_names.index(self.viz_start_cb.get())
            except: return messagebox.showerror("Ошибка", "Выберите старт")
            self._draw_base_graph(ax)
            _, prev = self.graph.dijkstra(s)
            tree = [(prev[i], i) for i in range(self.graph.V) if prev[i]!=-1]
            pos = self._get_coords_pos()
            nx.draw_networkx_edges(nx.Graph(), pos, edgelist=tree, width=2, edge_color='red', ax=ax)
            ax.set_title(f"Дерево кратчайших путей от {self.graph.get_vertex_name(s)}")
        else:
            dist, _ = self.graph.floyd_warshall()
            if dist is None: return messagebox.showerror("Ошибка", "Отрицательный цикл")
            mx = np.max(dist[~np.isinf(dist)])
            plot = dist.copy(); plot[np.isinf(plot)] = mx*1.5
            im = ax.imshow(plot, cmap='YlOrRd', aspect='auto')
            for i in range(self.graph.V):
                for j in range(self.graph.V):
                    txt = "∞" if np.isinf(dist[i][j]) else f"{dist[i][j]:.0f}"
                    ax.text(j, i, txt, ha="center", va="center", color="black" if np.isinf(dist[i][j]) else "white" if dist[i][j]>mx/2 else "black")
            ax.set_xticks(range(self.graph.V)); ax.set_yticks(range(self.graph.V))
            ax.set_xticklabels(self.graph.vertices_names, rotation=45, ha='right')
            ax.set_yticklabels(self.graph.vertices_names)
            ax.set_title("Матрица расстояний")
            plt.colorbar(im, ax=ax)
        ax.axis('off')
        FigureCanvasTkAgg(fig, self.viz_plot).get_tk_widget().pack(fill='both', expand=True)

    def compare(self):
        if not self.graph: return messagebox.showerror("Ошибка", "Загрузите карту")
        try:
            s = self.graph.vertices_names.index(self.c_start.get())
            e = self.graph.vertices_names.index(self.c_end.get())
        except: return messagebox.showerror("Ошибка", "Выберите точки")
        old, sys.stdout = sys.stdout, StringIO()
        res = self.graph.compare_algorithms(s, e)
        out = sys.stdout.getvalue(); sys.stdout = old
        self.c_res.delete(1.0, tk.END); self.c_res.insert(tk.END, out)

        for w in self.c_plot.winfo_children(): w.destroy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
        algos = ['Дейкстра', 'A*', 'Флойд']
        times = [res['dijkstra']['time']*1000, res['a_star']['time']*1000, res['floyd']['time']*1000]
        dists = [res['dijkstra']['dist'], res['a_star']['dist'], res['floyd']['dist']]
        colors = ['#66b3ff','#99ff99','#ffcc99']
        ax1.bar(algos, times, color=colors); ax1.set_ylabel("Время (мс)"); ax1.set_title("Сравнение времени")
        ax1.grid(True, alpha=0.3)
        ax2.bar(algos, dists, color=colors); ax2.set_ylabel("Расстояние"); ax2.set_title("Найденные расстояния")
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f"Сравнение: {self.c_start.get()} → {self.c_end.get()}")
        plt.tight_layout()
        FigureCanvasTkAgg(fig, self.c_plot).get_tk_widget().pack(fill='both', expand=True)

def main():
    root = tk.Tk()
    RouteGUI(root)
    root.mainloop()