import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
import math
import sys
from io import StringIO
from shortest_path_algorithms import Graph, ShortestPathComparator

class ShortestPathGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Алгоритмы поиска кратчайших путей в графе")
        self.root.geometry("1300x750")
        self.graph = None
        self.results = None
        self.pos = None
        self.start_vertex = 0
        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        self.create_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.create_frame, text="Создание графа")
        self.setup_create_tab()
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Визуализация")
        self.setup_viz_tab()
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Результаты")
        self.setup_results_tab()
        self.path_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.path_frame, text="Поиск пути")
        self.setup_path_tab()

    def setup_create_tab(self):
        params = ttk.LabelFrame(self.create_frame, text="Параметры графа", padding=10)
        params.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(params, text="Вершин:").grid(row=0, column=0, padx=5)
        self.v_var = tk.StringVar(value="6")
        ttk.Entry(params, textvariable=self.v_var, width=5).grid(row=0, column=1)
        
        ttk.Label(params, text="Тип:").grid(row=0, column=2, padx=15)
        self.type_var = tk.StringVar(value="undirected")
        ttk.Radiobutton(params, text="Неор.", variable=self.type_var, value="undirected").grid(row=0, column=3)
        ttk.Radiobutton(params, text="Ориент.", variable=self.type_var, value="directed").grid(row=0, column=4)
        
        ttk.Label(params, text="Плотность:").grid(row=1, column=0, padx=5)
        self.dens_var = tk.StringVar(value="0.4")
        ttk.Entry(params, textvariable=self.dens_var, width=5).grid(row=1, column=1)
        
        ttk.Label(params, text="Вес [min, max]:").grid(row=1, column=2, padx=15)
        self.wmin_var = tk.StringVar(value="1")
        ttk.Entry(params, textvariable=self.wmin_var, width=4).grid(row=1, column=3)
        self.wmax_var = tk.StringVar(value="15")
        ttk.Entry(params, textvariable=self.wmax_var, width=4).grid(row=1, column=4)
        
        self.neg_var = tk.BooleanVar()
        ttk.Checkbutton(params, text="Отриц. веса", variable=self.neg_var).grid(row=2, column=0, columnspan=5, pady=5)
        
        btns = ttk.Frame(params)
        btns.grid(row=3, column=0, columnspan=5, pady=5)
        ttk.Button(btns, text="Сгенерировать", command=self.generate_graph).pack(side='left', padx=5)
        ttk.Button(btns, text="Пример", command=self.load_example).pack(side='left', padx=5)
        ttk.Button(btns, text="Очистить", command=self.clear_graph).pack(side='left', padx=5)
        
        edges_frm = ttk.LabelFrame(self.create_frame, text="Добавить ребро", padding=5)
        edges_frm.pack(fill='x', padx=10, pady=5)
        self.u_e = tk.StringVar(); self.v_e = tk.StringVar(); self.w_e = tk.StringVar()
        ttk.Label(edges_frm, text="U:").pack(side='left'); ttk.Entry(edges_frm, textvariable=self.u_e, width=5).pack(side='left', padx=2)
        ttk.Label(edges_frm, text="V:").pack(side='left'); ttk.Entry(edges_frm, textvariable=self.v_e, width=5).pack(side='left', padx=2)
        ttk.Label(edges_frm, text="Вес:").pack(side='left'); ttk.Entry(edges_frm, textvariable=self.w_e, width=5).pack(side='left', padx=2)
        ttk.Button(edges_frm, text="Добавить", command=self.add_edge).pack(side='left', padx=10)
        
        self.info_text = scrolledtext.ScrolledText(self.create_frame, height=8, font=('Courier', 9))
        self.info_text.pack(fill='both', expand=True, padx=10, pady=5)
        ttk.Button(self.create_frame, text="Найти кратчайшие пути", command=self.find_shortest_paths).pack(pady=10)

    def setup_viz_tab(self):
        ctrl = ttk.Frame(self.viz_frame)
        ctrl.pack(fill='x', padx=10, pady=5)
        self.viz_mode = tk.StringVar(value="graph")
        ttk.Label(ctrl, text="Показать:").pack(side='left', padx=5)
        ttk.Radiobutton(ctrl, text="Граф", variable=self.viz_mode, value="graph").pack(side='left')
        ttk.Radiobutton(ctrl, text="Дерево путей", variable=self.viz_mode, value="tree").pack(side='left')
        ttk.Radiobutton(ctrl, text="Матрица", variable=self.viz_mode, value="matrix").pack(side='left')
        self.viz_start = tk.StringVar(value="V1")
        ttk.Label(ctrl, text="Старт:").pack(side='left', padx=(15,5))
        ttk.Entry(ctrl, textvariable=self.viz_start, width=4).pack(side='left')
        ttk.Button(ctrl, text="Обновить", command=self.update_visualization).pack(side='left', padx=15)
        self.plot_frm = ttk.Frame(self.viz_frame)
        self.plot_frm.pack(fill='both', expand=True, padx=10, pady=5)

    def setup_results_tab(self):
        self.res_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, width=80, height=30, font=('Courier', 10))
        self.res_text.pack(fill='both', expand=True, padx=10, pady=10)

    def setup_path_tab(self):
        sel = ttk.LabelFrame(self.path_frame, text="Поиск маршрута", padding=10)
        sel.pack(fill='x', padx=10, pady=5)
        self.ps = tk.StringVar(value="V1"); self.pe = tk.StringVar(value="V6")
        self.pa = tk.StringVar(value="dijkstra")
        ttk.Label(sel, text="От:").grid(row=0, column=0); ttk.Entry(sel, textvariable=self.ps, width=5).grid(row=0, column=1, padx=5)
        ttk.Label(sel, text="До:").grid(row=0, column=2); ttk.Entry(sel, textvariable=self.pe, width=5).grid(row=0, column=3, padx=5)
        ttk.Label(sel, text="Алгоритм:").grid(row=0, column=4)
        ttk.Combobox(sel, textvariable=self.pa, values=["dijkstra","bellman_ford","floyd_warshall"], state="readonly", width=14).grid(row=0, column=5, padx=5)
        ttk.Button(sel, text="Найти", command=self.find_path).grid(row=0, column=6, padx=15)
        self.path_res = scrolledtext.ScrolledText(self.path_frame, height=8)
        self.path_res.pack(fill='both', expand=True, padx=10, pady=5)
        self.path_plot = ttk.Frame(self.path_frame)
        self.path_plot.pack(fill='both', expand=True, padx=10, pady=5)

    def generate_graph(self):
        try:
            v = int(self.v_var.get())
            d = float(self.dens_var.get())
            w1, w2 = int(self.wmin_var.get()), int(self.wmax_var.get())
            if v < 2 or d <= 0 or d > 1: raise ValueError
            self.graph = Graph(v, self.type_var.get() == "directed")
            self.graph.generate_random_graph(d, w1, w2, self.neg_var.get())
            self.pos = nx.spring_layout(self.create_nx_graph(), k=3, iterations=50)
            self.display_info()
            messagebox.showinfo("Успех", f"Граф с {v} вершинами создан")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def load_example(self):
        self.graph = Graph(5, directed=True)
        edges = [(0,1,6),(0,2,2),(1,3,1),(2,1,3),(2,3,5),(3,4,2),(4,2,-4)]
        for u,v,w in edges: self.graph.add_edge(u,v,w)
        self.pos = nx.spring_layout(self.create_nx_graph(), k=2, iterations=50)
        self.display_info()
        messagebox.showinfo("Успех", "Пример загружен")

    def clear_graph(self):
        self.graph = None; self.results = None
        self.info_text.delete(1.0, tk.END); self.res_text.delete(1.0, tk.END)
        self.path_res.delete(1.0, tk.END)
        for w in self.plot_frm.winfo_children(): w.destroy()
        for w in self.path_plot.winfo_children(): w.destroy()

    def add_edge(self):
        if not self.graph:
            try: self.graph = Graph(int(self.v_var.get()), self.type_var.get() == "directed")
            except: return messagebox.showerror("Ошибка", "Сначала создайте граф")
        try:
            u = int(self.u_e.get().replace('V','')) - 1
            v = int(self.v_e.get().replace('V','')) - 1
            w = int(self.w_e.get())
            self.graph.add_edge(u, v, w)
            self.pos = nx.spring_layout(self.create_nx_graph(), k=3, iterations=50)
            self.display_info()
            self.u_e.set(""); self.v_e.set(""); self.w_e.set("")
        except: messagebox.showerror("Ошибка", "Неверный формат")

    def display_info(self):
        if not self.graph: return
        self.info_text.delete(1.0, tk.END)
        old, sys.stdout = sys.stdout, StringIO()
        self.graph.print_graph()
        out = sys.stdout.getvalue(); sys.stdout = old
        self.info_text.insert(tk.END, out)

    def create_nx_graph(self):
        if not self.graph: return None
        G = nx.DiGraph() if self.graph.directed else nx.Graph()
        for i in range(self.graph.V): G.add_node(i, label=self.graph.get_vertex_name(i))
        seen = set()
        for u, v, w in self.graph.edges:
            edge = (u, v) if self.graph.directed else tuple(sorted([u, v]))
            if edge not in seen:
                seen.add(edge)
                G.add_edge(u, v, weight=w)
        return G

    def find_shortest_paths(self):
        if not self.graph: return messagebox.showerror("Ошибка", "Создайте граф")
        try:
            s = int(self.viz_start.get().replace('V','')) - 1
            if not (0 <= s < self.graph.V): raise ValueError
            self.start_vertex = s
        except: return messagebox.showerror("Ошибка", "Неверная стартовая вершина")
        
        old, sys.stdout = sys.stdout, StringIO()
        cmp = ShortestPathComparator()
        self.results = cmp.compare_algorithms(self.graph, s)
        out = sys.stdout.getvalue(); sys.stdout = old
        self.res_text.delete(1.0, tk.END); self.res_text.insert(tk.END, out)
        self.notebook.select(2)
        self.update_visualization()

    def update_visualization(self):
        if not self.graph: return
        for w in self.plot_frm.winfo_children(): w.destroy()
        mode = self.viz_mode.get()
        if mode == "graph": self.plot_original()
        elif mode == "tree": self.plot_tree()
        else: self.plot_matrix()

    def plot_original(self):
        fig, ax = plt.subplots(figsize=(9, 6))
        G = self.create_nx_graph()
        labels = {i: self.graph.get_vertex_name(i) for i in range(self.graph.V)}
        if self.graph.directed:
            nx.draw(G, self.pos, labels=labels, node_color='lightblue', node_size=400, ax=ax, arrows=True)
        else:
            nx.draw(G, self.pos, labels=labels, node_color='lightblue', node_size=400, ax=ax)
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels={(u,v):d['weight'] for u,v,d in G.edges(data=True)}, ax=ax)
        ax.set_title("Исходный граф", fontsize=14); ax.axis('off')
        FigureCanvasTkAgg(fig, self.plot_frm).get_tk_widget().pack(fill='both', expand=True)

    def plot_tree(self):
        if not self.results or 'dijkstra' not in self.results: return
        fig, ax = plt.subplots(figsize=(9, 6))
        G = self.create_nx_graph()
        nx.draw(G, self.pos, node_color='lightblue', node_size=400, ax=ax)
        prev = self.results['dijkstra']['predecessors']
        tree = [(prev[i], i) for i in range(self.graph.V) if prev[i] != -1]
        nx.draw_networkx_edges(G, self.pos, edgelist=tree, width=3, edge_color='red', ax=ax)
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels={(u,v):d['weight'] for u,v,d in G.edges(data=True)}, ax=ax)
        ax.set_title(f"Дерево кратчайших путей от {self.graph.get_vertex_name(self.start_vertex)}", fontsize=14); ax.axis('off')
        FigureCanvasTkAgg(fig, self.plot_frm).get_tk_widget().pack(fill='both', expand=True)

    def plot_matrix(self):
        if not self.results or 'floyd_warshall' not in self.results: return
        fig, ax = plt.subplots(figsize=(9, 6))
        dist = self.results['floyd_warshall']['distances']
        plot_m = dist.copy()
        mx = np.max(plot_m[~np.isinf(plot_m)])
        plot_m[np.isinf(plot_m)] = mx * 1.5
        im = ax.imshow(plot_m, cmap='YlOrRd', aspect='auto')
        for i in range(self.graph.V):
            for j in range(self.graph.V):
                txt = "∞" if np.isinf(dist[i][j]) else f"{dist[i][j]:.0f}"
                ax.text(j, i, txt, ha="center", va="center", color="black" if np.isinf(dist[i][j]) else "white" if dist[i][j] > mx/2 else "black")
        ax.set_xticks(range(self.graph.V)); ax.set_yticks(range(self.graph.V))
        ax.set_xticklabels([self.graph.get_vertex_name(i) for i in range(self.graph.V)])
        ax.set_yticklabels([self.graph.get_vertex_name(i) for i in range(self.graph.V)])
        ax.set_title("Матрица кратчайших расстояний", fontsize=14); plt.colorbar(im); FigureCanvasTkAgg(fig, self.plot_frm).get_tk_widget().pack(fill='both', expand=True)

    def find_path(self):
        if not self.graph or not self.results: return messagebox.showwarning("Внимание", "Сначала запустите алгоритмы")
        try:
            s = int(self.ps.get().replace('V',''))-1
            e = int(self.pe.get().replace('V',''))-1
            algo = self.pa.get()
            res = self.results.get(algo, {})
            
            path, dist = None, None
            if algo == "dijkstra":
                dist = res['distances'][e]
                if not math.isinf(dist): path = self.graph.get_path(res['predecessors'], s, e)
            elif algo == "bellman_ford":
                if res['distances'] is None: return self.path_res.insert(tk.END, "Обнаружен отрицательный цикл!")
                dist = res['distances'][e]
                if not math.isinf(dist): path = self.graph.get_path(res['predecessors'], s, e)
            elif algo == "floyd_warshall":
                if res['distances'] is None: return self.path_res.insert(tk.END, "Обнаружен отрицательный цикл!")
                dist = res['distances'][s][e]
                if not math.isinf(dist): path = self.graph.get_path_floyd(res['next'], s, e)
                
            self.path_res.delete(1.0, tk.END)
            if path:
                self.path_res.insert(tk.END, f"Маршрут: {' → '.join(path)}\nРасстояние: {dist}")
                self.plot_specific_path(path, s, e, algo)
            else:
                self.path_res.insert(tk.END, f"Путь от {self.graph.get_vertex_name(s)} до {self.graph.get_vertex_name(e)} не существует")
        except Exception as ex:
            messagebox.showerror("Ошибка", str(ex))

    def plot_specific_path(self, path, s, e, algo_name):
        for w in self.path_plot.winfo_children(): w.destroy()
        fig, ax = plt.subplots(figsize=(9, 5))
        G = self.create_nx_graph()
        idxs = [self.graph.vertices_names.index(v) for v in path]
        colors = ['lightblue']*self.graph.V; colors[s]='green'; colors[e]='red'
        nx.draw(G, self.pos, node_color=colors, node_size=400, ax=ax)
        edges = [(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
        nx.draw_networkx_edges(G, self.pos, edgelist=edges, width=3, edge_color='red', ax=ax)
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels={(u,v):d['weight'] for u,v,d in G.edges(data=True)}, ax=ax)
        ax.set_title(f"{algo_name.replace('_',' ').title()}: {' → '.join(path)}", fontsize=12); ax.axis('off')
        FigureCanvasTkAgg(fig, self.path_plot).get_tk_widget().pack(fill='both', expand=True)

def main():
    root = tk.Tk()
    ttk.Style().configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
    ShortestPathGUI(root)
    root.mainloop()