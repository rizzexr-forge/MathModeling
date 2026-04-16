import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
import sys
from io import StringIO
from mst_algorithms import Graph, MSTComparator

class MSTGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Алгоритмы Прима и Краскала - Минимальное остовное дерево")
        self.root.geometry("1300x750")

        self.graph = None
        self.kruskal_result = None
        self.prim_result = None
        self.pos = None

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

        self.compare_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.compare_frame, text="Сравнение")
        self.setup_compare_tab()

    def setup_create_tab(self):
        params_frame = ttk.LabelFrame(self.create_frame, text="Параметры графа", padding=10)
        params_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(params_frame, text="Количество вершин:").grid(row=0, column=0, padx=5, pady=5)
        self.vertices_var = tk.StringVar(value="6")
        ttk.Entry(params_frame, textvariable=self.vertices_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Плотность (0-1):").grid(row=0, column=2, padx=20, pady=5)
        self.density_var = tk.StringVar(value="0.5")
        ttk.Entry(params_frame, textvariable=self.density_var, width=10).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(params_frame, text="Мин. вес:").grid(row=0, column=4, padx=20, pady=5)
        self.min_weight_var = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.min_weight_var, width=10).grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(params_frame, text="Макс. вес:").grid(row=0, column=6, padx=5, pady=5)
        self.max_weight_var = tk.StringVar(value="15")
        ttk.Entry(params_frame, textvariable=self.max_weight_var, width=10).grid(row=0, column=7, padx=5, pady=5)

        btn_frame = ttk.Frame(params_frame)
        btn_frame.grid(row=1, column=0, columnspan=8, pady=10)
        ttk.Button(btn_frame, text="Сгенерировать случайный граф", command=self.generate_random_graph).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Загрузить пример", command=self.load_example).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Очистить", command=self.clear_graph).pack(side='left', padx=5)

        edges_frame = ttk.LabelFrame(self.create_frame, text="Ручное добавление рёбер", padding=10)
        edges_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(edges_frame, text="Вершина 1:").grid(row=0, column=0, padx=5, pady=5)
        self.u_var = tk.StringVar()
        ttk.Entry(edges_frame, textvariable=self.u_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(edges_frame, text="Вершина 2:").grid(row=0, column=2, padx=5, pady=5)
        self.v_var = tk.StringVar()
        ttk.Entry(edges_frame, textvariable=self.v_var, width=10).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(edges_frame, text="Вес:").grid(row=0, column=4, padx=5, pady=5)
        self.weight_var = tk.StringVar()
        ttk.Entry(edges_frame, textvariable=self.weight_var, width=10).grid(row=0, column=5, padx=5, pady=5)

        ttk.Button(edges_frame, text="Добавить ребро", command=self.add_edge).grid(row=0, column=6, padx=20, pady=5)

        self.graph_display = scrolledtext.ScrolledText(self.create_frame, height=12, font=('Courier', 9))
        self.graph_display.pack(fill='both', expand=True, padx=10, pady=5)

        ttk.Button(self.create_frame, text="Найти минимальное остовное дерево", command=self.find_mst, style='Accent.TButton').pack(pady=10)

    def setup_viz_tab(self):
        control_frame = ttk.Frame(self.viz_frame)
        control_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(control_frame, text="Показать:").pack(side='left', padx=5)
        self.viz_method = tk.StringVar(value="original")
        ttk.Radiobutton(control_frame, text="Исходный граф", variable=self.viz_method, value="original").pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Алгоритм Краскала", variable=self.viz_method, value="kruskal").pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Алгоритм Прима", variable=self.viz_method, value="prim").pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Сравнение", variable=self.viz_method, value="both").pack(side='left', padx=5)
        ttk.Button(control_frame, text="Обновить", command=self.update_visualization).pack(side='left', padx=20)

        self.plot_frame = ttk.Frame(self.viz_frame)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=5)

    def setup_results_tab(self):
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, width=80, height=30, font=('Courier', 10))
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        self.results_text.insert(tk.END, "РЕЗУЛЬТАТЫ РАБОТЫ АЛГОРИТМОВ\n" + "="*70 + "\n\n")
        self.results_text.insert(tk.END, "Здесь будут отображаться результаты после нахождения MST.\n")

    def setup_compare_tab(self):
        stats_frame = ttk.LabelFrame(self.compare_frame, text="Сравнительная статистика", padding=10)
        stats_frame.pack(fill='x', padx=10, pady=5)
        self.stats_text = tk.Text(stats_frame, height=12, width=60, font=('Courier', 10))
        self.stats_text.pack()

        compare_plot_frame = ttk.LabelFrame(self.compare_frame, text="Визуальное сравнение", padding=10)
        compare_plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.compare_plot_frame = ttk.Frame(compare_plot_frame)
        self.compare_plot_frame.pack(fill='both', expand=True)

    def generate_random_graph(self):
        try:
            vertices = int(self.vertices_var.get())
            density = float(self.density_var.get())
            min_weight = int(self.min_weight_var.get())
            max_weight = int(self.max_weight_var.get())

            if vertices < 2:
                messagebox.showerror("Ошибка", "Количество вершин должно быть не менее 2")
                return
            if not (0 < density <= 1):
                messagebox.showerror("Ошибка", "Плотность должна быть в диапазоне (0, 1]")
                return

            self.graph = Graph(vertices)
            self.graph.generate_random_graph(density, min_weight, max_weight)
            self.pos = nx.spring_layout(self.create_nx_graph(), k=2, iterations=50)
            self.display_graph_info()
            messagebox.showinfo("Успех", f"Сгенерирован граф с {vertices} вершинами и {len(self.graph.edges)} рёбрами")
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте правильность введённых данных")

    def load_example(self):
        self.graph = Graph(6)
        edges = [(0, 1, 7), (0, 2, 9), (0, 5, 14), (1, 2, 10), (1, 3, 15), 
                 (2, 3, 11), (2, 5, 2), (3, 4, 6), (4, 5, 9)]
        for u, v, w in edges:
            self.graph.add_edge(u, v, w)
        self.pos = nx.spring_layout(self.create_nx_graph(), k=2, iterations=50)
        self.display_graph_info()
        messagebox.showinfo("Успех", "Загружен пример графа")

    def clear_graph(self):
        self.graph = None
        self.kruskal_result = None
        self.prim_result = None
        self.graph_display.delete(1.0, tk.END)

    def add_edge(self):
        if self.graph is None:
            try:
                vertices = int(self.vertices_var.get())
                self.graph = Graph(vertices)
            except:
                messagebox.showerror("Ошибка", "Сначала создайте граф")
                return
        try:
            u_name = self.u_var.get().strip().upper()
            v_name = self.v_var.get().strip().upper()
            weight = int(self.weight_var.get())
            
            u = int(u_name.replace('V', '')) - 1
            v = int(v_name.replace('V', '')) - 1

            if not (0 <= u < self.graph.V and 0 <= v < self.graph.V):
                messagebox.showerror("Ошибка", "Неверный номер вершины")
                return
                
            self.graph.add_edge(u, v, weight)
            self.pos = nx.spring_layout(self.create_nx_graph(), k=2, iterations=50)
            self.display_graph_info()
            self.u_var.set(""); self.v_var.set(""); self.weight_var.set("")
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте правильность введённых данных")

    def display_graph_info(self):
        if self.graph is None: return
        self.graph_display.delete(1.0, tk.END)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        self.graph.print_graph()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        self.graph_display.insert(tk.END, output)

    def create_nx_graph(self):
        if self.graph is None: return None
        G = nx.Graph()
        for i in range(self.graph.V):
            G.add_node(i, label=self.graph.get_vertex_name(i))
        for weight, u, v in self.graph.edges:
            G.add_edge(u, v, weight=weight)
        return G

    def find_mst(self):
        if self.graph is None:
            messagebox.showerror("Ошибка", "Сначала создайте граф")
            return
            
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        comparator = MSTComparator()
        results = comparator.compare_algorithms(self.graph)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        self.kruskal_result = results['kruskal']
        self.prim_result = results['prim']
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, output)
        self.update_statistics()
        self.notebook.select(2)
        self.update_visualization()

    def update_statistics(self):
        if self.kruskal_result is None or self.prim_result is None: return
        self.stats_text.delete(1.0, tk.END)
        k_w = self.kruskal_result['weight']
        p_w = self.prim_result['weight']
        diff = abs(k_w - p_w)
        match = "совпадают" if diff < 1e-6 else "различаются"
        
        stats = f"""СРАВНИТЕЛЬНЫЙ АНАЛИЗ АЛГОРИТМОВ
{'='*50}
Характеристики графа:
• Количество вершин: {self.graph.V}
• Количество рёбер: {len(self.graph.edges)}

Алгоритм Краскала:
• Вес MST: {k_w}
• Количество рёбер: {len(self.kruskal_result['edges'])}
• Сложность: O(E log E)

Алгоритм Прима:
• Вес MST: {p_w}
• Количество рёбер: {len(self.prim_result['edges'])}
• Сложность: O(E log V)

Сравнение:
• Разница в весе: {diff}
• Результаты {match}"""
        self.stats_text.insert(tk.END, stats)

    def update_visualization(self):
        if self.graph is None:
            messagebox.showwarning("Предупреждение", "Сначала создайте граф")
            return
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        method = self.viz_method.get()
        if method == "original": self.plot_original_graph()
        elif method == "kruskal": self.plot_mst("Алгоритм Краскала", self.kruskal_result)
        elif method == "prim": self.plot_mst("Алгоритм Прима", self.prim_result)
        else: self.plot_comparison()

    def plot_original_graph(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        G = self.create_nx_graph()
        nx.draw_networkx_nodes(G, self.pos, node_color='lightblue', node_size=500, ax=ax)
        nx.draw_networkx_labels(G, self.pos, labels={i: self.graph.get_vertex_name(i) for i in range(self.graph.V)}, ax=ax)
        nx.draw_networkx_edges(G, self.pos, alpha=0.5, ax=ax)
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, ax=ax)
        ax.set_title("Исходный граф", fontsize=14, fontweight='bold')
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_mst(self, title, result):
        if result is None:
            messagebox.showwarning("Предупреждение", f"Сначала найдите MST алгоритмом {title}")
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        G = self.create_nx_graph()
        mst_edges = set((u, v) for _, u, v in result['edges'])
        
        nx.draw_networkx_nodes(G, self.pos, node_color='lightgreen', node_size=500, ax=ax)
        nx.draw_networkx_labels(G, self.pos, labels={i: self.graph.get_vertex_name(i) for i in range(self.graph.V)}, ax=ax)
        
        non_mst = [(u, v) for u, v in G.edges() if (u, v) not in mst_edges and (v, u) not in mst_edges]
        nx.draw_networkx_edges(G, self.pos, edgelist=non_mst, alpha=0.2, style='dashed', ax=ax)
        nx.draw_networkx_edges(G, self.pos, edgelist=list(mst_edges), width=3, edge_color='red', ax=ax)
        
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, ax=ax)
        ax.set_title(f"{title}\nОбщий вес: {result['weight']}", fontsize=14, fontweight='bold')
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_comparison(self):
        if self.kruskal_result is None or self.prim_result is None:
            messagebox.showwarning("Предупреждение", "Сначала найдите MST обоими алгоритмами")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        G = self.create_nx_graph()
        
        def draw_mst(ax, result, title):
            mst_edges = set((u, v) for _, u, v in result['edges'])
            nx.draw_networkx_nodes(G, self.pos, node_color='lightgreen', node_size=400, ax=ax)
            nx.draw_networkx_labels(G, self.pos, labels={i: self.graph.get_vertex_name(i) for i in range(self.graph.V)}, ax=ax, fontsize=8)
            non_mst = [(u, v) for u, v in G.edges() if (u, v) not in mst_edges and (v, u) not in mst_edges]
            nx.draw_networkx_edges(G, self.pos, edgelist=non_mst, alpha=0.2, style='dashed', ax=ax)
            nx.draw_networkx_edges(G, self.pos, edgelist=list(mst_edges), width=3, edge_color='red', ax=ax)
            ax.set_title(f"{title}\nВес: {result['weight']}", fontsize=12)
            ax.axis('off')

        draw_mst(ax1, self.kruskal_result, "Краскал")
        draw_mst(ax2, self.prim_result, "Прим")
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

def main():
    root = tk.Tk()
    style = ttk.Style()
    style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
    app = MSTGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()