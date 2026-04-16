import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
import sys
from io import StringIO
from matrix_representations import GraphMatrixRepresentation, MatrixOperations

class MatrixGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Матричные способы задания сетей")
        self.root.geometry("1300x800")
        self.graph = None
        self.current_matrix = None
        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        for name, frame in [("Создание сети", "create"), ("Матрицы", "matrix"), 
                            ("Преобразования", "transform"), ("Визуализация", "viz"), ("Анализ сети", "analysis")]:
            f = ttk.Frame(self.notebook)
            self.notebook.add(f, text=name)
            setattr(self, f"{frame}_frame", f)
            getattr(self, f"setup_{frame}_tab")()

    def setup_create_tab(self):
        p = ttk.LabelFrame(self.create_frame, text="Параметры сети", padding=10)
        p.pack(fill='x', padx=10, pady=5)
        
        self.v_var, self.dens_var, self.wmin_var, self.wmax_var = tk.StringVar(value="6"), tk.StringVar(value="0.35"), tk.StringVar(value="1"), tk.StringVar(value="10")
        self.dir_var = tk.BooleanVar(value=False)
        
        ttk.Label(p, text="Вершин:").grid(row=0, column=0, padx=5)
        ttk.Entry(p, textvariable=self.v_var, width=6).grid(row=0, column=1)
        ttk.Label(p, text="Тип:").grid(row=0, column=2, padx=15)
        ttk.Radiobutton(p, text="Неор.", variable=self.dir_var, value=False).grid(row=0, column=3)
        ttk.Radiobutton(p, text="Ориент.", variable=self.dir_var, value=True).grid(row=0, column=4)
        
        ttk.Label(p, text="Плотность:").grid(row=1, column=0, padx=5)
        ttk.Entry(p, textvariable=self.dens_var, width=6).grid(row=1, column=1)
        ttk.Label(p, text="Вес [min, max]:").grid(row=1, column=2, padx=15)
        ttk.Entry(p, textvariable=self.wmin_var, width=5).grid(row=1, column=3)
        ttk.Entry(p, textvariable=self.wmax_var, width=5).grid(row=1, column=4)
        
        btns = ttk.Frame(p)
        btns.grid(row=2, column=0, columnspan=5, pady=10)
        ttk.Button(btns, text="Сгенерировать", command=self.generate_graph).pack(side='left', padx=5)
        ttk.Button(btns, text="Пример", command=self.load_example).pack(side='left', padx=5)
        
        ef = ttk.LabelFrame(self.create_frame, text="Ручное редактирование", padding=10)
        ef.pack(fill='x', padx=10, pady=5)
        self.eu, self.ev = ttk.Combobox(ef, values=[], state='readonly', width=6), ttk.Combobox(ef, values=[], state='readonly', width=6)
        self.ew = tk.StringVar(value="1")
        ttk.Label(ef, text="От:").pack(side='left')
        self.eu.pack(side='left', padx=3)
        ttk.Label(ef, text="До:").pack(side='left')
        self.ev.pack(side='left', padx=3)
        ttk.Label(ef, text="Вес:").pack(side='left')
        ttk.Entry(ef, textvariable=self.ew, width=6).pack(side='left', padx=3)
        ttk.Button(ef, text="Добавить", command=self.add_edge_manual).pack(side='left', padx=10)
        ttk.Button(ef, text="Удалить", command=self.remove_edge_manual).pack(side='left', padx=5)
        
        self.info_text = scrolledtext.ScrolledText(self.create_frame, height=8, font=('Courier', 9))
        self.info_text.pack(fill='x', padx=10, pady=5)

    def setup_matrix_tab(self):
        sf = ttk.LabelFrame(self.matrix_frame, text="Выбор матрицы", padding=10)
        sf.pack(fill='x', padx=10, pady=5)
        self.m_type = tk.StringVar(value="adjacency")
        for val, txt in [("adjacency","Смежности"), ("incidence","Инцидентности"), ("distance","Расстояний"), ("reachability","Достижимости"), ("laplacian","Лапласа")]:
            ttk.Radiobutton(sf, text=txt, variable=self.m_type, value=val).pack(side='left', padx=8)
        ttk.Button(sf, text="Показать", command=self.show_matrix).pack(side='left', padx=20)
        self.m_display = scrolledtext.ScrolledText(self.matrix_frame, height=20, font=('Courier', 10))
        self.m_display.pack(fill='both', expand=True, padx=10, pady=10)

    def setup_transform_tab(self):
        tf = ttk.LabelFrame(self.transform_frame, text="Преобразование матриц", padding=10)
        tf.pack(fill='x', padx=10, pady=5)
        self.from_m = ttk.Combobox(tf, values=["adjacency","incidence"], state='readonly', width=12)
        self.from_m.set("adjacency")
        self.to_m = ttk.Combobox(tf, values=["adjacency","incidence"], state='readonly', width=12)
        self.to_m.set("incidence")
        ttk.Label(tf, text="Из:").pack(side='left')
        self.from_m.pack(side='left', padx=5)
        ttk.Label(tf, text="→ В:").pack(side='left')
        self.to_m.pack(side='left', padx=5)
        ttk.Button(tf, text="Преобразовать", command=self.transform_matrix).pack(side='left', padx=20)
        
        op = ttk.LabelFrame(self.transform_frame, text="Операции", padding=10)
        op.pack(fill='x', padx=10, pady=5)
        self.pwr_k = tk.StringVar(value="2")
        ttk.Label(op, text="Степень k:").pack(side='left')
        ttk.Entry(op, textvariable=self.pwr_k, width=5).pack(side='left', padx=5)
        ttk.Button(op, text="Возвести в степень", command=self.matrix_power).pack(side='left', padx=10)
        ttk.Button(op, text="Пути длины k", command=self.count_paths).pack(side='left', padx=10)
        
        self.t_display = scrolledtext.ScrolledText(self.transform_frame, height=15, font=('Courier', 10))
        self.t_display.pack(fill='both', expand=True, padx=10, pady=10)

    def setup_viz_tab(self):
        cf = ttk.Frame(self.viz_frame)
        cf.pack(fill='x', padx=10, pady=5)
        self.viz_type = tk.StringVar(value="graph")
        ttk.Label(cf, text="Показать:").pack(side='left', padx=5)
        ttk.Radiobutton(cf, text="Граф", variable=self.viz_type, value="graph").pack(side='left')
        ttk.Radiobutton(cf, text="Тепловая карта", variable=self.viz_type, value="heatmap").pack(side='left')
        ttk.Button(cf, text="Обновить", command=self.update_viz).pack(side='left', padx=20)
        self.viz_plot = ttk.Frame(self.viz_frame)
        self.viz_plot.pack(fill='both', expand=True, padx=10, pady=10)

    def setup_analysis_tab(self):
        af = ttk.LabelFrame(self.analysis_frame, text="Характеристики сети", padding=10)
        af.pack(fill='x', padx=10, pady=5)
        ttk.Button(af, text="Вычислить характеристики", command=self.analyze_graph).pack(pady=10)
        self.a_display = scrolledtext.ScrolledText(self.analysis_frame, height=15, font=('Courier', 10))
        self.a_display.pack(fill='both', expand=True, padx=10, pady=10)

    def generate_graph(self):
        try:
            v, d, w1, w2, dir = int(self.v_var.get()), float(self.dens_var.get()), int(self.wmin_var.get()), int(self.wmax_var.get()), self.dir_var.get()
            if v < 2: raise ValueError("Вершин должно быть ≥ 2")
            self.graph = GraphMatrixRepresentation(v, dir)
            self.graph.generate_random_graph(d, w1, w2)
            self._update_combos()
            self._show_info()
            self.m_type.set("adjacency"); self.show_matrix(); self.update_viz()
            messagebox.showinfo("Успех", f"Сеть с {v} вершинами создана")
        except Exception as e: messagebox.showerror("Ошибка", str(e))

    def load_example(self):
        self.v_var.set("6"); self.dir_var.set(False)
        self.graph = GraphMatrixRepresentation(6, False)
        for u,v,w in [(0,1,7),(0,2,9),(0,5,14),(1,2,10),(1,3,15),(2,3,11),(2,5,2),(3,4,6),(4,5,9)]:
            self.graph.add_edge(u,v,w)
        self._update_combos(); self._show_info(); self.m_type.set("adjacency"); self.show_matrix(); self.update_viz()

    def _update_combos(self):
        vals = [f"V{i+1}" for i in range(self.graph.V)]
        for cb in [self.eu, self.ev]: cb['values'] = vals; cb.set(vals[0] if vals else "")

    def add_edge_manual(self):
        if not self.graph: return messagebox.showerror("Ошибка", "Сначала создайте сеть")
        try:
            u = int(self.eu.get().replace('V',''))-1; v = int(self.ev.get().replace('V',''))-1; w = int(self.ew.get())
            self.graph.add_edge(u, v, w); self._update_combos(); self.show_matrix(); self.update_viz()
            messagebox.showinfo("Успех", f"Ребро добавлено")
        except: messagebox.showerror("Ошибка", "Неверный формат")

    def remove_edge_manual(self):
        if not self.graph: return messagebox.showerror("Ошибка", "Сначала создайте сеть")
        try:
            u = int(self.eu.get().replace('V',''))-1; v = int(self.ev.get().replace('V',''))-1
            self.graph.remove_edge(u, v); self.show_matrix(); self.update_viz()
            messagebox.showinfo("Успех", f"Ребро удалено")
        except: messagebox.showerror("Ошибка", "Неверный формат")

    def _show_info(self):
        self.info_text.delete(1.0, tk.END)
        old, sys.stdout = sys.stdout, StringIO()
        self.graph.print_statistics(); out = sys.stdout.getvalue(); sys.stdout = old
        self.info_text.insert(tk.END, out)

    def show_matrix(self):
        if not self.graph: return messagebox.showerror("Ошибка", "Создайте сеть")
        self.m_display.delete(1.0, tk.END)
        mt = self.m_type.get()
        if mt == "adjacency": m = self.graph.get_adjacency_matrix()
        elif mt == "incidence": m = self.graph.get_incidence_matrix()
        elif mt == "distance": m = self.graph.get_distance_matrix()
        elif mt == "reachability": m = self.graph.get_reachability_matrix()
        else: m = self.graph.get_laplacian_matrix()
        self.current_matrix = m
        
        old, sys.stdout = sys.stdout, StringIO()
        self.graph.print_matrix(m, f"Матрица {mt}", format_float=(mt=="distance"))
        out = sys.stdout.getvalue(); sys.stdout = old
        self.m_display.insert(tk.END, out)

    def transform_matrix(self):
        if not self.graph: return messagebox.showerror("Ошибка", "Создайте сеть")
        self.t_display.delete(1.0, tk.END)
        fm, tm = self.from_m.get(), self.to_m.get()
        self.t_display.insert(tk.END, f"Преобразование: {fm} → {tm}\n" + "="*50 + "\n")
        if fm == "adjacency" and tm == "incidence":
            inc = self.graph.adjacency_to_incidence()
            self._show_incidence_in_text(inc)
        elif fm == "incidence" and tm == "adjacency":
            adj = self.graph.incidence_to_adjacency(self.graph.get_incidence_matrix())
            old, sys.stdout = sys.stdout, StringIO()
            self.graph.print_matrix(adj, "Полученная матрица смежности"); out = sys.stdout.getvalue(); sys.stdout = old
            self.t_display.insert(tk.END, out)
        else: self.t_display.insert(tk.END, "Не поддерживается\n")

    def _show_incidence_in_text(self, m):
        V, E = m.shape
        self.t_display.insert(tk.END, "  " + "".join([f"e{j+1:>5}" for j in range(E)]) + "\n")
        for i in range(V):
            row = f"{self.graph.vertices_names[i]}:"
            for j in range(E):
                val = int(m[i][j])
                row += " +1" if val==1 else (" -1" if val==-1 else "  0")
            self.t_display.insert(tk.END, row + "\n")

    def matrix_power(self):
        if self.current_matrix is None: return messagebox.showerror("Ошибка", "Выберите матрицу")
        try:
            k = int(self.pwr_k.get()); res = MatrixOperations.matrix_power(self.current_matrix, k)
            self.t_display.delete(1.0, tk.END)
            self.t_display.insert(tk.END, f"Матрица в степени {k}:\n" + "="*40 + "\n")
            for i in range(res.shape[0]):
                self.t_display.insert(tk.END, "  " + "".join([f"{int(res[i][j]):6d}" for j in range(res.shape[1])]) + "\n")
        except: messagebox.showerror("Ошибка", "Неверная степень")

    def count_paths(self):
        if self.current_matrix is None: return messagebox.showerror("Ошибка", "Выберите матрицу смежности")
        try:
            k = int(self.pwr_k.get()); paths = MatrixOperations.count_paths(self.current_matrix, k)
            self.t_display.delete(1.0, tk.END)
            self.t_display.insert(tk.END, f"Количество путей длины {k}:\n" + "="*40 + "\n")
            for i in range(paths.shape[0]):
                row = f"{self.graph.vertices_names[i]}:"
                for j in range(paths.shape[1]): row += f"{int(paths[i][j]):6d}"
                self.t_display.insert(tk.END, row + "\n")
        except: messagebox.showerror("Ошибка", "Неверная степень")

    def update_viz(self):
        if not self.graph: return messagebox.showwarning("Предупреждение", "Создайте сеть")
        for w in self.viz_plot.winfo_children(): w.destroy()
        if self.viz_type.get() == "graph": self.plot_graph()
        else: self.plot_heatmap()

    def plot_graph(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        G = nx.DiGraph() if self.graph.directed else nx.Graph()
        for i in range(self.graph.V): G.add_node(i, label=self.graph.vertices_names[i])
        seen = set()
        for u,v,w in self.graph.edges:
            e = (u,v) if self.graph.directed else tuple(sorted([u,v]))
            if e not in seen: seen.add(e); G.add_edge(u,v,weight=w)
        pos = {i: self.graph.coordinates[i] for i in range(self.graph.V)}
        nx.draw(G, pos, with_labels=True, labels={i: self.graph.vertices_names[i] for i in range(self.graph.V)}, 
                node_color='lightblue', node_size=500, font_weight='bold', arrows=self.graph.directed, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):d['weight'] for u,v,d in G.edges(data=True)}, font_size=8, ax=ax)
        ax.set_title("Визуализация сети", fontsize=14, fontweight='bold'); ax.axis('off')
        FigureCanvasTkAgg(fig, self.viz_plot).get_tk_widget().pack(fill='both', expand=True)

    def plot_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        adj = self.graph.get_adjacency_matrix()
        im = ax.imshow(adj, cmap='YlOrRd', aspect='auto')
        for i in range(self.graph.V):
            for j in range(self.graph.V):
                val = adj[i, j]
                if val > 0: ax.text(j, i, f'{int(val)}', ha="center", va="center", color="black")
                elif val == 0 and i != j: ax.text(j, i, '0', ha="center", va="center", color="gray")
        ax.set_xticks(range(self.graph.V)); ax.set_yticks(range(self.graph.V))
        ax.set_xticklabels(self.graph.vertices_names); ax.set_yticklabels(self.graph.vertices_names)
        ax.set_title("Матрица смежности (тепловая карта)", fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Вес ребра')
        FigureCanvasTkAgg(fig, self.viz_plot).get_tk_widget().pack(fill='both', expand=True)

    def analyze_graph(self):
        if not self.graph: return messagebox.showerror("Ошибка", "Создайте сеть")
        self.a_display.delete(1.0, tk.END)
        self.a_display.insert(tk.END, "АНАЛИЗ СЕТИ\n" + "="*60 + "\n\n")
        self.a_display.insert(tk.END, f"Вершин: {self.graph.V}\nРёбер: {len(self.graph.edges)//(2 if not self.graph.directed else 1)}\n")
        max_e = self.graph.V*(self.graph.V-1)//(1 if not self.graph.directed else 2)
        dens = (len(self.graph.edges)//(2 if not self.graph.directed else 1))/max_e if max_e>0 else 0
        self.a_display.insert(tk.END, f"Плотность: {dens:.3f} ({dens*100:.1f}%)\n\n")
        if not self.graph.directed:
            self.a_display.insert(tk.END, f"Радиус: {MatrixOperations.find_radius(self.graph)}\nДиаметр: {MatrixOperations.find_diameter(self.graph)}\n")
            centers, _ = MatrixOperations.find_centers(self.graph)
            self.a_display.insert(tk.END, f"Центры: {', '.join(centers)}\n\nСтепени:\n")
            for i, d in enumerate(np.sum(self.graph.adj_matrix>0, axis=1)):
                self.a_display.insert(tk.END, f"  {self.graph.vertices_names[i]}: {int(d)}\n")
            conn = MatrixOperations.is_connected(self.graph.adj_matrix)
            self.a_display.insert(tk.END, f"\nСвязность: {'Связный' if conn else 'Несвязный'}\n")

def main():
    root = tk.Tk()
    MatrixGUI(root)
    root.mainloop()