import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import heapq
import sys
import time

# === БЛОК 1: Алгоритм метода ветвей и границ для TSP ===
class TSPBranchBound:
    """Реализация метода ветвей и границ для симметричной задачи коммивояжера."""

    class Node:
        """Узел дерева решений."""
        def __init__(self, reduced_matrix, lower_bound, visited, current_city, path, N):
            self.reduced_matrix = reduced_matrix
            self.lower_bound = lower_bound
            self.visited = visited
            self.current_city = current_city
            self.path = path[:]
            self.N = N

        def __lt__(self, other):
            return self.lower_bound < other.lower_bound

    @staticmethod
    def reduce_matrix(matrix):
        """Приведение матрицы и вычисление константы приведения."""
        n = len(matrix)
        reduced = np.array(matrix, dtype=float)
        reduction_cost = 0.0

        for i in range(n):
            row_min = np.min(reduced[i, :])
            if row_min != np.inf and row_min > 0:
                reduced[i, :] -= row_min
                reduction_cost += row_min
            if np.all(np.isinf(reduced[i, :])):
                return None, np.inf

        for j in range(n):
            col_min = np.min(reduced[:, j])
            if col_min != np.inf and col_min > 0:
                reduced[:, j] -= col_min
                reduction_cost += col_min

        return reduced, reduction_cost

    @staticmethod
    def find_branch_edge(matrix):
        """Поиск нулевого элемента с максимальным штрафом для ветвления."""
        n = len(matrix)
        best_i, best_j = -1, -1
        max_penalty = -1.0

        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 0:
                    min_row = min((matrix[i, k] for k in range(n) if k != j), default=np.inf)
                    min_col = min((matrix[k, j] for k in range(n) if k != i), default=np.inf)

                    penalty = (0 if np.isinf(min_row) else min_row) + \
                              (0 if np.isinf(min_col) else min_col)
                    if penalty > max_penalty:
                        max_penalty = penalty
                        best_i, best_j = i, j
        return best_i, best_j

    @staticmethod
    def create_child_node(parent_node, i, j):
        """Создание дочернего узла при включении ребра (i, j)."""
        N = parent_node.N
        new_matrix = np.copy(parent_node.reduced_matrix)
        new_matrix[i, :] = np.inf
        new_matrix[:, j] = np.inf
        new_matrix[j, i] = np.inf  # блокировка обратного пути (защита от подциклов)

        reduced_matrix, reduction_cost = TSPBranchBound.reduce_matrix(new_matrix)
        if reduced_matrix is None:
            return None

        new_lower_bound = parent_node.lower_bound + parent_node.reduced_matrix[i, j] + reduction_cost
        new_path = parent_node.path + [j]
        return TSPBranchBound.Node(reduced_matrix, new_lower_bound,
                                   parent_node.visited | {j}, j, new_path, N)

    @staticmethod
    def solve(matrix, start_city=0):
        """Основной метод решения TSP."""
        N = len(matrix)
        if N <= 1:
            return [0], 0.0

        initial_matrix, reduction_cost = TSPBranchBound.reduce_matrix(np.array(matrix, dtype=float))
        if initial_matrix is None:
            return None, np.inf

        root_node = TSPBranchBound.Node(initial_matrix, reduction_cost,
                                        {start_city}, start_city, [start_city], N)
        pq = []
        heapq.heappush(pq, (root_node.lower_bound, root_node))

        best_path = None
        best_cost = np.inf

        while pq:
            _, current_node = heapq.heappop(pq)

            if current_node.lower_bound >= best_cost:
                continue

            if len(current_node.path) == N:
                last_city = current_node.current_city
                first_city = current_node.path[0]
                return_cost = current_node.reduced_matrix[last_city, first_city]
                if return_cost < np.inf:
                    total_cost = current_node.lower_bound + return_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = current_node.path + [first_city]
                continue

            i, j = TSPBranchBound.find_branch_edge(current_node.reduced_matrix)
            if i == -1 or j == -1:
                continue

            # Ветка 1: включаем ребро (i, j)
            child_incl = TSPBranchBound.create_child_node(current_node, i, j)
            if child_incl and child_incl.lower_bound < best_cost:
                heapq.heappush(pq, (child_incl.lower_bound, child_incl))

            # Ветка 2: исключаем ребро (i, j)
            excl_matrix = np.copy(current_node.reduced_matrix)
            excl_matrix[i, j] = np.inf
            red_excl, red_cost = TSPBranchBound.reduce_matrix(excl_matrix)
            if red_excl is not None:
                excl_node = TSPBranchBound.Node(red_excl, current_node.lower_bound + red_cost,
                                                current_node.visited, current_node.current_city,
                                                current_node.path, N)
                if excl_node.lower_bound < best_cost:
                    heapq.heappush(pq, (excl_node.lower_bound, excl_node))

        return best_path, best_cost


# === БЛОК 2: Графический интерфейс (Tkinter) ===
class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP: Метод ветвей и границ")
        self.root.geometry("1200x700")

        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="Количество городов (N):").pack(anchor=tk.W, pady=5)
        self.n_entry = ttk.Entry(control_frame, width=10)
        self.n_entry.insert(0, "4")
        self.n_entry.pack(anchor=tk.W)

        ttk.Button(control_frame, text="Создать матрицу", command=self.create_matrix_input).pack(anchor=tk.W, pady=5)

        self.matrix_frame = ttk.Frame(control_frame)
        self.matrix_frame.pack(anchor=tk.W, pady=10)

        ttk.Button(control_frame, text="Решить задачу", command=self.solve_tsp).pack(anchor=tk.W, pady=5)
        ttk.Button(control_frame, text="Загрузить пример (4 города)", command=self.load_example).pack(anchor=tk.W, pady=5)

        result_frame = ttk.LabelFrame(control_frame, text="Результат", padding="5")
        result_frame.pack(anchor=tk.W, fill=tk.X, pady=10)
        self.result_text = tk.Text(result_frame, height=6, width=40, state=tk.DISABLED)
        self.result_text.pack()

        plot_frame = ttk.Frame(root, padding="10")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.matrix_entries = []
        self.N = 0

    def create_matrix_input(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        self.matrix_entries = []
        try:
            self.N = int(self.n_entry.get())
            if self.N < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное N >= 2")
            return

        ttk.Label(self.matrix_frame, text="Матрица расстояний:").grid(row=0, column=0, columnspan=self.N + 1, pady=5)
        for j in range(self.N):
            ttk.Label(self.matrix_frame, text=f"Г{j+1}").grid(row=1, column=j + 1, padx=2)

        for i in range(self.N):
            ttk.Label(self.matrix_frame, text=f"Г{i+1}").grid(row=i + 2, column=0, padx=2)
            row_entries = []
            for j in range(self.N):
                e = ttk.Entry(self.matrix_frame, width=6)
                e.grid(row=i + 2, column=j + 1, padx=1, pady=1)
                if i == j:
                    e.insert(0, "0")
                    e.config(state=tk.DISABLED)
                else:
                    e.insert(0, "0")
                row_entries.append(e)
            self.matrix_entries.append(row_entries)

    def load_example(self):
        self.n_entry.delete(0, tk.END)
        self.n_entry.insert(0, "4")
        self.create_matrix_input()
        example = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.matrix_entries[i][j].delete(0, tk.END)
                    self.matrix_entries[i][j].insert(0, str(example[i][j]))

    def read_matrix(self):
        try:
            matrix = []
            for i in range(self.N):
                row = []
                for j in range(self.N):
                    val = self.matrix_entries[i][j].get()
                    row.append(0 if i == j else float(val) if val else 0)
                matrix.append(row)
            return np.array(matrix)
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте корректность чисел в матрице.")
            return None

    def plot_graph(self, matrix, path):
        self.ax.clear()
        N = len(matrix)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radius = 10
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        for i in range(N):
            for j in range(i + 1, N):
                self.ax.plot([x[i], x[j]], [y[i], y[j]], 'gray', linestyle=':', linewidth=0.5, alpha=0.5)

        self.ax.scatter(x, y, c='red', s=200, zorder=5)
        for i, (xi, yi) in enumerate(zip(x, y)):
            self.ax.annotate(str(i + 1), (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)

        if path:
            path_x = [x[p] for p in path]
            path_y = [y[p] for p in path]
            self.ax.plot(path_x, path_y, 'blue', linewidth=2.5, marker='o', markersize=8, zorder=4)

        self.ax.set_title("Визуализация оптимального маршрута")
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.canvas.draw()

    def solve_tsp(self):
        matrix = self.read_matrix()
        if matrix is None:
            return

        start_time = time.time()
        try:
            path, cost = TSPBranchBound.solve(matrix.tolist(), start_city=0)
        except Exception as e:
            messagebox.showerror("Ошибка алгоритма", str(e))
            return
        exec_time = time.time() - start_time

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        if path is None or cost == np.inf:
            self.result_text.insert(tk.END, "Решение не найдено (нет Гамильтонова цикла).")
        else:
            path_display = [p + 1 for p in path]
            self.result_text.insert(tk.END, f"Маршрут: {' → '.join(map(str, path_display))}\n")
            self.result_text.insert(tk.END, f"Длина: {cost:.2f}\n")
            self.result_text.insert(tk.END, f"Время: {exec_time:.4f} сек")
        self.result_text.config(state=tk.DISABLED)
        self.plot_graph(matrix, path)


if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()