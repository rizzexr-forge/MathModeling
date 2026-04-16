import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transport_methods import TransportSolver
import sys
from io import StringIO

class TransportGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Транспортная задача - Методы северо-западного угла и Фогеля")
        self.root.geometry("1200x700")

        self.supplies = []
        self.demands = []
        self.costs = []
        self.northwest_plan = None
        self.vogel_plan = None

        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        self.input_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.input_frame, text="Ввод данных")
        self.setup_input_tab()

        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Результаты")
        self.setup_results_tab()

        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Визуализация")
        self.setup_viz_tab()

    def setup_input_tab(self):
        dim_frame = ttk.LabelFrame(self.input_frame, text="Размерность задачи", padding=10)
        dim_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(dim_frame, text="Количество поставщиков (m):").grid(row=0, column=0, padx=5, pady=5)
        self.m_var = tk.StringVar(value="3")
        ttk.Entry(dim_frame, textvariable=self.m_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(dim_frame, text="Количество потребителей (n):").grid(row=0, column=2, padx=5, pady=5)
        self.n_var = tk.StringVar(value="3")
        ttk.Entry(dim_frame, textvariable=self.n_var, width=10).grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(dim_frame, text="Создать таблицу", command=self.create_table).grid(row=0, column=4, padx=20, pady=5)

        self.data_frame = ttk.LabelFrame(self.input_frame, text="Ввод данных", padding=10)
        self.data_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.table_frame = ttk.Frame(self.data_frame)
        self.table_frame.pack(fill='both', expand=True)

        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.pack(fill='x', padx=10, pady=10)
        ttk.Button(btn_frame, text="Загрузить пример", command=self.load_example).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Решить задачу", command=self.solve_problem).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Очистить", command=self.clear_input).pack(side='left', padx=5)

    def setup_results_tab(self):
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, width=80, height=30)
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        self.results_text.insert(tk.END, "Результаты решения транспортной задачи\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        self.results_text.insert(tk.END, "Здесь будут отображаться результаты после решения задачи.\n")

    def setup_viz_tab(self):
        control_frame = ttk.Frame(self.viz_frame)
        control_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(control_frame, text="Выберите метод:").pack(side='left', padx=5)
        self.viz_method = tk.StringVar(value="both")
        ttk.Radiobutton(control_frame, text="Северо-западный угол", variable=self.viz_method, value="northwest").pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Метод Фогеля", variable=self.viz_method, value="vogel").pack(side='left', padx=5)
        ttk.Radiobutton(control_frame, text="Сравнение", variable=self.viz_method, value="both").pack(side='left', padx=5)
        ttk.Button(control_frame, text="Обновить графики", command=self.update_visualization).pack(side='left', padx=20)

        self.plot_frame = ttk.Frame(self.viz_frame)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=5)

    def create_table(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        try:
            m = int(self.m_var.get())
            n = int(self.n_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные значения m и n")
            return

        headers_frame = ttk.Frame(self.table_frame)
        headers_frame.pack()
        ttk.Label(headers_frame, text="Поставщики/Потребители", width=20).grid(row=0, column=0, padx=2, pady=2)
        for j in range(n):
            ttk.Label(headers_frame, text=f"B{j+1}", width=10).grid(row=0, column=j+1, padx=2, pady=2)
        ttk.Label(headers_frame, text="Запасы", width=10).grid(row=0, column=n+1, padx=2, pady=2)

        self.cost_entries = []
        self.supply_entries = []
        for i in range(m):
            ttk.Label(headers_frame, text=f"A{i+1}", width=10).grid(row=i+1, column=0, padx=2, pady=2)
            row_entries = []
            for j in range(n):
                entry = ttk.Entry(headers_frame, width=10)
                entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                entry.insert(0, "10")
                row_entries.append(entry)
            self.cost_entries.append(row_entries)

            supply_entry = ttk.Entry(headers_frame, width=10)
            supply_entry.grid(row=i+1, column=n+1, padx=2, pady=2)
            supply_entry.insert(0, "100")
            self.supply_entries.append(supply_entry)

        ttk.Label(headers_frame, text="Потребности", width=10).grid(row=m+1, column=0, padx=2, pady=2)
        self.demand_entries = []
        for j in range(n):
            demand_entry = ttk.Entry(headers_frame, width=10)
            demand_entry.grid(row=m+1, column=j+1, padx=2, pady=2)
            demand_entry.insert(0, "100")
            self.demand_entries.append(demand_entry)

        instructions = ("Инструкция по вводу:\n"
                        "- Введите стоимость перевозок от каждого поставщика к каждому потребителю\n"
                        "- Введите запасы поставщиков в последнем столбце\n"
                        "- Введите потребности потребителей в последней строке")
        ttk.Label(self.table_frame, text=instructions, justify='left').pack(pady=10)

    def load_example(self):
        self.m_var.set("3")
        self.n_var.set("4")
        self.create_table()

        example_costs = [[10, 7, 4, 1], [2, 7, 10, 6], [8, 5, 3, 9]]
        example_supplies = [100, 150, 200]
        example_demands = [150, 120, 80, 100]

        total_supply = sum(example_supplies)
        total_demand = sum(example_demands)

        if total_supply != total_demand:
            diff = total_supply - total_demand
            if diff > 0:
                example_demands.append(diff)
                for i in range(3):
                    example_costs[i].append(0)
            else:
                example_supplies.append(-diff)
                example_costs.append([0] * len(example_demands))

        for i in range(len(example_costs)):
            for j in range(len(example_costs[0])):
                if i < len(self.cost_entries) and j < len(self.cost_entries[i]):
                    self.cost_entries[i][j].delete(0, tk.END)
                    self.cost_entries[i][j].insert(0, str(example_costs[i][j]))

        for i in range(len(example_supplies)):
            if i < len(self.supply_entries):
                self.supply_entries[i].delete(0, tk.END)
                self.supply_entries[i].insert(0, str(example_supplies[i]))

        for j in range(len(example_demands)):
            if j < len(self.demand_entries):
                self.demand_entries[j].delete(0, tk.END)
                self.demand_entries[j].insert(0, str(example_demands[j]))

    def clear_input(self):
        for widget in self.table_frame.winfo_children():
            widget.destroy()

    def solve_problem(self):
        try:
            m = len(self.cost_entries)
            n = len(self.cost_entries[0]) if m > 0 else 0
            if m == 0 or n == 0:
                messagebox.showerror("Ошибка", "Сначала создайте таблицу")
                return

            supplies = [float(self.supply_entries[i].get()) for i in range(m)]
            demands = [float(self.demand_entries[j].get()) for j in range(n)]
            costs = [[float(self.cost_entries[i][j].get()) for j in range(n)] for i in range(m)]

            solver = TransportSolver(supplies, demands, costs)

            old_stdout = sys.stdout
            sys.stdout = StringIO()
            self.northwest_plan, northwest_cost = solver.northwest_corner_method()
            self.vogel_plan, vogel_cost = solver.vogel_approximation_method()
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "РЕЗУЛЬТАТЫ РЕШЕНИЯ ТРАНСПОРТНОЙ ЗАДАЧИ\n")
            self.results_text.insert(tk.END, "=" * 60 + "\n\n")

            self.results_text.insert(tk.END, "МЕТОД СЕВЕРО-ЗАПАДНОГО УГЛА\n")
            self.results_text.insert(tk.END, "-" * 30 + "\n")
            self.results_text.insert(tk.END, "Матрица перевозок:\n")
            for i in range(len(self.northwest_plan)):
                row_str = ""
                for j in range(len(self.northwest_plan[i])):
                    row_str += f"[{self.northwest_plan[i, j]:6.1f}]" if self.northwest_plan[i, j] > 0 else f"{self.northwest_plan[i, j]:6.1f} "
                self.results_text.insert(tk.END, row_str + "\n")
            self.results_text.insert(tk.END, f"Общая стоимость: {northwest_cost:.2f}\n\n")

            self.results_text.insert(tk.END, "МЕТОД ФОГЕЛЯ\n")
            self.results_text.insert(tk.END, "-" * 30 + "\n")
            self.results_text.insert(tk.END, "Матрица перевозок:\n")
            for i in range(len(self.vogel_plan)):
                row_str = ""
                for j in range(len(self.vogel_plan[i])):
                    row_str += f"[{self.vogel_plan[i, j]:6.1f}]" if self.vogel_plan[i, j] > 0 else f"{self.vogel_plan[i, j]:6.1f} "
                self.results_text.insert(tk.END, row_str + "\n")
            self.results_text.insert(tk.END, f"Общая стоимость: {vogel_cost:.2f}\n\n")

            self.results_text.insert(tk.END, "СРАВНЕНИЕ МЕТОДОВ\n")
            self.results_text.insert(tk.END, "-" * 30 + "\n")
            difference = northwest_cost - vogel_cost
            percent = (difference / northwest_cost) * 100 if northwest_cost != 0 else 0
            self.results_text.insert(tk.END, f"Метод северо-западного угла: {northwest_cost:.2f}\n")
            self.results_text.insert(tk.END, f"Метод Фогеля: {vogel_cost:.2f}\n")
            self.results_text.insert(tk.END, f"Разница: {difference:.2f} ({percent:.1f}%)\n")
            if difference > 0:
                self.results_text.insert(tk.END, f"Метод Фогеля эффективнее на {percent:.1f}%\n")
            elif difference < 0:
                self.results_text.insert(tk.END, f"Метод северо-западного угла эффективнее на {abs(percent):.1f}%\n")
            else:
                self.results_text.insert(tk.END, "Методы дают одинаковый результат\n")

            self.notebook.select(1)
            messagebox.showinfo("Успех", "Задача успешно решена!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

    def update_visualization(self):
        if self.northwest_plan is None or self.vogel_plan is None:
            messagebox.showwarning("Предупреждение", "Сначала решите задачу")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        method = self.viz_method.get()
        if method == "northwest":
            self.create_cost_comparison_plot("Метод северо-западного угла", self.northwest_plan)
        elif method == "vogel":
            self.create_cost_comparison_plot("Метод Фогеля", self.vogel_plan)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            self.plot_plan(ax1, self.northwest_plan, "Северо-западный угол")
            self.plot_plan(ax2, self.vogel_plan, "Метод Фогеля")
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_cost_comparison_plot(self, title, plan):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_plan(ax, plan, title)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_plan(self, ax, plan, title):
        m, n = plan.shape
        im = ax.imshow(plan, cmap='YlOrRd', aspect='auto')
        for i in range(m):
            for j in range(n):
                if plan[i, j] > 0:
                    ax.text(j, i, f'{plan[i, j]:.0f}', ha="center", va="center", color="black", fontweight='bold')
                else:
                    ax.text(j, i, '0', ha="center", va="center", color="gray")
        ax.set_xticks(range(n))
        ax.set_yticks(range(m))
        ax.set_xticklabels([f'B{j+1}' for j in range(n)])
        ax.set_yticklabels([f'A{i+1}' for i in range(m)])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Объем перевозок')

def main():
    root = tk.Tk()
    app = TransportGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()