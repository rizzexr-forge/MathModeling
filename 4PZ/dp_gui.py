import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import random
import sys
from io import StringIO
from dynamic_programming import ResourceAllocationDP, ProductionPlanningDP, KnapsackDP

class DynamicProgrammingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод динамического программирования - Производственные задачи")
        self.root.geometry("1200x800")
        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        self.resource_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.resource_frame, text="Распределение ресурсов")
        self.setup_resource_tab()

        self.planning_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.planning_frame, text="Планирование производства")
        self.setup_planning_tab()

        self.knapsack_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.knapsack_frame, text="Инвестиционные проекты")
        self.setup_knapsack_tab()

    def setup_resource_tab(self):
        params = ttk.LabelFrame(self.resource_frame, text="Параметры задачи", padding=10)
        params.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(params, text="Участков:").grid(row=0, column=0, padx=5)
        self.num_units_var = tk.StringVar(value="3")
        ttk.Entry(params, textvariable=self.num_units_var, width=8).grid(row=0, column=1)
        
        ttk.Label(params, text="Ресурсов:").grid(row=0, column=2, padx=20)
        self.total_res_var = tk.StringVar(value="5")
        ttk.Entry(params, textvariable=self.total_res_var, width=8).grid(row=0, column=3)
        
        ttk.Button(params, text="Создать таблицу функций", command=self.create_profit_table).grid(row=1, column=0, columnspan=4, pady=10)

        func_frame = ttk.Frame(self.resource_frame)
        func_frame.pack(fill='x', padx=10)
        ttk.Label(func_frame, text="Тип функции:").pack(side='left')
        self.profit_type = tk.StringVar(value="quadratic")
        ttk.Radiobutton(func_frame, text="Линейная", variable=self.profit_type, value="linear").pack(side='left')
        ttk.Radiobutton(func_frame, text="Квадратичная", variable=self.profit_type, value="quadratic").pack(side='left')
        ttk.Label(func_frame, text="Коэфф. a:").pack(side='left', padx=(20,5))
        self.profit_coeff = tk.StringVar(value="10")
        ttk.Entry(func_frame, textvariable=self.profit_coeff, width=6).pack(side='left')
        ttk.Button(func_frame, text="Применить ко всем", command=self.apply_function_to_all).pack(side='left', padx=10)

        self.profit_container = ttk.Frame(self.resource_frame)
        self.profit_container.pack(fill='both', expand=True, padx=10, pady=5)

        btns = ttk.Frame(self.resource_frame)
        btns.pack(fill='x', padx=10, pady=5)
        ttk.Button(btns, text="Загрузить пример", command=self.load_resource_example).pack(side='left', padx=5)
        ttk.Button(btns, text="Решить задачу", command=self.solve_resource_problem).pack(side='left', padx=5)

        self.res_results = scrolledtext.ScrolledText(self.resource_frame, height=8)
        self.res_results.pack(fill='both', expand=True, padx=10, pady=5)
        self.res_plot = ttk.Frame(self.resource_frame)
        self.res_plot.pack(fill='both', expand=True, padx=10, pady=5)

    def create_profit_table(self):
        for w in self.profit_container.winfo_children(): w.destroy()
        try:
            n = int(self.num_units_var.get())
            S = int(self.total_res_var.get())
        except:
            return messagebox.showerror("Ошибка", "Введите корректные числа")

        self.profit_entries = []
        scroll = ttk.Frame(self.profit_container)
        scroll.pack(fill='both', expand=True)
        canvas = tk.Canvas(scroll)
        sb = ttk.Scrollbar(scroll, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        ttk.Label(frame, text="Участок \\ x", width=12).grid(row=0, column=0)
        for x in range(S + 1):
            ttk.Label(frame, text=f"x={x}", width=6).grid(row=0, column=x+1)
        for i in range(n):
            ttk.Label(frame, text=f"Участок {i+1}", width=12).grid(row=i+1, column=0)
            row = []
            for x in range(S + 1):
                e = ttk.Entry(frame, width=8)
                e.grid(row=i+1, column=x+1)
                e.insert(0, str(int(10*(x+1) if self.profit_type.get()=="linear" else 5*x*x)))
                row.append(e)
            self.profit_entries.append(row)

    def apply_function_to_all(self):
        try:
            a = float(self.profit_coeff.get())
            t = self.profit_type.get()
            S = int(self.total_res_var.get())
            for row in self.profit_entries:
                for x, e in enumerate(row):
                    val = a*(x) if t=="linear" else a*x*x
                    e.delete(0, tk.END)
                    e.insert(0, str(int(val)))
        except:
            messagebox.showerror("Ошибка", "Проверьте коэффициенты")

    def load_resource_example(self):
        self.num_units_var.set("3"); self.total_res_var.set("5")
        self.create_profit_table()
        data = [[0,10,18,24,28,30], [0,12,20,26,30,32], [0,8,15,21,26,30]]
        for i in range(3):
            for x in range(6):
                self.profit_entries[i][x].delete(0, tk.END)
                self.profit_entries[i][x].insert(0, str(data[i][x]))

    def solve_resource_problem(self):
        try:
            n = int(self.num_units_var.get())
            S = int(self.total_res_var.get())
            profits = [[float(e.get()) for e in row] for row in self.profit_entries]
            solver = ResourceAllocationDP(n, S, profits)
            
            old, sys.stdout = sys.stdout, StringIO()
            alloc, tp = solver.solve()
            solver.print_solution(alloc, tp)
            out = sys.stdout.getvalue(); sys.stdout = old
            
            self.res_results.delete(1.0, tk.END)
            self.res_results.insert(tk.END, out)
            
            for w in self.res_plot.winfo_children(): w.destroy()
            solver.visualize_solution(alloc, tp, self.res_plot)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def setup_planning_tab(self):
        p = ttk.LabelFrame(self.planning_frame, text="Параметры", padding=10)
        p.pack(fill='x', padx=10)
        self.per_var = tk.StringVar(value="6"); self.cap_var = tk.StringVar(value="100")
        self.stor_var = tk.StringVar(value="2"); self.prod_var = tk.StringVar(value="10")
        self.inv_var = tk.StringVar(value="0")
        fields = [("Периодов:", self.per_var, 0), ("Мощность:", self.cap_var, 1), 
                  ("Хранение:", self.stor_var, 2), ("Произв.:", self.prod_var, 3), ("Нач. запас:", self.inv_var, 4)]
        for lbl, var, col in fields:
            ttk.Label(p, text=lbl).grid(row=0, column=col*2, padx=5)
            ttk.Entry(p, textvariable=var, width=6).grid(row=0, column=col*2+1)

        self.demand_frame = ttk.LabelFrame(self.planning_frame, text="Спрос по периодам", padding=10)
        self.demand_frame.pack(fill='x', padx=10, pady=5)
        self.demand_entries = []
        ttk.Button(self.demand_frame, text="Сгенерировать спрос", command=self.gen_demand).pack(pady=5)

        btns = ttk.Frame(self.planning_frame)
        btns.pack(fill='x', padx=10, pady=5)
        ttk.Button(btns, text="Загрузить пример", command=self.load_plan_example).pack(side='left', padx=5)
        ttk.Button(btns, text="Решить задачу", command=self.solve_plan_problem).pack(side='left', padx=5)

        self.plan_res = scrolledtext.ScrolledText(self.planning_frame, height=8)
        self.plan_res.pack(fill='both', expand=True, padx=10, pady=5)
        self.plan_plot = ttk.Frame(self.planning_frame)
        self.plan_plot.pack(fill='both', expand=True, padx=10, pady=5)

    def gen_demand(self):
        for w in self.demand_frame.winfo_children()[1:]: w.destroy()
        self.demand_entries = []
        try: T = int(self.per_var.get())
        except: return
        for t in range(T):
            ttk.Label(self.demand_frame, text=f"Период {t+1}:").grid(row=0, column=t*2, padx=5)
            e = ttk.Entry(self.demand_frame, width=6)
            e.grid(row=0, column=t*2+1)
            e.insert(0, str(random.randint(30, 100)))
            self.demand_entries.append(e)

    def load_plan_example(self):
        self.per_var.set("6"); self.gen_demand()
        ex = [50,60,70,80,60,50]
        for i, e in enumerate(self.demand_entries[:6]):
            e.delete(0, tk.END); e.insert(0, str(ex[i]))

    def solve_plan_problem(self):
        try:
            T = int(self.per_var.get()); cap = int(self.cap_var.get())
            stor = float(self.stor_var.get()); prod = float(self.prod_var.get())
            inv0 = int(self.inv_var.get())
            dem = [int(e.get()) for e in self.demand_entries]
            solver = ProductionPlanningDP(T, dem, cap, stor, prod, inv0)
            
            old, sys.stdout = sys.stdout, StringIO()
            plan, inv, cost = solver.solve()
            solver.print_solution(plan, inv, cost)
            out = sys.stdout.getvalue(); sys.stdout = old
            
            self.plan_res.delete(1.0, tk.END)
            self.plan_res.insert(tk.END, out)
            for w in self.plan_plot.winfo_children(): w.destroy()
            solver.visualize_solution(plan, inv, cost, self.plan_plot)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def setup_knapsack_tab(self):
        p = ttk.LabelFrame(self.knapsack_frame, text="Параметры", padding=10)
        p.pack(fill='x', padx=10)
        self.bud_var = tk.StringVar(value="50")
        self.proj_var = tk.StringVar(value="8")
        ttk.Label(p, text="Бюджет:").grid(row=0, column=0)
        ttk.Entry(p, textvariable=self.bud_var, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(p, text="Проектов:").grid(row=0, column=2)
        ttk.Entry(p, textvariable=self.proj_var, width=6).grid(row=0, column=3, padx=5)
        ttk.Button(p, text="Создать таблицу", command=self.create_proj_table).grid(row=1, column=0, columnspan=4, pady=5)

        self.proj_frame = ttk.LabelFrame(self.knapsack_frame, text="Инвестиционные проекты", padding=10)
        self.proj_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.proj_cont = ttk.Frame(self.proj_frame)
        self.proj_cont.pack(fill='both', expand=True)
        self.w_entries = []; self.v_entries = []

        btns = ttk.Frame(self.knapsack_frame)
        btns.pack(fill='x', padx=10, pady=5)
        ttk.Button(btns, text="Пример", command=self.load_knapsack_example).pack(side='left', padx=5)
        ttk.Button(btns, text="Рандом", command=self.gen_projects).pack(side='left', padx=5)
        ttk.Button(btns, text="Решить задачу", command=self.solve_knapsack).pack(side='left', padx=5)

        self.knapsack_res = scrolledtext.ScrolledText(self.knapsack_frame, height=8)
        self.knapsack_res.pack(fill='both', expand=True, padx=10, pady=5)
        self.knapsack_plot = ttk.Frame(self.knapsack_frame)
        self.knapsack_plot.pack(fill='both', expand=True, padx=10, pady=5)

    def create_proj_table(self):
        for w in self.proj_cont.winfo_children(): w.destroy()
        self.w_entries = []; self.v_entries = []
        try: n = int(self.proj_var.get())
        except: return
        ttk.Label(self.proj_cont, text="Проект", width=10).grid(row=0, column=0)
        ttk.Label(self.proj_cont, text="Затраты", width=8).grid(row=0, column=1)
        ttk.Label(self.proj_cont, text="Прибыль", width=8).grid(row=0, column=2)
        for i in range(n):
            ttk.Label(self.proj_cont, text=f"Проект {i+1}").grid(row=i+1, column=0)
            we = ttk.Entry(self.proj_cont, width=8); we.grid(row=i+1, column=1)
            we.insert(0, str(5 + i*3)); self.w_entries.append(we)
            ve = ttk.Entry(self.proj_cont, width=8); ve.grid(row=i+1, column=2)
            ve.insert(0, str(10 + i*8)); self.v_entries.append(ve)

    def gen_projects(self):
        try: n = int(self.proj_var.get())
        except: return
        for we, ve in zip(self.w_entries, self.v_entries):
            w = random.randint(5, 20); v = random.randint(w*2, w*5)
            we.delete(0, tk.END); we.insert(0, str(w))
            ve.delete(0, tk.END); ve.insert(0, str(v))

    def load_knapsack_example(self):
        self.bud_var.set("50"); self.proj_var.set("8"); self.create_proj_table()
        ws = [10,15,20,12,18,8,25,14]; vs = [30,40,50,35,45,20,60,38]
        for i in range(8):
            self.w_entries[i].delete(0, tk.END); self.w_entries[i].insert(0, str(ws[i]))
            self.v_entries[i].delete(0, tk.END); self.v_entries[i].insert(0, str(vs[i]))

    def solve_knapsack(self):
        try:
            budget = int(self.bud_var.get())
            weights = [int(e.get()) for e in self.w_entries]
            values = [int(e.get()) for e in self.v_entries]
            solver = KnapsackDP(budget, weights, values)
            
            old, sys.stdout = sys.stdout, StringIO()
            sel, tv = solver.solve()
            solver.print_solution(sel, tv)
            out = sys.stdout.getvalue(); sys.stdout = old
            
            self.knapsack_res.delete(1.0, tk.END)
            self.knapsack_res.insert(tk.END, out)
            for w in self.knapsack_plot.winfo_children(): w.destroy()
            solver.visualize_solution(sel, tv, self.knapsack_plot)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

def main():
    root = tk.Tk()
    DynamicProgrammingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()