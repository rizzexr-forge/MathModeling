import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from gomori_core import GomoriSolver

class GomoriApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Решение ЦЛП методом Гомори")
        self.root.geometry("900x700")

        self.n_var = tk.IntVar(value=2)
        self.m_var = tk.IntVar(value=2)
        self.opt_type = tk.StringVar(value="max")

        dim_frame = ttk.LabelFrame(root, text="Размерность задачи", padding=10)
        dim_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(dim_frame, text="Количество переменных(n):").grid(row=0, column=0, padx=5)
        ttk.Spinbox(dim_frame, from_=1, to=10, textvariable=self.n_var, width=5, command=self.rebuild_input_grid).grid(row=0, column=1)
        ttk.Label(dim_frame, text="Количество ограничений(m):").grid(row=0, column=2, padx=5)
        ttk.Spinbox(dim_frame, from_=1, to=10, textvariable=self.m_var, width=5, command=self.rebuild_input_grid).grid(row=0, column=3)
        ttk.Label(dim_frame, text="Тип оптимизации:").grid(row=0, column=4, padx=5)
        ttk.Radiobutton(dim_frame, text="Max", variable=self.opt_type, value="max").grid(row=0, column=5)
        ttk.Radiobutton(dim_frame, text="Min", variable=self.opt_type, value="min").grid(row=0, column=6)

        input_frame = ttk.LabelFrame(root, text="Ввод коэффициентов", padding=10)
        input_frame.pack(fill="both", expand=True, padx=10, pady=5)
        canvas = tk.Canvas(input_frame)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.A_entries, self.b_entries, self.c_entries = [], [], []
        self.create_input_grid()

        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="Решить", command=self.solve_problem).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Очистить", command=self.clear_all).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Заполнить пример", command=self.fill_example).pack(side="left", padx=5)

        result_frame = ttk.LabelFrame(root, text="Результаты", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.result_text = scrolledtext.ScrolledText(result_frame, height=12, wrap=tk.WORD)
        self.result_text.pack(fill="both", expand=True)

        summary_frame = ttk.Frame(root)
        summary_frame.pack(fill="x", padx=10, pady=5)
        self.x_label = ttk.Label(summary_frame, text="x*=")
        self.x_label.pack(side="left", padx=10)
        self.z_label = ttk.Label(summary_frame, text="Z*=")
        self.z_label.pack(side="left", padx=10)

    def create_input_grid(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        n, m = self.n_var.get(), self.m_var.get()
        self.A_entries, self.b_entries, self.c_entries = [], [], []

        ttk.Label(self.scrollable_frame, text="A (коэффициенты ограничений):", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=n, sticky="w", pady=5)
        for i in range(m):
            row_entries = []
            for j in range(n):
                e = ttk.Entry(self.scrollable_frame, width=8)
                e.grid(row=i+1, column=j, padx=2, pady=2)
                e.insert(0, "0")
                row_entries.append(e)
            self.A_entries.append(row_entries)

        ttk.Label(self.scrollable_frame, text="b (правые части):", font=('Arial', 10, 'bold')).grid(row=0, column=n+1, columnspan=1, sticky="w", pady=5, padx=(20,0))
        for i in range(m):
            e = ttk.Entry(self.scrollable_frame, width=8)
            e.grid(row=i+1, column=n+1, padx=(20,2), pady=2)
            e.insert(0, "0")
            self.b_entries.append(e)

        ttk.Label(self.scrollable_frame, text="C (целевая функция):", font=('Arial', 10, 'bold')).grid(row=m+2, column=0, columnspan=n, sticky="w", pady=(15,5))
        for j in range(n):
            e = ttk.Entry(self.scrollable_frame, width=8)
            e.grid(row=m+3, column=j, padx=2, pady=2)
            e.insert(0, "0")
            self.c_entries.append(e)

    def rebuild_input_grid(self):
        self.create_input_grid()

    def get_data_from_entries(self):
        try:
            n, m = self.n_var.get(), self.m_var.get()
            A = [[float(self.A_entries[i][j].get()) for j in range(n)] for i in range(m)]
            b = [float(self.b_entries[i].get()) for i in range(m)]
            c = [float(self.c_entries[j].get()) for j in range(n)]
            return A, b, c
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Проверьте правильность введенных чисел.\n{str(e)}")
            return None, None, None

    def solve_problem(self):
        A, b, c = self.get_data_from_entries()
        if A is None: return
        maximize = (self.opt_type.get() == "max")
        self.result_text.delete(1.0, tk.END)
        try:
            solver = GomoriSolver()
            result = solver.solve(c, A, b, maximize)
            self.result_text.insert(tk.END, result.get('log', 'Лог не доступен'))
            if result['success']:
                x_str = ", ".join([f"x{i+1}={val}" for i, val in enumerate(result['x'])])
                self.x_label.config(text=f"x*=({x_str})")
                self.z_label.config(text=f"Z*={result['z']:.2f}")
                self.result_text.insert(tk.END, f"\n\nИТОГ: Целочисленное решение найдено за {result['iterations']} итераций.\n")
            else:
                self.x_label.config(text="x*= Не найдено")
                self.z_label.config(text="Z*=-")
                self.result_text.insert(tk.END, f"\n\nОШИБКА: {result['message']}\n")
        except Exception as e:
            messagebox.showerror("Ошибка выполнения", str(e))

    def clear_all(self):
        for row in self.A_entries:
            for e in row: e.delete(0, tk.END); e.insert(0, "0")
        for e in self.b_entries: e.delete(0, tk.END); e.insert(0, "0")
        for e in self.c_entries: e.delete(0, tk.END); e.insert(0, "0")
        self.result_text.delete(1.0, tk.END)
        self.x_label.config(text="x*=")
        self.z_label.config(text="Z*=")

    def fill_example(self):
        self.n_var.set(2); self.m_var.set(2)
        self.rebuild_input_grid()
        self.A_entries[0][0].delete(0, tk.END); self.A_entries[0][0].insert(0, "2")
        self.A_entries[0][1].delete(0, tk.END); self.A_entries[0][1].insert(0, "4")
        self.A_entries[1][0].delete(0, tk.END); self.A_entries[1][0].insert(0, "3")
        self.A_entries[1][1].delete(0, tk.END); self.A_entries[1][1].insert(0, "2")
        self.b_entries[0].delete(0, tk.END); self.b_entries[0].insert(0, "13")
        self.b_entries[1].delete(0, tk.END); self.b_entries[1].insert(0, "12")
        self.c_entries[0].delete(0, tk.END); self.c_entries[0].insert(0, "2")
        self.c_entries[1].delete(0, tk.END); self.c_entries[1].insert(0, "3")
        self.opt_type.set("max")

if __name__ == "__main__":
    root = tk.Tk()
    app = GomoriApp(root)
    root.mainloop()