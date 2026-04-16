import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

class ResourceAllocationDP:
    """Решение задачи распределения ресурсов методом ДП"""
    def __init__(self, num_units, total_resources, profit_functions):
        self.n = num_units
        self.S = total_resources
        self.profit_functions = profit_functions
        self.dp_table = None
        self.allocation_table = None

    def solve(self):
        self.dp_table = np.zeros((self.n + 1, self.S + 1))
        self.allocation_table = np.zeros((self.n + 1, self.S + 1), dtype=int)

        print("\n" + "="*70)
        print("ЗАДАЧА РАСПРЕДЕЛЕНИЯ РЕСУРСОВ МЕЖДУ ПРОИЗВОДСТВЕННЫМИ УЧАСТКАМИ")
        print("="*70)
        print(f"Количество участков: {self.n}")
        print(f"Общий объём ресурсов: {self.S}")

        for k in range(1, self.n + 1):
            for s in range(self.S + 1):
                max_profit = -float('inf')
                best_x = 0
                for x in range(s + 1):
                    profit = self._get_profit(k - 1, x) + self.dp_table[k - 1][s - x]
                    if profit > max_profit:
                        max_profit = profit
                        best_x = x
                self.dp_table[k][s] = max_profit
                self.allocation_table[k][s] = best_x

        allocation = [0] * self.n
        remaining = self.S
        for k in range(self.n, 0, -1):
            allocation[k - 1] = self.allocation_table[k][remaining]
            remaining -= allocation[k - 1]

        return allocation, self.dp_table[self.n][self.S]

    def _get_profit(self, unit_idx, x):
        profit_func = self.profit_functions[unit_idx]
        if callable(profit_func):
            return profit_func(x)
        elif isinstance(profit_func, (list, np.ndarray)):
            return profit_func[x] if x < len(profit_func) else 0
        return 0

    def print_solution(self, allocation, total_profit):
        print("\n" + "-"*50)
        print("РЕЗУЛЬТАТЫ РАСПРЕДЕЛЕНИЯ РЕСУРСОВ")
        print("-"*50)
        print("\nОптимальное распределение ресурсов:")
        for i, alloc in enumerate(allocation):
            profit = self._get_profit(i, alloc)
            print(f"  Участок {i+1}: {alloc} ед. ресурсов → прибыль: {profit}")
        print(f"\nСуммарная максимальная прибыль: {total_profit}")
        print(f"Эффективность: {total_profit/self.S:.2f} прибыли на ед. ресурса")

    def visualize_solution(self, allocation, total_profit, master=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        units = [f"Участок {i+1}" for i in range(self.n)]
        profits = [self._get_profit(i, allocation[i]) for i in range(self.n)]

        bars1 = ax1.bar(units, allocation, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Производственные участки')
        ax1.set_ylabel('Выделено ресурсов')
        ax1.set_title('Распределение ресурсов между участками')
        ax1.grid(True, alpha=0.3)
        for bar, val in zip(bars1, allocation):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(int(val)), ha='center', va='bottom')

        bars2 = ax2.bar(units, profits, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Производственные участки')
        ax2.set_ylabel('Прибыль')
        ax2.set_title('Прибыль по участкам')
        ax2.grid(True, alpha=0.3)
        for bar, val in zip(bars2, profits):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}", ha='center', va='bottom')

        fig.suptitle(f'Решение задачи распределения ресурсов\nСуммарная прибыль: {total_profit:.2f}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if master:
            canvas = FigureCanvasTkAgg(fig, master=master)
            canvas.draw()
            return canvas.get_tk_widget()
        else:
            plt.show()

class ProductionPlanningDP:
    """Задача производственного планирования (управление запасами)"""
    def __init__(self, periods, demands, production_capacity, storage_cost, production_cost, initial_inventory=0):
        self.T = periods
        self.d = demands
        self.P_max = production_capacity
        self.h = storage_cost
        self.c = production_cost
        self.I0 = initial_inventory
        self.dp_table = None

    def solve(self):
        max_inventory = sum(self.d) + self.I0
        self.dp_table = np.full((self.T + 1, max_inventory + 1), float('inf'))
        self.decision_table = np.zeros((self.T + 1, max_inventory + 1), dtype=int)
        self.dp_table[0][self.I0] = 0

        print("\n" + "="*70)
        print("ЗАДАЧА ПРОИЗВОДСТВЕННОГО ПЛАНИРОВАНИЯ (УПРАВЛЕНИЕ ЗАПАСАМИ)")
        print("="*70)
        print(f"Периодов: {self.T} | Спрос: {self.d} | Мощность: {self.P_max}")

        for t in range(self.T):
            for inv in range(max_inventory + 1):
                if self.dp_table[t][inv] == float('inf'):
                    continue
                min_prod = max(0, self.d[t] - inv)
                max_prod = min(self.P_max, max_inventory - inv + self.d[t])
                
                for prod in range(min_prod, max_prod + 1):
                    new_inv = inv + prod - self.d[t]
                    cost = prod * self.c + new_inv * self.h
                    total_cost = self.dp_table[t][inv] + cost
                    if total_cost < self.dp_table[t + 1][new_inv]:
                        self.dp_table[t + 1][new_inv] = total_cost
                        self.decision_table[t + 1][new_inv] = prod

        min_cost = float('inf')
        best_final_inv = 0
        for inv in range(max_inventory + 1):
            if self.dp_table[self.T][inv] < min_cost:
                min_cost = self.dp_table[self.T][inv]
                best_final_inv = inv

        production_plan = [0] * self.T
        inventory = [0] * (self.T + 1)
        inventory[self.T] = best_final_inv
        current_inv = best_final_inv

        for t in range(self.T - 1, -1, -1):
            production_plan[t] = self.decision_table[t + 1][current_inv]
            current_inv = current_inv + self.d[t] - production_plan[t]
            inventory[t] = current_inv

        return production_plan, inventory, min_cost

    def print_solution(self, production_plan, inventory, total_cost):
        print("\n" + "-"*50)
        print("РЕЗУЛЬТАТЫ ПРОИЗВОДСТВЕННОГО ПЛАНИРОВАНИЯ")
        print("-"*50)
        print("\nОптимальный план производства:")
        print("Период | Производство | Спрос | Запас на конец")
        print("-"*45)
        for t in range(self.T):
            print(f"   {t+1:2}    |      {production_plan[t]:3}      |  {self.d[t]:3}  |     {inventory[t+1]:3}")
        print(f"\nОбщие затраты: {total_cost:.2f}")

    def visualize_solution(self, production_plan, inventory, total_cost, master=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        periods = list(range(1, self.T + 1))
        x = np.arange(len(periods))
        width = 0.35

        ax1.bar(x - width/2, production_plan, width, label='Производство', color='skyblue', alpha=0.7)
        ax1.bar(x + width/2, self.d, width, label='Спрос', color='lightcoral', alpha=0.7)
        ax1.set_xlabel('Период')
        ax1.set_ylabel('Объём')
        ax1.set_title('Производство и спрос по периодам')
        ax1.set_xticks(x)
        ax1.set_xticklabels(periods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        inventory_levels = inventory[1:]
        ax2.plot(periods, inventory_levels, marker='o', linewidth=2, markersize=8, color='green')
        ax2.fill_between(periods, 0, inventory_levels, alpha=0.3, color='green')
        ax2.set_xlabel('Период')
        ax2.set_ylabel('Уровень запасов')
        ax2.set_title('Динамика изменения запасов')
        ax2.grid(True, alpha=0.3)
        for i, val in enumerate(inventory_levels):
            ax2.annotate(f'{val}', (periods[i], val), textcoords="offset points", xytext=(0,10), ha='center')

        fig.suptitle(f'Решение задачи производственного планирования\nОбщие затраты: {total_cost:.2f}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if master:
            canvas = FigureCanvasTkAgg(fig, master=master)
            canvas.draw()
            return canvas.get_tk_widget()
        else:
            plt.show()

class KnapsackDP:
    """Задача о рюкзаке (выбор инвестиционных проектов)"""
    def __init__(self, capacity, weights, values):
        self.W = capacity
        self.w = weights
        self.v = values
        self.n = len(weights)
        self.dp_table = None
        self.selected = None

    def solve(self):
        self.dp_table = np.zeros((self.n + 1, self.W + 1))
        self.selected = np.zeros((self.n + 1, self.W + 1), dtype=bool)

        print("\n" + "="*70)
        print("ЗАДАЧА О РЮКЗАКЕ (ВЫБОР ИНВЕСТИЦИОННЫХ ПРОЕКТОВ)")
        print("="*70)
        print(f"Бюджет: {self.W} | Проектов: {self.n}")

        for i in range(1, self.n + 1):
            for w in range(self.W + 1):
                if self.w[i-1] <= w:
                    without = self.dp_table[i-1][w]
                    with_item = self.dp_table[i-1][w - self.w[i-1]] + self.v[i-1]
                    if with_item > without:
                        self.dp_table[i][w] = with_item
                        self.selected[i][w] = True
                    else:
                        self.dp_table[i][w] = without
                else:
                    self.dp_table[i][w] = self.dp_table[i-1][w]

        selected_projects = []
        w = self.W
        for i in range(self.n, 0, -1):
            if self.selected[i][w]:
                selected_projects.append(i - 1)
                w -= self.w[i-1]
        selected_projects.reverse()

        total_value = self.dp_table[self.n][self.W]
        return selected_projects, total_value

    def print_solution(self, selected_projects, total_value):
        print("\n" + "-"*50)
        print("РЕЗУЛЬТАТЫ ВЫБОРА ПРОЕКТОВ")
        print("-"*50)
        if not selected_projects:
            print("Не выбран ни один проект")
        else:
            print("\nВыбранные проекты:")
            print("  Проект | Затраты | Прибыль")
            print("-"*30)
            total_weight = 0
            for idx in selected_projects:
                print(f"   {idx+1:2}    |   {self.w[idx]:3}   |   {self.v[idx]:4}")
                total_weight += self.w[idx]
            print(f"\nСуммарные затраты: {total_weight} из {self.W}")
            print(f"Суммарная прибыль: {total_value}")

    def visualize_solution(self, selected_projects, total_value, master=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        selected_flags = [1 if i in selected_projects else 0 for i in range(self.n)]
        colors = ['green' if x == 1 else 'lightgray' for x in selected_flags]

        bars1 = ax1.bar(range(1, self.n + 1), self.v, color=colors, alpha=0.7)
        ax1.set_xlabel('Номер проекта')
        ax1.set_ylabel('Прибыль')
        ax1.set_title('Выбранные инвестиционные проекты')
        ax1.grid(True, alpha=0.3)
        for bar, val in zip(bars1, self.v):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha='center', va='bottom')

        used_budget = sum(self.w[i] for i in selected_projects)
        remaining_budget = self.W - used_budget
        budget_parts = [used_budget, remaining_budget]
        budget_labels = [f'Использовано\n{used_budget}', f'Остаток\n{remaining_budget}']
        budget_colors = ['#66b3ff', '#ff9999']

        ax2.pie(budget_parts, labels=budget_labels, colors=budget_colors, autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
        ax2.set_title(f'Использование бюджета\nСуммарная прибыль: {total_value}')

        fig.suptitle('Решение задачи о рюкзаке (выбор инвестиционных проектов)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if master:
            canvas = FigureCanvasTkAgg(fig, master=master)
            canvas.draw()
            return canvas.get_tk_widget()
        else:
            plt.show()