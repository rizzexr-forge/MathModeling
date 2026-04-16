import numpy as np
from collections import defaultdict
from datetime import datetime

class TransportSolver:
    def __init__(self, supply, demand, costs):
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        self.costs = np.array(costs, dtype=float)
        
        self.original_supply = self.supply.copy()
        self.original_demand = self.demand.copy()
        self.original_costs = self.costs.copy()
        
        self.num_suppliers, self.num_consumers = self.costs.shape
        self.is_balanced = False
        self.plan = None
        self.u = None
        self.v = None
        self.iteration_history = []
        
    def balance_problem(self):
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if abs(total_supply - total_demand) < 1e-6:
            self.is_balanced = True
            print("✓ Задача закрытая. Баланс соблюден.")
            return
        
        if total_supply > total_demand:
            fake = total_supply - total_demand
            print(f"⚠ Задача открытая. Запасы > спрос на {fake:.2f}. Добавлен фиктивный потребитель.")
            self.demand = np.append(self.demand, fake)
            self.costs = np.hstack([self.costs, np.zeros((self.num_suppliers, 1))])
            self.num_consumers += 1
        else:
            fake = total_demand - total_supply
            print(f"⚠ Задача открытая. Спрос > запасы на {fake:.2f}. Добавлен фиктивный поставщик.")
            self.supply = np.append(self.supply, fake)
            self.costs = np.vstack([self.costs, np.zeros((1, self.num_consumers))])
            self.num_suppliers += 1
        self.is_balanced = True

    def north_west_corner(self):
        print("\n📍 Начальный план: Северо-западный угол")
        self.plan = np.zeros((self.num_suppliers, self.num_consumers))
        s, d = self.supply.copy(), self.demand.copy()
        i = j = 0
        
        while i < self.num_suppliers and j < self.num_consumers:
            q = min(s[i], d[j])
            self.plan[i, j] = q
            s[i] -= q; d[j] -= q
            print(f"  [{i+1},{j+1}] = {q:.2f}")
            
            if s[i] < 1e-6: i += 1
            elif d[j] < 1e-6: j += 1
            else: i += 1  # Вырожденность
            
        self._ensure_basis()
        print("✓ Начальный план построен.\n")

    def min_cost_method(self):
        print("\n📍 Начальный план: Минимальная стоимость")
        self.plan = np.zeros((self.num_suppliers, self.num_consumers))
        s, d = self.supply.copy(), self.demand.copy()
        rows_done = np.zeros(self.num_suppliers, bool)
        cols_done = np.zeros(self.num_consumers, bool)
        target = self.num_suppliers + self.num_consumers - 1
        count = 0
        
        while count < target:
            min_c, best = np.inf, None
            for i in range(self.num_suppliers):
                if rows_done[i]: continue
                for j in range(self.num_consumers):
                    if cols_done[j]: continue
                    if self.costs[i, j] < min_c:
                        min_c, best = self.costs[i, j], (i, j)
            if best is None: break
            i, j = best
            q = min(s[i], d[j])
            self.plan[i, j] = q
            s[i] -= q; d[j] -= q; count += 1
            if s[i] < 1e-6: rows_done[i] = True
            if d[j] < 1e-6: cols_done[j] = True
            
        self._ensure_basis()
        print("✓ Начальный план построен.\n")

    def _ensure_basis(self):
        """Гарантирует ровно m+n-1 базисных клеток, добавляя 1e-9 при вырожденности."""
        basic = [(i, j) for i in range(self.num_suppliers) 
                 for j in range(self.num_consumers) if self.plan[i, j] > 1e-6]
        needed = self.num_suppliers + self.num_consumers - 1
        while len(basic) < needed:
            for i in range(self.num_suppliers):
                for j in range(self.num_consumers):
                    if self.plan[i, j] < 1e-6:
                        self.plan[i, j] = 1e-9
                        basic.append((i, j))
                        break
                if len(basic) >= needed: break

    def calculate_potentials(self):
        self.u = np.full(self.num_suppliers, np.nan)
        self.v = np.full(self.num_consumers, np.nan)
        self.u[0] = 0
        
        basis = [(i, j) for i in range(self.num_suppliers) 
                 for j in range(self.num_consumers) if self.plan[i, j] > 1e-9]
        
        # Итеративный расчёт без рекурсии
        for _ in range(1000):  # Защита от зацикливания
            changed = False
            for i, j in basis:
                if not np.isnan(self.u[i]) and np.isnan(self.v[j]):
                    self.v[j] = self.costs[i, j] - self.u[i]
                    changed = True
                elif np.isnan(self.u[i]) and not np.isnan(self.v[j]):
                    self.u[i] = self.costs[i, j] - self.v[j]
                    changed = True
            if not changed: break
            
        # Соединяем компоненты графа, если остались NaN
        while np.isnan(self.u).any() or np.isnan(self.v).any():
            found = False
            for i in range(self.num_suppliers):
                for j in range(self.num_consumers):
                    if self.plan[i, j] < 1e-6:
                        if not np.isnan(self.u[i]) and np.isnan(self.v[j]):
                            self.plan[i, j] = 1e-9; basis.append((i, j))
                            self.v[j] = self.costs[i, j] - self.u[i]; found = True; break
                        elif np.isnan(self.u[i]) and not np.isnan(self.v[j]):
                            self.plan[i, j] = 1e-9; basis.append((i, j))
                            self.u[i] = self.costs[i, j] - self.v[j]; found = True; break
                if found: break
            if not found: break
            
        return True

    def find_cycle(self, si, sj):
        # Строим двудольный граф: строки R0..Rm-1, столбцы Cm..Cm+n-1
        adj = defaultdict(list)
        for i in range(self.num_suppliers):
            for j in range(self.num_consumers):
                if self.plan[i, j] > 1e-9 or (i == si and j == sj):
                    adj[f"R{i}"].append(f"C{j}")
                    adj[f"C{j}"].append(f"R{i}")
                    
        stack = [(f"R{si}", [f"R{si}", f"C{sj}"])]
        visited = set()
        
        while stack:
            node, path = stack.pop()
            if node == f"R{si}" and len(path) > 2:
                return [(int(path[k][1:]), int(path[k+1][1:])) for k in range(0, len(path)-1, 2)]
            for nb in adj[node]:
                edge = tuple(sorted([node, nb]))
                if edge not in visited:
                    visited.add(edge)
                    stack.append((nb, path + [nb]))
        return None

    def recalculate_plan(self, cycle):
        theta = min(self.plan[cycle[k][0], cycle[k][1]] for k in range(1, len(cycle), 2))
        if theta < 1e-9: theta = 1e-9
        print(f"  ↻ θ = {theta:.4f}")
        
        for k, (i, j) in enumerate(cycle):
            self.plan[i, j] += theta if k % 2 == 0 else -theta
        self.plan = np.maximum(self.plan, 0)
        self._ensure_basis()

    def solve(self, method='min_cost', max_iter=50):
        print(f"\n{'='*60}\n🚚 РЕШЕНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ\n{'='*60}")
        self.balance_problem()
        if method == 'north_west': self.north_west_corner()
        else: self.min_cost_method()
        
        for it in range(1, max_iter + 1):
            print(f"\n🔄 Итерация #{it}")
            self.calculate_potentials()
            print(f"  u: {np.round(self.u, 3)}")
            print(f"  v: {np.round(self.v, 3)}")
            
            min_d, cell, deltas = np.inf, None, {}
            for i in range(self.num_suppliers):
                for j in range(self.num_consumers):
                    if self.plan[i, j] < 1e-6:
                        d = self.costs[i, j] - (self.u[i] + self.v[j])
                        deltas[(i, j)] = d
                        if d < min_d: min_d, cell = d, (i, j)
                        
            neg = {k: v for k, v in deltas.items() if v < -1e-6}
            if neg: print(f"  Отрицательные Δ: {neg}")
            print(f"  min Δ = {min_d:.4f} в клетке {cell}")
            self.iteration_history.append({'plan': self.plan.copy(), 'min_delta': min_d})
            
            if min_d >= -1e-6:
                print("✅ План оптимален!"); break
                
            cycle = self.find_cycle(cell[0], cell[1])
            if cycle:
                print(f"  🔁 Цикл: {'→'.join(f'[{i+1},{j+1}]' for i,j in cycle)}")
                self.recalculate_plan(cycle)
            else:
                print("❌ Цикл не найден."); break
        else:
            print("⚠ Превышен лимит итераций.")
        self.print_solution()
        return self.plan

    def print_solution(self):
        print(f"\n{'='*60}\n📊 ОПТИМАЛЬНЫЙ ПЛАН\n{'='*60}")
        for i in range(self.num_suppliers):
            row = "  ".join(f"{self.plan[i,j]:6.2f}" if self.plan[i,j]>1e-6 else "      -" 
                           for j in range(self.num_consumers))
            print(f"P{i+1}: {row} | Запас: {self.original_supply[i] if i<len(self.original_supply) else 'фикт.'}")
        
        total = sum(self.plan[i,j]*self.original_costs[i,j] 
                   for i in range(min(self.num_suppliers, len(self.original_supply)))
                   for j in range(min(self.num_consumers, len(self.original_demand))))
        print(f"\n💰 Минимальные затраты: {total:.2f}")

    def save_report(self, filename='report.txt'):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("ОТЧЁТ: Лабораторная работа №4\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Запасы: {list(self.original_supply)}\nПотребности: {list(self.original_demand)}\n\n")
            for i, h in enumerate(self.iteration_history):
                f.write(f"Итерация {i+1}: min Δ = {h['min_delta']:.4f}\n")
            f.write(f"\nИтоговая стоимость: {np.sum(self.plan[:len(self.original_supply), :len(self.original_demand)] * self.original_costs):.2f}\n")
        print(f"📄 Отчёт сохранён: {filename}")


if __name__ == "__main__":
    print("🎓 ЛАБОРАТОРНАЯ РАБОТА №4: Метод потенциалов\n")
    
    # Тест 1
    print("\n" + "-"*40 + "\n🧪 ТЕСТ 1: Сбалансированная 2x3")
    s1, d1, c1 = [50, 60], [30, 40, 40], [[3,2,5],[4,1,6]]
    sol1 = TransportSolver(s1, d1, c1)
    sol1.solve(method='min_cost')
    sol1.save_report('report_test1.txt')
    
    # Тест 2
    print("\n" + "-"*40 + "\n🧪 ТЕСТ 2: Открытая 3x3 (спрос > запасы)")
    s2, d2, c2 = [100,150,200], [150,200,150], [[10,8,12],[9,7,11],[14,6,13]]
    sol2 = TransportSolver(s2, d2, c2)
    sol2.solve(method='north_west')
    sol2.save_report('report_test2.txt')
    
    print("\n✅ Все тесты завершены!")