import numpy as np

class TransportSolver:
    def __init__(self, supplies, demands, costs):
        self.supplies = np.array(supplies, dtype=float)
        self.demands = np.array(demands, dtype=float)
        self.costs = np.array(costs, dtype=float)
        self.m = len(supplies)
        self.n = len(demands)

        # Проверка баланса
        total_supply = np.sum(self.supplies)
        total_demand = np.sum(self.demands)

        if not np.isclose(total_supply, total_demand):
            print(f"Задача не сбалансирована! Запасы: {total_supply}, Потребности: {total_demand}")
            if total_supply > total_demand:
                # Добавляем фиктивного потребителя
                self.demands = np.append(self.demands, total_supply - total_demand)
                self.costs = np.column_stack([self.costs, np.zeros(self.m)])
                self.n += 1
            else:
                # Добавляем фиктивного поставщика
                self.supplies = np.append(self.supplies, total_demand - total_supply)
                self.costs = np.row_stack([self.costs, np.zeros(self.n)])
                self.m += 1
            print("Задача сбалансирована добавлением фиктивного элемента.")

    def northwest_corner_method(self):
        supplies = self.supplies.copy()
        demands = self.demands.copy()
        plan = np.zeros((self.m, self.n))
        i, j = 0, 0

        while i < self.m and j < self.n:
            amount = min(supplies[i], demands[j])
            if amount > 0:
                plan[i, j] = amount
                supplies[i] -= amount
                demands[j] -= amount

            if supplies[i] == 0 and i < self.m - 1:
                i += 1
            elif demands[j] == 0 and j < self.n - 1:
                j += 1
            else:
                if i < self.m - 1:
                    i += 1
                if j < self.n - 1:
                    j += 1
                else:
                    break

        total_cost = np.sum(plan * self.costs)
        return plan, total_cost

    def vogel_approximation_method(self):
        supplies = self.supplies.copy()
        demands = self.demands.copy()
        costs = self.costs.copy()
        plan = np.zeros((self.m, self.n))
        active_rows = np.ones(self.m, dtype=bool)
        active_cols = np.ones(self.n, dtype=bool)

        while np.any(active_rows) and np.any(active_cols):
            row_penalties = []
            for i in range(self.m):
                if not active_rows[i]:
                    row_penalties.append(-1)
                    continue
                row_costs = [costs[i, j] for j in range(self.n) if active_cols[j]]
                if len(row_costs) >= 2:
                    sorted_costs = sorted(row_costs)
                    penalty = sorted_costs[1] - sorted_costs[0]
                else:
                    penalty = 0
                row_penalties.append(penalty)

            col_penalties = []
            for j in range(self.n):
                if not active_cols[j]:
                    col_penalties.append(-1)
                    continue
                col_costs = [costs[i, j] for i in range(self.m) if active_rows[i]]
                if len(col_costs) >= 2:
                    sorted_costs = sorted(col_costs)
                    penalty = sorted_costs[1] - sorted_costs[0]
                else:
                    penalty = 0
                col_penalties.append(penalty)

            max_row_penalty = max(row_penalties) if any(p >= 0 for p in row_penalties) else -1
            max_col_penalty = max(col_penalties) if any(p >= 0 for p in col_penalties) else -1

            if max_row_penalty >= max_col_penalty:
                idx = row_penalties.index(max_row_penalty)
                valid_cols = [j for j in range(self.n) if active_cols[j]]
                min_cost_idx = min(valid_cols, key=lambda j: costs[idx, j])
                amount = min(supplies[idx], demands[min_cost_idx])
                if amount > 0:
                    plan[idx, min_cost_idx] = amount
                    supplies[idx] -= amount
                    demands[min_cost_idx] -= amount
                if supplies[idx] == 0:
                    active_rows[idx] = False
                if demands[min_cost_idx] == 0:
                    active_cols[min_cost_idx] = False
            else:
                idx = col_penalties.index(max_col_penalty)
                valid_rows = [i for i in range(self.m) if active_rows[i]]
                min_cost_idx = min(valid_rows, key=lambda i: costs[i, idx])
                amount = min(supplies[min_cost_idx], demands[idx])
                if amount > 0:
                    plan[min_cost_idx, idx] = amount
                    supplies[min_cost_idx] -= amount
                    demands[idx] -= amount
                if supplies[min_cost_idx] == 0:
                    active_rows[min_cost_idx] = False
                if demands[idx] == 0:
                    active_cols[idx] = False

        total_cost = np.sum(plan * self.costs)
        return plan, total_cost

    def print_solution(self, plan, method_name):
        print(f"\n{'='*50}")
        print(f"Решение методом: {method_name}")
        print(f"{'='*50}")
        print("\nМатрица перевозок:")
        for i in range(self.m):
            row_str = ""
            for j in range(self.n):
                if plan[i, j] > 0:
                    row_str += f"[{plan[i, j]:6.1f}]"
                else:
                    row_str += f"{plan[i, j]:6.1f} "
            print(row_str)
        print("\nМатрица стоимостей:")
        for i in range(self.m):
            row_str = ""
            for j in range(self.n):
                row_str += f"{self.costs[i, j]:6.1f} "
            print(row_str)
        total_cost = np.sum(plan * self.costs)
        print(f"\nОбщая стоимость перевозок: {total_cost:.2f}")