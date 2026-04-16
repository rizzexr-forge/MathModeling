import numpy as np

class GomoriSolver:
    def __init__(self, tolerance=1e-7):
        self.tolerance = tolerance
        self.iterations = 0
        self.log = []

    def _is_integer(self, val):
        return abs(val - round(val)) < self.tolerance

    def _fractional_part(self, val):
        # Дробная часть по методу Гомори: f = val - floor(val)
        if val >= 0:
            return val - np.floor(val)
        else:
            return val - np.floor(val)

    def _simplex(self, tableau, basis):
        m, n = tableau.shape
        m -= 1
        n -= 1

        while True:
            if np.all(tableau[-1, :-1] >= -self.tolerance):
                return tableau, basis, True, "Оптимальное решение найдено."

            entering = np.argmin(tableau[-1, :-1])

            if np.all(tableau[:-1, entering] <= self.tolerance):
                return tableau, basis, False, "Задача не ограничена."

            ratios = []
            for i in range(m):
                if tableau[i, entering] > self.tolerance:
                    ratios.append(tableau[i, -1] / tableau[i, entering])
                else:
                    ratios.append(np.inf)

            leaving = np.argmin(ratios)
            if ratios[leaving] == np.inf:
                return tableau, basis, False, "Задача не ограничена (нет допустимых решений)."

            pivot = tableau[leaving, entering]
            tableau[leaving, :] /= pivot
            for i in range(m + 1):
                if i != leaving:
                    factor = tableau[i, entering]
                    tableau[i, :] -= factor * tableau[leaving, :]

            basis[leaving] = entering

    def _dual_simplex(self, tableau, basis):
        m, n = tableau.shape
        m -= 1
        n -= 1

        while True:
            if np.all(tableau[:-1, -1] >= -self.tolerance):
                if np.all(tableau[-1, :-1] >= -self.tolerance):
                    return tableau, basis, True, "Оптимальное решение найдено после отсечения."
                else:
                    return self._simplex(tableau, basis)

            leaving = np.argmin(tableau[:-1, -1])
            if tableau[leaving, -1] >= -self.tolerance:
                continue

            if np.all(tableau[leaving, :-1] >= -self.tolerance):
                return tableau, basis, False, "Двойственная задача неразрешима."

            ratios = []
            for j in range(n):
                if tableau[leaving, j] < -self.tolerance:
                    ratio = abs(tableau[-1, j] / tableau[leaving, j])
                    ratios.append((ratio, j))

            if not ratios:
                return tableau, basis, False, "Ошибка выбора ведущего элемента."

            entering = min(ratios, key=lambda x: x[0])[1]

            pivot = tableau[leaving, entering]
            tableau[leaving, :] /= pivot
            for i in range(m + 1):
                if i != leaving:
                    factor = tableau[i, entering]
                    tableau[i, :] -= factor * tableau[leaving, :]
            basis[leaving] = entering

    def solve(self, c, A, b, maximize=True):
        self.iterations = 0
        self.log = []
        m = len(A)
        n = len(A[0])

        tableau = np.zeros((m + 1, n + m + 1))
        for i in range(m):
            for j in range(n):
                tableau[i, j] = A[i][j]
            tableau[i, n + i] = 1.0
            tableau[i, -1] = b[i]

        for j in range(n):
            tableau[-1, j] = -c[j] if maximize else c[j]

        basis = list(range(n, n + m))
        self.log.append("Начальная симплекс-таблица (ослабленная задача):")
        self.log.append(str(np.round(tableau, 4)))

        tableau, basis, success, message = self._simplex(tableau.copy(), basis.copy())
        if not success:
            return {'success': False, 'message': message, 'iterations': self.iterations, 'x': None, 'z': None}
        self.iterations += 1
        self.log.append(f"Итерация {self.iterations}: {message}")
        self.log.append(str(np.round(tableau, 4)))

        while True:
            x = [0] * n
            for i, var_index in enumerate(basis):
                if var_index < n:
                    x[var_index] = tableau[i, -1]

            integer_solution = all(self._is_integer(val) for val in x)
            if integer_solution:
                z = tableau[-1, -1] if maximize else -tableau[-1, -1]
                self.log.append("Получено целочисленное решение!")
                return {
                    'success': True, 'message': 'Целочисленное решение найдено',
                    'iterations': self.iterations, 'x': [round(v) for v in x], 'z': z,
                    'log': '\n'.join(self.log)
                }

            max_frac = 0
            row_index = -1
            for i, var_index in enumerate(basis):
                if var_index < n:
                    frac = self._fractional_part(tableau[i, -1])
                    if frac > max_frac + self.tolerance:
                        max_frac = frac
                        row_index = i

            if row_index == -1:
                for i, var_index in enumerate(basis):
                    if not self._is_integer(tableau[i, -1]):
                        row_index = i
                        break

            self.log.append(f"Строка для отсечения: {row_index} (базисная переменная x_{basis[row_index]+1})")

            new_row = np.zeros(tableau.shape[1])
            f0 = self._fractional_part(tableau[row_index, -1])
            for j in range(tableau.shape[1] - 1):
                fj = self._fractional_part(tableau[row_index, j])
                new_row[j] = -fj
            new_row[-1] = -f0

            self.log.append(f"Отсечение: {np.round(new_row, 4)}")

            new_tableau = np.zeros((tableau.shape[0] + 1, tableau.shape[1] + 1))
            new_tableau[:-1, :-2] = tableau[:, :-1]
            new_tableau[:-1, -1] = tableau[:, -1]
            new_tableau[-1, -2] = 1.0
            new_tableau[-1, :new_row.shape[0]-1] = new_row[:-1]
            new_tableau[-1, -1] = new_row[-1]

            new_basis = basis + [new_tableau.shape[1] - 2]
            tableau, basis = new_tableau, new_basis
            self.iterations += 1
            self.log.append(f"Итерация {self.iterations}: Добавлено отсечение. Таблица перед двойственным методом:")
            self.log.append(str(np.round(tableau, 4)))

            tableau, basis, success, message = self._dual_simplex(tableau, basis)
            if not success:
                return {'success': False, 'message': message, 'iterations': self.iterations, 'x': None, 'z': None}
            self.log.append(f"После двойственного метода: {message}")
            self.log.append(str(np.round(tableau, 4)))