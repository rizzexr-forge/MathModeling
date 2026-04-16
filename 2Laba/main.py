import numpy as np

class SimplexSolver:
    """Класс для решения задач линейного программирования симплекс-методом"""
    
    def __init__(self, c, A, b, maximize=True):
        """
        Инициализация симплекс-решателя
        
        Параметры:
        c - коэффициенты целевой функции
        A - матрица ограничений
        b - правые части ограничений
        maximize - максимизация (True) или минимизация (False)
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.maximize = maximize
        
        # Количество переменных и ограничений
        self.n_vars = len(c)
        self.n_constraints = len(b)
        
        # Инициализация симплекс-таблицы
        self.tableau = None
        self.basic_vars = None
        self.non_basic_vars = None
        
    def create_initial_tableau(self):
        """Создание начальной симплекс-таблицы"""
        # Добавляем slack-переменные для преобразования неравенств в равенства
        slack_matrix = np.eye(self.n_constraints)
        
        # Создаем расширенную матрицу [A | I]
        A_augmented = np.hstack([self.A, slack_matrix])
        
        # Коэффициенты целевой функции для всех переменных
        # Основные переменные + slack-переменные (коэффициенты 0 для slack)
        c_full = np.concatenate([self.c, np.zeros(self.n_constraints)])
        
        # Если минимизация, меняем знак коэффициентов целевой функции
        if not self.maximize:
            c_full = -c_full
        
        # Создаем симплекс-таблицу
        self.tableau = np.zeros((self.n_constraints + 1, 
                                 self.n_vars + self.n_constraints + 1))
        
        # Верхняя часть: коэффициенты ограничений
        self.tableau[:-1, :-1] = A_augmented
        self.tableau[:-1, -1] = self.b
        
        # Последняя строка: коэффициенты целевой функции
        self.tableau[-1, :-1] = -c_full
        
        # Индексы базисных и небазисных переменных
        self.basic_vars = list(range(self.n_vars, self.n_vars + self.n_constraints))
        self.non_basic_vars = list(range(self.n_vars))
    
    def find_pivot_column(self):
        """Нахождение ведущего столбца (переменной, вводимой в базис)"""
        # Берем последнюю строку (целевую функцию), исключая последний столбец
        last_row = self.tableau[-1, :-1]
        
        # Для максимизации ищем минимальный отрицательный коэффициент
        # Для минимизации ищем максимальный положительный коэффициент
        if self.maximize:
            # Находим индекс минимального (самого отрицательного) элемента
            min_val = np.min(last_row)
            if min_val >= 0:  # Все коэффициенты неотрицательны - решение оптимально
                return -1
            pivot_col = np.argmin(last_row)
        else:
            # Для минимизации коэффициенты уже инвертированы при создании таблицы
            # Теперь ищем отрицательные коэффициенты
            min_val = np.min(last_row)
            if min_val >= 0:
                return -1
            pivot_col = np.argmin(last_row)
            
        return pivot_col
    
    def find_pivot_row(self, pivot_col):
        """Нахождение ведущей строки (переменной, выводимой из базиса)"""
        ratios = []
        
        for i in range(self.n_constraints):
            if self.tableau[i, pivot_col] > 0:
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                ratios.append(ratio)
            else:
                ratios.append(np.inf)
        
        # Находим строку с минимальным положительным отношением
        min_ratio = min(ratios)
        if min_ratio == np.inf:
            raise ValueError("Задача неограничена")
        
        pivot_row = ratios.index(min_ratio)
        return pivot_row
    
    def pivot(self, pivot_row, pivot_col):
        """Выполнение шага поворота (пересчет симплекс-таблицы)"""
        # Обновляем базисные переменные
        outgoing_var = self.basic_vars[pivot_row]
        self.basic_vars[pivot_row] = pivot_col
        
        if outgoing_var in self.non_basic_vars:
            self.non_basic_vars.remove(outgoing_var)
        
        # Нормализуем ведущую строку
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_element
        
        # Пересчитываем остальные строки
        for i in range(self.n_constraints + 1):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                self.tableau[i, :] -= factor * self.tableau[pivot_row, :]

    def solve(self, max_iterations=100):
        """Основной метод решения задачи"""
        # Создаем начальную симплекс-таблицу
        self.create_initial_tableau()
        
        print("Начальная симплекс-таблица:")
        print(self.tableau)
        print(f"Базисные переменные: {self.basic_vars}")
        print()
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"\nИтерация {iteration}:")
            
            # Проверка оптимальности
            pivot_col = self.find_pivot_column()
            if pivot_col == -1:
                print("Достигнуто оптимальное решение!")
                break
            
            # Нахождение ведущей строки
            pivot_row = self.find_pivot_row(pivot_col)
            
            print(f"Ведущий элемент: строка {pivot_row}, столбец {pivot_col}")
            print(f"Значение ведущего элемента: {self.tableau[pivot_row, pivot_col]}")
            
            # Выполнение поворота
            self.pivot(pivot_row, pivot_col)
            
            print("Обновленная симплекс-таблица:")
            print(self.tableau)
            print(f"Базисные переменные: {self.basic_vars}")
        
        if iteration >= max_iterations:
            print(f"Достигнуто максимальное число итераций ({max_iterations})")
        
        return self.get_solution()
    
    def get_solution(self):
        """Извлечение решения из симплекс-таблицы"""
        solution = np.zeros(self.n_vars + self.n_constraints)
        
        # Значения базисных переменных
        for i, var_idx in enumerate(self.basic_vars):
            if var_idx < len(solution):
                solution[var_idx] = self.tableau[i, -1]
        
        # Значение целевой функции
        objective_value = self.tableau[-1, -1]
        if not self.maximize:
            objective_value = -objective_value
        
        return {
            'x': solution[:self.n_vars],
            'slack': solution[self.n_vars:],
            'z': objective_value,
            'basic_vars': self.basic_vars
        }
    
    def print_solution(self, solution):
        """Вывод решения в удобном формате"""
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ РЕШЕНИЯ")
        print("="*50)
        
        print(f"\nЗначение целевой функции: Z = {solution['z']:.4f}")
        
        print("\nЗначения основных переменных:")
        for i, value in enumerate(solution['x']):
            print(f"  x{i+1} = {value:.4f}")
        
        print("\nЗначения дополнительных (slack) переменных:")
        for i, value in enumerate(solution['slack']):
            print(f"  s{i+1} = {value:.4f}")
        
        print("\nБазисные переменные (индексы):")
        print(f"  {solution['basic_vars']}")



def test_example_1():
    """Тест 1: Простой пример максимизации"""
    print("="*60)
    print("ТЕСТ 1: Максимизация с двумя переменными")
    print("="*60)
    
    # Задача:
    # Максимизировать: Z = 3x₁ + 5x₂
    # При условиях:
    #   x₁ ≤ 4
    #   2x₂ ≤ 12
    #   3x₁ + 2x₂ ≤ 18
    #   x₁, x₂ ≥ 0
    
    c = [3, 5]  # Коэффициенты целевой функции
    A = [[1, 0],  # Матрица ограничений
         [0, 2],
         [3, 2]]
    b = [4, 12, 18]  # Правые части
    
    solver = SimplexSolver(c, A, b, maximize=True)
    solution = solver.solve()
    solver.print_solution(solution)
    
    # Ожидаемое решение: x₁ = 2, x₂ = 6, Z = 36
    print("\nОжидаемое решение: x₁ = 2, x₂ = 6, Z = 36")

def test_example_2():
    """Тест 2: Пример минимизации"""
    print("\n" + "="*60)
    print("ТЕСТ 2: Минимизация с двумя переменными")
    print("="*60)
    
    # Задача:
    # Минимизировать: Z = 6x₁ + 8x₂
    # При условиях:
    #   3x₁ + x₂ ≥ 4
    #   x₁ + 2x₂ ≥ 4
    #   x₁, x₂ ≥ 0
    
    # Преобразуем в стандартную форму (умножаем на -1 для ≥)
    c = [6, 8]
    A = [[-3, -1],  # Умножили на -1 для преобразования ≥ в ≤
         [-1, -2]]
    b = [-4, -4]    # Умножили на -1
    
    solver = SimplexSolver(c, A, b, maximize=False)
    solution = solver.solve()
    solver.print_solution(solution)
    
    print("\nОжидаемое решение: x₁ = 0.8, x₂ = 1.6, Z = 17.6")

def test_example_3():
    """Тест 3: Пример с тремя переменными"""
    print("\n" + "="*60)
    print("ТЕСТ 3: Максимизация с тремя переменными")
    print("="*60)
    
    # Задача:
    # Максимизировать: Z = 4x₁ + 6x₂ + 3x₃
    # При условиях:
    #   x₁ + x₂ + x₃ ≤ 100
    #   10x₁ + 4x₂ + 5x₃ ≤ 600
    #   2x₁ + 2x₂ + 6x₃ ≤ 300
    #   x₁, x₂, x₃ ≥ 0
    
    c = [4, 6, 3]
    A = [[1, 1, 1],
         [10, 4, 5],
         [2, 2, 6]]
    b = [100, 600, 300]
    
    solver = SimplexSolver(c, A, b, maximize=True)
    solution = solver.solve()
    solver.print_solution(solution)

def interactive_example():
    """Интерактивный ввод данных задачи"""
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("="*60)
    
    try:
        n_vars = int(input("Введите количество переменных: "))
        n_constraints = int(input("Введите количество ограничений: "))
        
        print("\nВведите коэффициенты целевой функции:")
        c = []
        for i in range(n_vars):
            coeff = float(input(f"  c{i+1}: "))
            c.append(coeff)
        
        maximize = input("Максимизация (max) или минимизация (min)? ").lower()
        maximize = maximize.startswith('max')
        
        print("\nВведите коэффициенты ограничений (построчно):")
        A = []
        for i in range(n_constraints):
            row = []
            print(f"Ограничение {i+1}:")
            for j in range(n_vars):
                coeff = float(input(f"  a{i+1}{j+1}: "))
                row.append(coeff)
            A.append(row)
        
        print("\nВведите правые части ограничений:")
        b = []
        for i in range(n_constraints):
            value = float(input(f"  b{i+1}: "))
            b.append(value)
        
        solver = SimplexSolver(c, A, b, maximize)
        solution = solver.solve()
        solver.print_solution(solution)
        
    except ValueError as e:
        print(f"Ошибка ввода данных: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    print("ЛАБОРАТОРНАЯ РАБОТА ПО СИМПЛЕКС-МЕТОДУ")
    print("Автоматизация решения задач линейного программирования")
    
    # Запуск тестов
    test_example_1()
    test_example_2()
    test_example_3()
    
    # Интерактивный режим (опционально)
    run_interactive = input("\nЗапустить интерактивный режим? (y/n): ").lower()
    if run_interactive == 'y':
        interactive_example()
    
    print("\nЛабораторная работа завершена!")
