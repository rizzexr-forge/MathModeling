"""
Модуль для решения задач линейного программирования графическим методом
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog
from typing import List, Tuple, Optional, Dict
import warnings

class Constraint:
    """Класс для представления линейного ограничения"""
    
    def __init__(self, a: float, b: float, c: float, 
                 constraint_type: str = '<='):
        """
        Инициализация ограничения
        
        Параметры:
        ----------
        a : float
            Коэффициент при x1
        b : float
            Коэффициент при x2  
        c : float
            Правая часть ограничения
        constraint_type : str
            Тип ограничения: '<=' или '>='
        """
        self.a = a
        self.b = b
        self.c = c
        self.type = constraint_type
        
        # Проверка корректности типа ограничения
        if constraint_type not in ['<=', '>=']:
            raise ValueError("Тип ограничения должен быть '<=' или '>='")
    
    def evaluate(self, x1: float, x2: float) -> float:
        """Вычисление левой части ограничения"""
        return self.a * x1 + self.b * x2
    
    def is_satisfied(self, x1: float, x2: float, tol: float = 1e-6) -> bool:
        """Проверка выполнения ограничения"""
        value = self.evaluate(x1, x2)
        
        if self.type == '<=':
            return value <= self.c + tol
        else:  # '>='
            return value >= self.c - tol
    
    def get_line_points(self, x_range: Tuple[float, float] = (0, 100)) -> Tuple[List[float], List[float]]:
        """
        Получение точек для построения линии ограничения
        
        Возвращает:
        -----------
        x_points, y_points : списки координат
        """
        if abs(self.b) > 1e-9:
            # Выражаем x2 через x1: a*x1 + b*x2 = c -> x2 = (c - a*x1)/b
            x1_vals = np.linspace(x_range[0], x_range[1], 100)
            x2_vals = (self.c - self.a * x1_vals) / self.b
        elif abs(self.a) > 1e-9:
            # Вертикальная линия: x1 = c/a
            x1_val = self.c / self.a
            x2_vals = np.linspace(0, 100, 100)
            x1_vals = np.full_like(x2_vals, x1_val)
        else:
            # Нулевое ограничение
            return [], []
        
        return x1_vals.tolist(), x2_vals.tolist()
    
    def __str__(self) -> str:
        """Строковое представление ограничения"""
        return f"{self.a}x1 + {self.b}x2 {self.type} {self.c}"


class LinearProgrammingProblem:
    """Класс для представления задачи линейного программирования"""
    
    def __init__(self, c1: float = 0, c2: float = 0, 
                 objective: str = 'max'):
        """
        Инициализация задачи ЛП
        
        Параметры:
        ----------
        c1 : float
            Коэффициент при x1 в целевой функции
        c2 : float
            Коэффициент при x2 в целевой функции
        objective : str
            Тип задачи: 'max' или 'min'
        """
        self.c1 = c1
        self.c2 = c2
        self.objective = objective
        self.constraints: List[Constraint] = []
        
        # Автоматически добавляем ограничения неотрицательности
        self.constraints.append(Constraint(1, 0, 0, '>='))  # x1 >= 0
        self.constraints.append(Constraint(0, 1, 0, '>='))  # x2 >= 0
    
    def add_constraint(self, constraint: Constraint):
        """Добавление ограничения"""
        self.constraints.append(constraint)
    
    def evaluate_objective(self, x1: float, x2: float) -> float:
        """Вычисление значения целевой функции"""
        return self.c1 * x1 + self.c2 * x2
    
    def is_feasible_point(self, x1: float, x2: float, tol: float = 1e-6) -> bool:
        """Проверка допустимости точки"""
        for constraint in self.constraints:
            if not constraint.is_satisfied(x1, x2, tol):
                return False
        return True
    
    def find_feasible_region_vertices(self) -> List[Tuple[float, float]]:
        """
        Нахождение вершин области допустимых решений
        
        Возвращает:
        -----------
        vertices : список кортежей (x1, x2)
        """
        vertices = []
        n = len(self.constraints)
        
        # Находим пересечения всех пар ограничений
        for i in range(n):
            for j in range(i + 1, n):
                point = self._intersect_constraints(
                    self.constraints[i], 
                    self.constraints[j]
                )
                if point is not None and self.is_feasible_point(*point):
                    vertices.append(point)
        
        # Добавляем точки пересечения с осями
        vertices.extend(self._find_axis_intersections())
        
        # Удаляем дубликаты с учетом погрешности
        unique_vertices = []
        for vertex in vertices:
            if not any(self._are_points_equal(vertex, uv) for uv in unique_vertices):
                unique_vertices.append(vertex)
        
        return unique_vertices
    
    def _intersect_constraints(self, c1: Constraint, c2: Constraint) -> Optional[Tuple[float, float]]:
        """Нахождение точки пересечения двух ограничений"""
        det = c1.a * c2.b - c2.a * c1.b
        
        if abs(det) < 1e-9:
            return None  # Параллельные линии
        
        x1 = (c1.c * c2.b - c2.c * c1.b) / det
        x2 = (c1.a * c2.c - c2.a * c1.c) / det
        
        return (x1, x2)
    
    def _find_axis_intersections(self) -> List[Tuple[float, float]]:
        """Нахождение пересечений ограничений с осями"""
        intersections = []
        
        # Проверяем начало координат
        if self.is_feasible_point(0, 0):
            intersections.append((0.0, 0.0))
        
        # Пересечения с осью x1 (x2 = 0)
        for constraint in self.constraints:
            if abs(constraint.b) > 1e-9:
                x1 = constraint.c / constraint.a if abs(constraint.a) > 1e-9 else 0
                point = (x1, 0)
                if self.is_feasible_point(*point):
                    intersections.append(point)
        
        # Пересечения с осью x2 (x1 = 0)
        for constraint in self.constraints:
            if abs(constraint.a) > 1e-9:
                x2 = constraint.c / constraint.b if abs(constraint.b) > 1e-9 else 0
                point = (0, x2)
                if self.is_feasible_point(*point):
                    intersections.append(point)
        
        return intersections
    
    def _are_points_equal(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                         tol: float = 1e-6) -> bool:
        """Проверка равенства точек с учетом погрешности"""
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol
    
    def find_optimal_solution(self) -> Dict:
        """
        Нахождение оптимального решения графическим методом
        
        Возвращает:
        -----------
        result : словарь с результатами
        """
        vertices = self.find_feasible_region_vertices()
        
        if not vertices:
            return {
                'optimal_point': None,
                'optimal_value': None,
                'vertices': [],
                'status': 'infeasible'
            }
        
        # Вычисляем значения целевой функции в вершинах
        values = []
        for x1, x2 in vertices:
            values.append(self.evaluate_objective(x1, x2))
        
        if self.objective == 'max':
            optimal_idx = np.argmax(values)
        else:  # 'min'
            optimal_idx = np.argmin(values)
        
        optimal_point = vertices[optimal_idx]
        optimal_value = values[optimal_idx]
        
        return {
            'optimal_point': optimal_point,
            'optimal_value': optimal_value,
            'vertices': vertices,
            'values': values,
            'status': 'optimal'
        }
    
    def solve_with_scipy(self) -> Dict:
        """
        Решение задачи с помощью scipy.optimize.linprog
        для проверки результатов
        """
        # Подготовка данных для scipy
        if self.objective == 'max':
            c = [-self.c1, -self.c2]  # Для максимизации
        else:
            c = [self.c1, self.c2]    # Для минимизации
        
        # Разделяем ограничения по типам
        A_ub = []  # Матрица для ограничений <=
        b_ub = []  # Правая часть для <=
        A_eq = []  # Матрица для ограничений =
        b_eq = []  # Правая часть для =
        
        for constraint in self.constraints:
            # Пропускаем ограничения неотрицательности
            if (constraint.a == 1 and constraint.b == 0 and 
                constraint.c == 0 and constraint.type == '>='):
                continue
            if (constraint.a == 0 and constraint.b == 1 and 
                constraint.c == 0 and constraint.type == '>='):
                continue
            
            if constraint.type == '<=':
                A_ub.append([constraint.a, constraint.b])
                b_ub.append(constraint.c)
            elif constraint.type == '>=':
                A_ub.append([-constraint.a, -constraint.b])
                b_ub.append(-constraint.c)
            # Можно добавить поддержку равенств при необходимости
        
        # Решаем задачу
        bounds = [(0, None), (0, None)]  # x1, x2 >= 0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                           bounds=bounds, method='highs')
        
        if result.success:
            x1_opt, x2_opt = result.x
            if self.objective == 'max':
                optimal_value = -result.fun
            else:
                optimal_value = result.fun
            
            return {
                'optimal_point': (x1_opt, x2_opt),
                'optimal_value': optimal_value,
                'success': True,
                'message': result.message
            }
        else:
            return {
                'optimal_point': None,
                'optimal_value': None,
                'success': False,
                'message': result.message
            }
    
    def __str__(self) -> str:
        """Строковое представление задачи"""
        lines = []
        lines.append(f"Целевая функция: {'max' if self.objective == 'max' else 'min'} Z = {self.c1}x1 + {self.c2}x2")
        lines.append("Ограничения:")
        for i, constraint in enumerate(self.constraints, 1):
            lines.append(f"  {i}. {constraint}")
        return "\n".join(lines)


class GraphicalOptimizer:
    """Класс для графического решения и визуализации"""
    
    def __init__(self, problem: LinearProgrammingProblem):
        self.problem = problem
        self.fig = None
        self.ax = None
        
    def plot_solution(self, figsize: Tuple[int, int] = (12, 8), 
                     dpi: int = 100, show_grid: bool = True):
        """
        Построение графического решения
        
        Параметры:
        ----------
        figsize : кортеж
            Размер фигуры
        dpi : int
            Разрешение
        show_grid : bool
            Показывать сетку
        """
        # Создаем фигуру
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Находим оптимальное решение
        result = self.problem.find_optimal_solution()
        vertices = result['vertices']
        
        # Определяем границы графика
        x_max = max([v[0] for v in vertices] + [50]) + 10
        y_max = max([v[1] for v in vertices] + [50]) + 10
        
        # Строим область допустимых решений
        self._plot_feasible_region(vertices)
        
        # Строим ограничения
        self._plot_constraints(x_max, y_max)
        
        # Строим линии уровня
        self._plot_objective_lines(x_max, y_max)
        
        # Отмечаем оптимальную точку
        if result['optimal_point'] is not None:
            self._plot_optimal_point(result['optimal_point'], 
                                   result['optimal_value'])
        
        # Настраиваем график
        self._configure_plot(x_max, y_max, show_grid)
        
        # Добавляем легенду
        self._add_legend()
        
        # Добавляем информацию о решении
        self._add_solution_info(result)
        
        plt.tight_layout()
        
        return self.fig, self.ax
    
    def _plot_feasible_region(self, vertices: List[Tuple[float, float]]):
        """Построение области допустимых решений"""
        if len(vertices) >= 3:
            # Сортируем вершины по углу для правильного построения полигона
            center = np.mean(vertices, axis=0)
            angles = np.arctan2(
                [v[1] - center[1] for v in vertices],
                [v[0] - center[0] for v in vertices]
            )
            sorted_vertices = [v for _, v in sorted(zip(angles, vertices))]
            
            # Создаем полигон
            polygon = Polygon(sorted_vertices, closed=True, 
                            alpha=0.3, color='lightblue', 
                            label='Область допустимых решений')
            self.ax.add_patch(polygon)
            
            # Рисуем вершины
            for i, (x, y) in enumerate(sorted_vertices):
                self.ax.plot(x, y, 'bo', markersize=8)
                self.ax.annotate(f'V{i+1}\n({x:.1f},{y:.1f})', 
                               xy=(x, y), xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=9, color='blue')
    
    def _plot_constraints(self, x_max: float, y_max: float):
        """Построение линий ограничений"""
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for i, constraint in enumerate(self.problem.constraints):
            if i >= len(colors):
                color = 'gray'
            else:
                color = colors[i]
            
            x_vals, y_vals = constraint.get_line_points((0, x_max))
            
            if x_vals and y_vals:
                # Рисуем линию ограничения
                line, = self.ax.plot(x_vals, y_vals, '-', 
                                   color=color, linewidth=2,
                                   label=f'{constraint}')
                
                # Добавляем стрелку для направления допустимой области
                if len(x_vals) > 10:
                    mid_idx = len(x_vals) // 2
                    x_mid = x_vals[mid_idx]
                    y_mid = y_vals[mid_idx]
                    
                    # Вычисляем нормаль к линии
                    if constraint.type == '<=':
                        # Для a*x1 + b*x2 <= c допустимая область ниже/левее линии
                        dx = -constraint.b
                        dy = constraint.a
                    else:  # '>='
                        dx = constraint.b
                        dy = -constraint.a
                    
                    # Нормализуем вектор
                    norm = np.sqrt(dx**2 + dy**2)
                    if norm > 0:
                        dx /= norm
                        dy /= norm
                        
                        # Рисуем стрелку
                        self.ax.arrow(x_mid, y_mid, dx*5, dy*5, 
                                    head_width=2, head_length=3,
                                    fc=color, ec=color, alpha=0.5)
    
    def _plot_objective_lines(self, x_max: float, y_max: float):
        """Построение линий уровня целевой функции"""
        if self.problem.c1 == 0 and self.problem.c2 == 0:
            return
        
        # Находим диапазон значений целевой функции
        result = self.problem.find_optimal_solution()
        
        if result['optimal_value'] is not None:
            optimal_value = result['optimal_value']
            
            # Строим несколько линий уровня
            n_lines = 4
            if self.problem.objective == 'max':
                values = np.linspace(optimal_value * 0.3, optimal_value, n_lines)
            else:
                values = np.linspace(optimal_value, optimal_value * 1.7, n_lines)
            
            for i, value in enumerate(values):
                # Вычисляем точки для линии уровня
                if abs(self.problem.c2) > 1e-9:
                    x1_vals = np.linspace(0, x_max, 100)
                    x2_vals = (value - self.problem.c1 * x1_vals) / self.problem.c2
                elif abs(self.problem.c1) > 1e-9:
                    x2_vals = np.linspace(0, y_max, 100)
                    x1_vals = np.full_like(x2_vals, value / self.problem.c1)
                else:
                    continue
                
                # Фильтруем точки в пределах графика
                mask = (x2_vals >= 0) & (x2_vals <= y_max)
                x1_vals = x1_vals[mask]
                x2_vals = x2_vals[mask]
                
                if len(x1_vals) > 1:
                    # Для оптимальной линии используем другой стиль
                    if i == n_lines - 1:  # Последняя линия - оптимальная
                        self.ax.plot(x1_vals, x2_vals, '--', 
                                   color='darkgreen', linewidth=2,
                                   label=f'Z = {value:.1f} (оптимальная)')
                    else:
                        self.ax.plot(x1_vals, x2_vals, ':', 
                                   color='green', alpha=0.5,
                                   label=f'Z = {value:.1f}')
    
    def _plot_optimal_point(self, point: Tuple[float, float], 
                          value: float):
        """Отображение оптимальной точки"""
        x1, x2 = point
        
        self.ax.plot(x1, x2, 'r*', markersize=15, 
                   label=f'Оптимальная точка\n({x1:.1f}, {x2:.1f})')
        
        # Добавляем аннотацию
        self.ax.annotate(f'Z* = {value:.1f}', xy=(x1, x2), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='yellow', alpha=0.7))
    
    def _configure_plot(self, x_max: float, y_max: float, show_grid: bool):
        """Настройка внешнего вида графика"""
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)
        
        self.ax.set_xlabel('x₁', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('x₂', fontsize=12, fontweight='bold')
        
        self.ax.set_title('Графический метод решения задачи ЛП', 
                         fontsize=14, fontweight='bold', pad=20)
        
        if show_grid:
            self.ax.grid(True, alpha=0.3, linestyle='--')
        
        self.ax.set_aspect('equal', adjustable='box')
        
        # Добавляем оси
        self.ax.axhline(y=0, color='k', linewidth=0.5)
        self.ax.axvline(x=0, color='k', linewidth=0.5)
    
    def _add_legend(self):
        """Добавление легенды"""
        # Собираем все метки из графиков
        handles, labels = self.ax.get_legend_handles_labels()
        
        # Убираем дубликаты
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        # Добавляем легенду
        if unique_handles:
            self.ax.legend(handles=unique_handles, labels=unique_labels,
                         loc='upper left', bbox_to_anchor=(1.05, 1),
                         borderaxespad=0., fontsize=9)
    
    def _add_solution_info(self, result: Dict):
        """Добавление информации о решении"""
        info_text = "Результаты решения:\n"
        info_text += f"Тип задачи: {self.problem.objective.upper()}\n"
        info_text += f"Целевая функция: Z = {self.problem.c1}x₁ + {self.problem.c2}x₂\n\n"
        
        if result['status'] == 'optimal':
            x1_opt, x2_opt = result['optimal_point']
            info_text += f"Оптимальное решение:\n"
            info_text += f"  x₁* = {x1_opt:.2f}\n"
            info_text += f"  x₂* = {x2_opt:.2f}\n"
            info_text += f"  Z* = {result['optimal_value']:.2f}\n\n"
            
            info_text += f"Вершин ОДР: {len(result['vertices'])}\n"
            for i, (x1, x2) in enumerate(result['vertices'], 1):
                info_text += f"  V{i}: ({x1:.1f}, {x2:.1f})"
                if (abs(x1 - x1_opt) < 1e-6 and abs(x2 - x2_opt) < 1e-6):
                    info_text += " ← оптимальная\n"
                else:
                    info_text += f", Z = {self.problem.evaluate_objective(x1, x2):.1f}\n"
        elif result['status'] == 'infeasible':
            info_text += "Задача не имеет допустимых решений!\n"
        else:
            info_text += "Решение не найдено\n"
        
        # Добавляем текст на график
        self.ax.text(1.05, 0.5, info_text, transform=self.ax.transAxes,
                   fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def save_plot(self, filename: str = 'graphical_solution.png'):
        """Сохранение графика в файл"""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"График сохранен в файл: {filename}")
    
    def show_interactive_plot(self):
        """Показ интерактивного графика"""
        plt.show()


class SensitivityAnalyzer:
    """Класс для анализа чувствительности"""
    
    def __init__(self, problem: LinearProgrammingProblem):
        self.problem = problem
    
    def analyze_objective_coefficients(self, 
                                     c1_range: Tuple[float, float] = (1, 5),
                                     c2_range: Tuple[float, float] = (1, 5),
                                     steps: int = 10):
        """
        Анализ чувствительности к коэффициентам целевой функции
        
        Возвращает:
        -----------
        results : DataFrame с результатами
        """
        import pandas as pd
        
        c1_values = np.linspace(c1_range[0], c1_range[1], steps)
        c2_values = np.linspace(c2_range[0], c2_range[1], steps)
        
        results = []
        
        for c1 in c1_values:
            for c2 in c2_values:
                # Создаем копию задачи с новыми коэффициентами
                temp_problem = LinearProgrammingProblem(c1, c2, self.problem.objective)
                for constraint in self.problem.constraints[2:]:  # Пропускаем x1>=0, x2>=0
                    temp_problem.add_constraint(constraint)
                
                # Решаем задачу
                result = temp_problem.find_optimal_solution()
                
                if result['optimal_point'] is not None:
                    x1_opt, x2_opt = result['optimal_point']
                    results.append({
                        'c1': c1,
                        'c2': c2,
                        'x1_opt': x1_opt,
                        'x2_opt': x2_opt,
                        'Z_opt': result['optimal_value']
                    })
        
        return pd.DataFrame(results)
    
    def plot_sensitivity_heatmap(self):
        """Построение тепловой карты чувствительности"""
        df = self.analyze_objective_coefficients()
        
        if df.empty:
            print("Нет данных для анализа чувствительности")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Тепловая карта для x1
        pivot_x1 = df.pivot(index='c2', columns='c1', values='x1_opt')
        im1 = axes[0, 0].imshow(pivot_x1.values, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Оптимальное значение x₁')
        axes[0, 0].set_xlabel('c₁')
        axes[0, 0].set_ylabel('c₂')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Тепловая карта для x2
        pivot_x2 = df.pivot(index='c2', columns='c1', values='x2_opt')
        im2 = axes[0, 1].imshow(pivot_x2.values, aspect='auto', cmap='plasma')
        axes[0, 1].set_title('Оптимальное значение x₂')
        axes[0, 1].set_xlabel('c₁')
        axes[0, 1].set_ylabel('c₂')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Тепловая карта для Z
        pivot_z = df.pivot(index='c2', columns='c1', values='Z_opt')
        im3 = axes[1, 0].imshow(pivot_z.values, aspect='auto', cmap='hot')
        axes[1, 0].set_title('Оптимальное значение Z')
        axes[1, 0].set_xlabel('c₁')
        axes[1, 0].set_ylabel('c₂')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # График зависимости Z от c1 при фиксированном c2
        c2_mid = df['c2'].median()
        df_fixed_c2 = df[abs(df['c2'] - c2_mid) < 0.1]
        axes[1, 1].plot(df_fixed_c2['c1'], df_fixed_c2['Z_opt'], 'b-o')
        axes[1, 1].set_title(f'Зависимость Z от c₁ (при c₂ ≈ {c2_mid:.1f})')
        axes[1, 1].set_xlabel('c₁')
        axes[1, 1].set_ylabel('Z')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def create_default_problem() -> LinearProgrammingProblem:
    """Создание задачи по умолчанию"""
    problem = LinearProgrammingProblem(c1=3, c2=2, objective='max')
    
    # Добавляем ограничения
    problem.add_constraint(Constraint(2, 1, 100, '<='))  # 2x1 + x2 <= 100
    problem.add_constraint(Constraint(1, 1, 80, '<='))   # x1 + x2 <= 80
    problem.add_constraint(Constraint(1, 0, 40, '<='))   # x1 <= 40
    
    return problem


def test_problem():
    """Тестирование решения задачи"""
    problem = create_default_problem()
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ГРАФИЧЕСКОГО МЕТОДА")
    print("=" * 60)
    print()
    print(problem)
    print()
    
    # Решение графическим методом
    print("1. Решение графическим методом:")
    result = problem.find_optimal_solution()
    
    if result['status'] == 'optimal':
        x1_opt, x2_opt = result['optimal_point']
        print(f"   Оптимальная точка: ({x1_opt:.2f}, {x2_opt:.2f})")
        print(f"   Оптимальное значение: Z = {result['optimal_value']:.2f}")
        
        print("\n   Вершины ОДР:")
        for i, (x1, x2) in enumerate(result['vertices'], 1):
            z = problem.evaluate_objective(x1, x2)
            print(f"   V{i}: ({x1:.1f}, {x2:.1f}), Z = {z:.1f}")
    else:
        print("   Задача не имеет допустимых решений")
    
    print("\n" + "-" * 60)
    
    # Решение с помощью scipy (для проверки)
    print("2. Проверка с помощью scipy.optimize.linprog:")
    scipy_result = problem.solve_with_scipy()
    
    if scipy_result['success']:
        x1_scipy, x2_scipy = scipy_result['optimal_point']
        print(f"   Решение scipy: ({x1_scipy:.2f}, {x2_scipy:.2f})")
        print(f"   Значение Z: {scipy_result['optimal_value']:.2f}")
        
        # Сравнение результатов
        if result['status'] == 'optimal':
            diff_x1 = abs(x1_opt - x1_scipy)
            diff_x2 = abs(x2_opt - x2_scipy)
            print(f"\n   Разница в решениях:")
            print(f"   Δx₁ = {diff_x1:.6f}")
            print(f"   Δx₂ = {diff_x2:.6f}")
            
            if diff_x1 < 1e-6 and diff_x2 < 1e-6:
                print("   ✓ Решения совпадают!")
            else:
                print("   ⚠ Решения различаются")
    else:
        print(f"   Ошибка при решении: {scipy_result['message']}")
    
    print("\n" + "=" * 60)
    
    # Визуализация
    print("\n3. Построение графика...")
    optimizer = GraphicalOptimizer(problem)
    fig, ax = optimizer.plot_solution()
    
    optimizer.save_plot('graphical_solution.png')
    optimizer.show_interactive_plot()
    
    return problem, result
# В КОНЕЦ ФАЙЛА linear_programming.py добавьте:

if __name__ == "__main__":
    # Запуск теста
    test_problem()