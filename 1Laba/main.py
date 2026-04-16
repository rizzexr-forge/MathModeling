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

4.2. Интерактивное приложение с Streamlit
app_streamlit.py - веб-интерфейс для решения задач
python
"""
Веб-приложение для графического решения задач ЛП
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_programming import (
    LinearProgrammingProblem, 
    Constraint, 
    GraphicalOptimizer,
    SensitivityAnalyzer,
    create_default_problem
)

# Настройка страницы
st.set_page_config(
    page_title="Графический метод оптимизации",
    page_icon="📈",
    layout="wide"
)

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #3498db;
    }
    .warning-box {
        background-color: #fff3cd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Заголовок приложения
    st.markdown('<h1 class="main-header">📈 Графический метод решения задач ЛП</h1>', 
                unsafe_allow_html=True)
    
    # Создаем две колонки
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h2 class="section-header">Ввод данных</h2>', 
                   unsafe_allow_html=True)
        
        # Выбор типа задачи
        problem_type = st.radio(
            "Тип задачи:",
            ["Максимизация", "Минимизация"],
            horizontal=True
        )
        
        # Коэффициенты целевой функции
        st.subheader("Целевая функция:")
        col_a, col_b = st.columns(2)
        with col_a:
            c1 = st.number_input("Коэффициент при x₁:", 
                               value=3.0, step=0.5, format="%.1f")
        with col_b:
            c2 = st.number_input("Коэффициент при x₂:", 
                               value=2.0, step=0.5, format="%.1f")
        
        st.markdown(f"**Z = {c1}x₁ + {c2}x₂ → {'max' if problem_type == 'Максимизация' else 'min'}**")
        
        # Ввод ограничений
        st.subheader("Ограничения:")
        
        # Количество ограничений (кроме неотрицательности)
        num_constraints = st.slider("Количество ограничений:", 
                                   min_value=1, max_value=5, value=3)
        
        constraints = []
        for i in range(num_constraints):
            st.markdown(f"**Ограничение {i+1}:**")
            
            col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
            
            with col1:
                a = st.number_input(f"a{i+1}", value=2.0 if i==0 else 1.0 if i==1 else 1.0, 
                                  key=f"a{i}", format="%.1f")
            
            with col2:
                b = st.number_input(f"b{i+1}", value=1.0 if i==0 else 1.0 if i==1 else 0.0, 
                                  key=f"b{i}", format="%.1f")
            
            with col3:
                st.markdown("<div style='text-align: center; margin-top: 10px;'>x₁ +</div>", 
                          unsafe_allow_html=True)
            
            with col4:
                st.markdown("<div style='text-align: center; margin-top: 10px;'>x₂</div>", 
                          unsafe_allow_html=True)
            
            col5, col6 = st.columns([1, 2])
            with col5:
                sign = st.selectbox(f"Знак {i+1}", ["≤", "≥"], key=f"sign{i}")
            with col6:
                c = st.number_input(f"Значение {i+1}", 
                                  value=100.0 if i==0 else 80.0 if i==1 else 40.0, 
                                  key=f"c{i}", format="%.1f")
            
            constraints.append({
                'a': a, 'b': b, 'c': c, 
                'sign': '<=' if sign == '≤' else '>='
            })
        
        # Кнопки управления
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            solve_btn = st.button("🔍 Решить задачу", use_container_width=True)
        with col_btn2:
            default_btn = st.button("📋 Пример задачи", use_container_width=True)
        with col_btn3:
            clear_btn = st.button("🧹 Очистить", use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Результаты решения</h2>', 
                   unsafe_allow_html=True)
        
        # Область для вывода результатов
        result_container = st.container()
        
        if 'problem' not in st.session_state:
            st.session_state.problem = None
            st.session_state.result = None
        
        # Обработка кнопок
        if default_btn:
            # Загрузка примера задачи
            st.session_state.problem = create_default_problem()
            solve_btn = True
        
        if solve_btn or st.session_state.problem is not None:
            if st.session_state.problem is None:
                # Создаем задачу из введенных данных
                objective = 'max' if problem_type == 'Максимизация' else 'min'
                problem = LinearProgrammingProblem(c1, c2, objective)
                
                for constr in constraints:
                    problem.add_constraint(
                        Constraint(constr['a'], constr['b'], 
                                 constr['c'], constr['sign'])
                    )
                
                st.session_state.problem = problem
            
            # Решаем задачу
            problem = st.session_state.problem
            result = problem.find_optimal_solution()
            st.session_state.result = result
            
            with result_container:
                # Вывод текстовых результатов
                if result['status'] == 'optimal':
                    x1_opt, x2_opt = result['optimal_point']
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.success("✅ Решение найдено!")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Оптимальное x₁", f"{x1_opt:.2f}")
                    with col_res2:
                        st.metric("Оптимальное x₂", f"{x2_opt:.2f}")
                    
                    st.metric("Оптимальное значение Z", f"{result['optimal_value']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Таблица вершин
                    st.subheader("Вершины области допустимых решений:")
                    vertices_data = []
                    for i, (x1, x2) in enumerate(result['vertices'], 1):
                        z = problem.evaluate_objective(x1, x2)
                        vertices_data.append({
                            'Вершина': f'V{i}',
                            'x₁': f"{x1:.2f}",
                            'x₂': f"{x2:.2f}",
                            'Z': f"{z:.2f}",
                            'Статус': 'Оптимальная' if (abs(x1 - x1_opt) < 1e-6 and abs(x2 - x2_opt) < 1e-6) else ''
                        })
                    
                    df_vertices = pd.DataFrame(vertices_data)
                    st.dataframe(df_vertices, use_container_width=True)
                    
                elif result['status'] == 'infeasible':
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.error("❌ Задача не имеет допустимых решений!")
                    st.info("Попробуйте изменить ограничения.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Построение графика
                st.subheader("Графическое решение:")
                try:
                    optimizer = GraphicalOptimizer(problem)
                    fig, ax = optimizer.plot_solution(figsize=(10, 8))
                    st.pyplot(fig)
                    
                    # Кнопка сохранения графика
                    if st.button("💾 Сохранить график"):
                        optimizer.save_plot('graphical_solution.png')
                        st.success("График сохранен как 'graphical_solution.png'")
                        
                except Exception as e:
                    st.error(f"Ошибка при построении графика: {str(e)}")
                
                # Анализ чувствительности
                if result['status'] == 'optimal':
                    st.subheader("Анализ чувствительности:")
                    
                    tab1, tab2, tab3 = st.tabs(["Коэффициенты", "Ограничения", "Визуализация"])
                    
                    with tab1:
                        st.write("Изменение коэффициентов целевой функции:")
                        
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            c1_change = st.slider("Изменение c₁", -2.0, 2.0, 0.0, 0.1)
                        with col_c2:
                            c2_change = st.slider("Изменение c₂", -2.0, 2.0, 0.0, 0.1)
                        
                        # Пересчет с измененными коэффициентами
                        new_c1 = problem.c1 + c1_change
                        new_c2 = problem.c2 + c2_change
                        
                        new_problem = LinearProgrammingProblem(new_c1, new_c2, problem.objective)
                        for constraint in problem.constraints[2:]:  # Пропускаем x1>=0, x2>=0
                            new_problem.add_constraint(constraint)
                        
                        new_result = new_problem.find_optimal_solution()
                        
                        if new_result['status'] == 'optimal':
                            new_x1, new_x2 = new_result['optimal_point']
                            st.info(f"При c₁={new_c1:.1f}, c₂={new_c2:.1f}:")
                            st.write(f"x₁ = {new_x1:.2f}, x₂ = {new_x2:.2f}, Z = {new_result['optimal_value']:.2f}")
                    
                    with tab2:
                        st.write("Анализ ресурсов:")
                        
                        # Определяем активные ограничения
                        active_constraints = []
                        for i, constraint in enumerate(problem.constraints[2:], 1):
                            x1_opt, x2_opt = result['optimal_point']
                            value = constraint.evaluate(x1_opt, x2_opt)
                            
                            if abs(value - constraint.c) < 1e-6:
                                status = "🔴 Активное (дефицитное)"
                            else:
                                status = "🟢 Неактивное"
                            
                            slack = constraint.c - value if constraint.type == '<=' else value - constraint.c
                            active_constraints.append({
                                'Ограничение': str(constraint),
                                'Использовано': f"{value:.1f}",
                                'Доступно': f"{constraint.c:.1f}",
                                'Запас': f"{slack:.1f}",
                                'Статус': status
                            })
                        
                        df_constraints = pd.DataFrame(active_constraints)
                        st.dataframe(df_constraints, use_container_width=True)
                    
                    with tab3:
                        st.write("Визуализация чувствительности:")
                        if st.button("Построить тепловые карты"):
                            analyzer = SensitivityAnalyzer(problem)
                            fig_sens = analyzer.plot_sensitivity_heatmap()
                            st.pyplot(fig_sens)
        
        if clear_btn:
            st.session_state.problem = None
            st.session_state.result = None
            st.rerun()
    
    # Нижняя панель с информацией
    st.markdown("---")
    st.markdown("""
    ### 📚 Теоретическая справка
    
    **Графический метод** позволяет решать задачи линейного программирования с **двумя переменными**.
    
    **Основные шаги:**
    1. Построение **области допустимых решений** (ОДР)
    2. Построение **линий уровня** целевой функции
    3. Определение **оптимальной точки** в вершине ОДР
    
    **Особые случаи:**
    - **Альтернативные оптимумы** - если целевая функция параллельна стороне ОДР
    - **Неограниченное решение** - если ОДР неограничена в направлении улучшения Z
    - **Недопустимая задача** - если ОДР пуста
    
    ⚠ **Ограничения метода:** работает только для задач с двумя переменными.
    """)

if __name__ == "__main__":
    main()
4.3. Примеры задач для тестирования
examples.py - набор тестовых задач
python
"""
Примеры задач для тестирования графического метода
"""

from linear_programming import LinearProgrammingProblem, Constraint

def example_maximization():
    """Пример задачи максимизации"""
    problem = LinearProgrammingProblem(c1=3, c2=2, objective='max')
    
    problem.add_constraint(Constraint(2, 1, 100, '<='))  # 2x1 + x2 <= 100
    problem.add_constraint(Constraint(1, 1, 80, '<='))   # x1 + x2 <= 80
    problem.add_constraint(Constraint(1, 0, 40, '<='))   # x1 <= 40
    
    return problem

def example_minimization():
    """Пример задачи минимизации"""
    problem = LinearProgrammingProblem(c1=2, c2=5, objective='min')
    
    problem.add_constraint(Constraint(3, 2, 60, '>='))  # 3x1 + 2x2 >= 60
    problem.add_constraint(Constraint(1, 2, 40, '>='))  # x1 + 2x2 >= 40
    
    return problem

def example_unbounded():
    """Пример задачи с неограниченным решением"""
    problem = LinearProgrammingProblem(c1=1, c2=1, objective='max')
    
    problem.add_constraint(Constraint(1, -1, 10, '<='))  # x1 - x2 <= 10
    problem.add_constraint(Constraint(-1, 2, 20, '<='))  # -x1 + 2x2 <= 20
    
    return problem

def example_infeasible():
    """Пример недопустимой задачи"""
    problem = LinearProgrammingProblem(c1=1, c2=1, objective='max')
    
    problem.add_constraint(Constraint(1, 1, 10, '<='))   # x1 + x2 <= 10
    problem.add_constraint(Constraint(1, 1, 20, '>='))   # x1 + x2 >= 20
    
    return problem

def example_alternative_optima():
    """Пример задачи с альтернативными оптимумами"""
    problem = LinearProgrammingProblem(c1=1, c2=1, objective='max')
    
    problem.add_constraint(Constraint(1, 2, 8, '<='))    # x1 + 2x2 <= 8
    problem.add_constraint(Constraint(4, 0, 16, '<='))   # 4x1 <= 16
    problem.add_constraint(Constraint(0, 4, 12, '<='))   # 4x2 <= 12
    
    return problem

def run_all_examples():
    """Запуск всех примеров"""
    examples = [
        ("Максимизация", example_maximization),
        ("Минимизация", example_minimization),
        ("Неограниченная", example_unbounded),
        ("Недопустимая", example_infeasible),
        ("Альтернативные оптимумы", example_alternative_optima)
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"Пример: {name}")
        print('='*60)
        
        problem = example_func()
        print(problem)
        
        result = problem.find_optimal_solution()
        
        if result['status'] == 'optimal':
            x1, x2 = result['optimal_point']
            print(f"\nОптимальное решение:")
            print(f"  x₁ = {x1:.2f}, x₂ = {x2:.2f}")
            print(f"  Z = {result['optimal_value']:.2f}")
            
            print(f"\nВершины ОДР ({len(result['vertices'])}):")
            for i, (vx, vy) in enumerate(result['vertices'], 1):
                print(f"  V{i}: ({vx:.1f}, {vy:.1f})")
        elif result['status'] == 'infeasible':
            print("\nЗадача не имеет допустимых решений!")
        else:
            print(f"\nСтатус: {result['status']}")
        
        # Проверка с помощью scipy
        scipy_result = problem.solve_with_scipy()
        if scipy_result['success']:
            sx1, sx2 = scipy_result['optimal_point']
            print(f"\nПроверка (scipy):")
            print(f"  x₁ = {sx1:.2f}, x₂ = {sx2:.2f}")
            print(f"  Z = {scipy_result['optimal_value']:.2f}")
        else:
            print(f"\nScipy: {scipy_result['message']}")
    
    print(f"\n{'='*60}")
    print("Все примеры завершены!")
    print('='*60)

if __name__ == "__main__":
    run_all_examples()
