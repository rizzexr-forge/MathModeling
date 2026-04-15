1. 🔢 Математические формулы и их реализация в Python
1.1 Целевая функция (Objective Function)
Формула:
Z
=
c
1
x
1
+
c
2
x
2
Z=c 
1
​
 x 
1
​
 +c 
2
​
 x 
2
​
 

Python реализация:

python
def evaluate_objective(self, x1, x2):
    return self.c1 * x1 + self.c2 * x2

# Пример для максимизации: Z = 3x₁ + 2x₂
# c1 = 3, c2 = 2
# Z = 3*x1 + 2*x2
1.2 Линейное ограничение (Linear Constraint)
Формула:
a
i
1
x
1
+
a
i
2
x
2
≤
b
i
a 
i1
​
 x 
1
​
 +a 
i2
​
 x 
2
​
 ≤b 
i
​
 

Python реализация:

python
class Constraint:
    def __init__(self, a, b, c, constraint_type='<='):
        self.a = a      # коэффициент при x₁
        self.b = b      # коэффициент при x₂
        self.c = c      # правая часть
        self.type = constraint_type  # '<=', '>=', '='
    
    def evaluate(self, x1, x2):
        return self.a * x1 + self.b * x2

# Пример: 2x₁ + x₂ ≤ 100
# constraint = Constraint(2, 1, 100, '<=')
1.3 Условия неотрицательности (Non-negativity constraints)
Формула:
x
1
≥
0
,
x
2
≥
0
x 
1
​
 ≥0,x 
2
​
 ≥0

Python реализация:

python
# Автоматически добавляются в задачу
self.constraints.append(Constraint(1, 0, 0, '>='))  # x₁ >= 0
self.constraints.append(Constraint(0, 1, 0, '>='))  # x₂ >= 0
1.4 Линия уровня (Level line / Isoquant)
Формула:
c
1
x
1
+
c
2
x
2
=
L
,
L
=
const
c 
1
​
 x 
1
​
 +c 
2
​
 x 
2
​
 =L,L=const

Выражение x₂ через x₁:
x
2
=
L
−
c
1
x
1
c
2
,
c
2
≠
0
x 
2
​
 = 
c 
2
​
 
L−c 
1
​
 x 
1
​
 
​
 ,c 
2
​
 

=0

Python реализация:

python
if abs(self.problem.c2) > 1e-9:
    x1_vals = np.linspace(0, x_max, 100)
    x2_vals = (L - self.problem.c1 * x1_vals) / self.problem.c2
1.5 Градиент целевой функции (Gradient)
Формула:
∇
Z
=
(
∂
Z
∂
x
1
∂
Z
∂
x
2
)
=
(
c
1
c
2
)
∇Z=( 
∂x 
1
​
 
∂Z
​
 
∂x 
2
​
 
∂Z
​
 
​
 )=( 
c 
1
​
 
c 
2
​
 
​
 )

Свойства:

Градиент показывает направление наискорейшего роста функции

Градиент перпендикулярен линиям уровня

Python:

python
gradient = (self.c1, self.c2)  # направление роста
# Для максимизации: двигаемся вдоль градиента
# Для минимизации: двигаемся против градиента
1.6 Угол наклона линии уровня (Slope angle)
Формула:
tan
⁡
(
α
)
=
−
c
1
c
2
tan(α)=− 
c 
2
​
 
c 
1
​
 
​
 

Python:

python
import math
if c2 != 0:
    alpha = math.atan2(-c1, c2)  # угол в радианах
    slope = -c1 / c2              # угловой коэффициент
1.7 Пересечение двух прямых (Line intersection)
Система уравнений:
{
a
1
x
1
+
b
1
x
2
=
c
1
a
2
x
1
+
b
2
x
2
=
c
2
{ 
a 
1
​
 x 
1
​
 +b 
1
​
 x 
2
​
 =c 
1
​
 
a 
2
​
 x 
1
​
 +b 
2
​
 x 
2
​
 =c 
2
​
 
​
 

Решение (метод Крамера):
Δ
=
a
1
b
2
−
a
2
b
1
Δ=a 
1
​
 b 
2
​
 −a 
2
​
 b 
1
​
 

x
1
=
c
1
b
2
−
c
2
b
1
Δ
x 
1
​
 = 
Δ
c 
1
​
 b 
2
​
 −c 
2
​
 b 
1
​
 
​
 

x
2
=
a
1
c
2
−
a
2
c
1
Δ
x 
2
​
 = 
Δ
a 
1
​
 c 
2
​
 −a 
2
​
 c 
1
​
 
​
 

Python реализация:

python
def _intersect_constraints(self, c1, c2):
    det = c1.a * c2.b - c2.a * c1.b
    
    if abs(det) < 1e-9:  # прямые параллельны
        return None
    
    x1 = (c1.c * c2.b - c2.c * c1.b) / det
    x2 = (c1.a * c2.c - c2.a * c1.c) / det
    
    return (x1, x2)
1.8 Проверка допустимости точки (Feasibility check)
Условие:
Точка $(x_1, x_2)$ допустима, если выполняются ВСЕ ограничения

Python:

python
def is_feasible_point(self, x1, x2, tol=1e-6):
    for constraint in self.constraints:
        if constraint.type == '<=':
            if constraint.evaluate(x1, x2) > constraint.c + tol:
                return False
        else:  # '>='
            if constraint.evaluate(x1, x2) < constraint.c - tol:
                return False
    return True
1.9 Формула для поиска оптимальной точки
Для максимизации:
(
x
1
∗
,
x
2
∗
)
=
arg
⁡
max
⁡
(
x
1
,
x
2
)
∈
ОДР
Z
(
x
1
,
x
2
)
(x 
1
∗
​
 ,x 
2
∗
​
 )=argmax 
(x 
1
​
 ,x 
2
​
 )∈ОДР
​
 Z(x 
1
​
 ,x 
2
​
 )

Для минимизации:
(
x
1
∗
,
x
2
∗
)
=
arg
⁡
min
⁡
(
x
1
,
x
2
)
∈
ОДР
Z
(
x
1
,
x
2
)
(x 
1
∗
​
 ,x 
2
∗
​
 )=argmin 
(x 
1
​
 ,x 
2
​
 )∈ОДР
​
 Z(x 
1
​
 ,x 
2
​
 )

Python:

python
if self.objective == 'max':
    optimal_idx = np.argmax(values)  # индекс вершины с max Z
else:
    optimal_idx = np.argmin(values)  # индекс вершины с min Z
1.10 Теневая цена (Shadow price)
Определение: Показывает, насколько изменится Z при увеличении ресурса на 1 единицу

Формула:
y
i
=
∂
Z
∂
b
i
y 
i
​
 = 
∂b 
i
​
 
∂Z
​
 

Python (активные ограничения):

python
# Активное ограничение (дефицитный ресурс)
if abs(value - constraint.c) < 1e-6:
    status = "🔴 Активное (дефицитное)"
    # Теневая цена > 0 для максимизации
else:
    status = "🟢 Неактивное"  
    # Теневая цена = 0
2. 📐 Геометрические понятия
2.1 Область допустимых решений (Feasible Region - ОДР)
Тип ОДР	Условие	Пример
Ограниченный многоугольник	ОДР замкнута и ограничена	x₁ + x₂ ≤ 10, x₁ ≥ 0, x₂ ≥ 0
Неограниченная область	ОДР уходит в бесконечность	x₁ - x₂ ≤ 10
Пустая область	Нет точек, удовлетворяющих всем ограничениям	x₁ + x₂ ≤ 10, x₁ + x₂ ≥ 20
Python определение:

python
vertices = self.find_feasible_region_vertices()
if not vertices:
    status = 'infeasible'  # пустая область
2.2 Типы оптимальных решений
Случай	Описание	Признак
Единственное решение	Одна вершина дает max/min	ЦФ не параллельна ни одной стороне
Альтернативные оптимумы	Любая точка на стороне оптимальна	ЦФ параллельна стороне ОДР
Неограниченное решение	Z → ∞ (или -∞)	ОДР неограничена в направлении улучшения
Нет решения	ОДР = ∅	Противоречивые ограничения
3. 🐍 Ключевые функции Python для демонстрации
3.1 Основной класс задачи
python
class LinearProgrammingProblem:
    def __init__(self, c1, c2, objective='max'):
        self.c1 = c1          # коэффициент при x1 в ЦФ
        self.c2 = c2          # коэффициент при x2 в ЦФ
        self.objective = objective  # 'max' или 'min'
        self.constraints = []       # список ограничений
3.2 Нахождение вершин ОДР
python
def find_feasible_region_vertices(self):
    vertices = []
    n = len(self.constraints)
    
    # Перебираем все пары ограничений
    for i in range(n):
        for j in range(i + 1, n):
            # Находим пересечение
            point = self._intersect_constraints(
                self.constraints[i], 
                self.constraints[j]
            )
            # Проверяем, что точка допустима
            if point and self.is_feasible_point(*point):
                vertices.append(point)
    
    return unique_vertices
3.3 Построение графика
python
def plot_solution(self):
    # 1. Создаем фигуру
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 2. Строим ОДР (полигон)
    polygon = Polygon(vertices, alpha=0.3, color='lightblue')
    
    # 3. Строим линии ограничений
    for constraint in constraints:
        x_vals, y_vals = constraint.get_line_points()
        ax.plot(x_vals, y_vals)
    
    # 4. Строим линии уровня ЦФ
    # Z = const → x2 = (const - c1*x1)/c2
    
    # 5. Отмечаем оптимальную точку
    ax.plot(x_opt, y_opt, 'r*', markersize=15)
3.4 Решение через scipy (для проверки)
python
from scipy.optimize import linprog

# Для максимизации: минимизируем -Z
c = [-c1, -c2]

# Ограничения в виде A_ub @ x <= b_ub
A_ub = [[a11, a12], [a21, a22]]
b_ub = [b1, b2]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)])
4. 📊 Основные типы задач
4.1 Задача максимизации (Maximization)
Постановка:

text
Maximize: Z = 3x₁ + 2x₂
Constraints:
    2x₁ + x₂ ≤ 100
    x₁ + x₂ ≤ 80
    x₁ ≤ 40
    x₁, x₂ ≥ 0
4.2 Задача минимизации (Minimization)
Постановка:

text
Minimize: Z = 2x₁ + 5x₂
Constraints:
    3x₁ + 2x₂ ≥ 60
    x₁ + 2x₂ ≥ 40
    x₁, x₂ ≥ 0
4.3 Задача с альтернативными оптимумами
Условие: Целевая функция параллельна стороне ОДР

text
Maximize: Z = x₁ + x₂  (коэффициенты 1 и 1)
Constraint: x₁ + x₂ ≤ 80 (параллельна!)
5. 🎯 Ключевые термины (для устного ответа)
Русский	English	Определение
Целевая функция	Objective function	Функция, которую оптимизируем (max/min)
Ограничения	Constraints	Условия, ограничивающие решения
ОДР	Feasible region	Множество всех допустимых решений
Линия уровня	Level line / Isoquant	Линия, где Z = const
Градиент	Gradient	Вектор наискорейшего роста функции
Оптимальная точка	Optimal point	Точка, где Z достигает max/min
Базовая переменная	Basic variable	Переменная, входящая в базис
Теневая цена	Shadow price	Изменение Z при изменении ресурса
Активное ограничение	Active constraint	Ограничение, выполняющееся как равенство
Симплекс-метод	Simplex method	Алгоритм решения задач ЛП
6. 🔄 Алгоритм графического метода (шаги)
text
Шаг 1: Построить оси координат (x₁, x₂)
        ↓
Шаг 2: Для каждого ограничения:
        - Преобразовать в уравнение прямой
        - Построить прямую
        - Определить нужную полуплоскость
        ↓
Шаг 3: Найти ОДР = пересечение всех полуплоскостей
        ↓
Шаг 4: Построить линию уровня Z = const
        ↓
Шаг 5: Перемещать линию уровня в направлении улучшения Z
        ↓
Шаг 6: Найти последнюю точку ОДР, которую пересекает линия уровня
        ↓
Шаг 7: Вычислить Z в этой точке
7. 💻 Полный рабочий пример
python
# 1. Импорт библиотек
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# 2. Данные задачи
c1, c2 = 3, 2          # коэффициенты ЦФ
objective = 'max'      # тип задачи

# 3. Ограничения: A_ub @ x <= b_ub
A_ub = [[2, 1], [1, 1], [1, 0]]
b_ub = [100, 80, 40]

# 4. Решение через scipy
c = [-c1, -c2]  # для максимизации
result = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                 bounds=[(0, None), (0, None)])

print(f"Оптимальное решение: x1 = {result.x[0]:.2f}, x2 = {result.x[1]:.2f}")
print(f"Z* = {-result.fun:.2f}")

# 5. Визуализация
x1 = np.linspace(0, 100, 100)
x2_1 = (100 - 2*x1) / 1   # 2x1 + x2 = 100
x2_2 = (80 - x1) / 1       # x1 + x2 = 80
x2_3 = 40 * np.ones_like(x1)  # x1 = 40

plt.figure(figsize=(10, 8))
plt.plot(x1, x2_1, 'r-', label='2x₁ + x₂ = 100')
plt.plot(x1, x2_2, 'g-', label='x₁ + x₂ = 80')
plt.axvline(x=40, color='orange', label='x₁ = 40')
plt.fill_between(x1, 0, np.minimum(x2_1, x2_2), 
                  where=(x1 <= 40), alpha=0.3)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Графическое решение задачи ЛП')
plt.show()
8. ❓ Часто задаваемые вопросы на зачете
Вопрос: Почему оптимальное решение всегда в вершине ОДР?
Ответ: Целевая функция линейна. При перемещении линии уровня, первое и последнее касание с выпуклым многоугольником ОДР происходит в вершине. Если линия уровня параллельна стороне, то оптимум достигается на всей стороне (включая вершины).

Вопрос: Как определить, что задача имеет альтернативные оптимумы?
Ответ: Если линия уровня параллельна одной из сторон ОДР и касается её при оптимальном значении, то любая точка на этой стороне дает одинаковое значение Z.

Python проверка:

python
# Проверка параллельности стороны и линии уровня
# Сторона: a*x1 + b*x2 = c
# Линия уровня: c1*x1 + c2*x2 = L
# Параллельны, если (c1, c2) пропорциональны (a, b)
if abs(c1/a - c2/b) < 1e-6:
    print("Возможны альтернативные оптимумы")
Вопрос: Что такое теневая цена и как её найти?
Ответ: Теневая цена - это производная Z по правой части ограничения. Для активных ограничений показывает, насколько увеличится Z при увеличении ресурса на 1.

Вопрос: В чем отличие графического метода от симплекс-метода?
Ответ: Графический метод работает только для 2 переменных, использует визуализацию. Симплекс-метод работает для любого количества переменных, алгебраический, перебирает вершины многогранника.

9. 📝 Памятка: Что нужно показать на зачете
Объяснить постановку задачи:

Что такое Целевая функция

Что такое ограничения

Условия неотрицательности

Показать построение:

Как строить прямую по уравнению

Как определить полуплоскость

Как найти ОДР

Продемонстрировать код:

Класс Constraint

Метод find_feasible_region_vertices

Функцию linprog из scipy

Проанализировать результат:

Оптимальная точка

Активные/неактивные ограничения

Анализ чувствительности

Описать особые случаи:

Неограниченное решение

Недопустимая задача

Альтернативные оптимумы

10. 🚀 Быстрый запуск для демонстрации
bash
# 1. Установка библиотек
pip install numpy matplotlib scipy pandas streamlit

# 2. Запуск теста
python -c "from linear_programming import test_problem; test_problem()"

# 3. Запуск веб-интерфейса
streamlit run app_streamlit.py

# 4. Запуск примеров
python examples.py
