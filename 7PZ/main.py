# =============================================================================
# Практическая работа №7. Простейшие вычисления в Python (аналог MATLAB)
# Тема: Основы работы в Python для математического моделирования
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Настройка отображения графиков в Jupyter / консоли
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

print("=== ЗАДАНИЕ 1. Базовые вычисления ===")
print(f"1) 25 + 18 × 3 = {25 + 18 * 3}")
print(f"2) (15 - 7) × (4 + 2) = {(15 - 7) * (4 + 2)}")
print(f"3) √144 + 5² = {np.sqrt(144) + 5**2}")
print(f"4) sin(π/3) + cos(π/6) = {np.sin(np.pi/3) + np.cos(np.pi/6):.4f}")
print(f"5) e² + ln(10) = {np.exp(2) + np.log(10):.4f}\n")

print("=== ЗАДАНИЕ 2. Работа с векторами ===")
v1 = np.arange(1, 21)                    # 1 до 20 шаг 1
v2 = np.arange(0, 10.5, 0.5)             # 0 до 10 шаг 0.5
v3 = np.arange(100, 49, -5)              # 100 до 50 шаг -5
v4 = np.linspace(0, 1, 10)               # 10 точек от 0 до 1

print(f"v1 (1:20): {v1}")
print(f"v2 (0:0.5:10): {v2}")
print(f"v3 (100:-5:50): {v3}")
print(f"v4 (linspace 0→1, 10 точек): {v4}\n")

print("=== ЗАДАНИЕ 3. Матричные операции ===")
A = np.array([[2, 1],
              [3, 4]])
B = np.array([[5, 2],
              [1, 6]])

print("Матрица A:\n", A)
print("Матрица B:\n", B)
print(f"A + B:\n{A + B}")
print(f"A × B (матричное умножение):\n{A @ B}")
print(f"Aᵀ (транспонирование):\n{A.T}")
print(f"A² (матрица в квадрате):\n{A @ A}\n")

print("=== ЗАДАНИЕ 4. Построение графиков ===")
x = np.arange(-2*np.pi, 2*np.pi + 0.1, 0.1)
y1 = np.sin(x)
y2 = np.cos(2*x)
y3 = x * np.sin(x)  # поэлементное умножение

plt.figure(figsize=(12, 5))
plt.plot(x, y1, 'r-', label='sin(x)', linewidth=2)
plt.plot(x, y2, 'b--', label='cos(2x)', linewidth=2)
plt.plot(x, y3, 'g-.', label='x·sin(x)', linewidth=2)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.title('Графики функций на интервале [-2π; 2π]', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()

# Подграфики (subplot) — как в MATLAB
print("Отображение подграфиков (4 функции в одном окне):")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
x_sub = np.linspace(0, 2*np.pi, 200)

axes[0, 0].plot(x_sub, np.sin(x_sub), 'r-', linewidth=1.5)
axes[0, 0].set_title('sin(x)'); axes[0, 0].grid(True); axes[0, 0].axhline(0, color='k', lw=0.3)

axes[0, 1].plot(x_sub, np.cos(x_sub), 'b--', linewidth=1.5)
axes[0, 1].set_title('cos(x)'); axes[0, 1].grid(True); axes[0, 1].axhline(0, color='k', lw=0.3)

axes[1, 0].plot(x_sub, np.sin(x_sub)*np.cos(x_sub), 'g:', linewidth=1.5)
axes[1, 0].set_title('sin(x)·cos(x)'); axes[1, 0].grid(True); axes[1, 0].axhline(0, color='k', lw=0.3)

axes[1, 1].plot(x_sub, np.sin(x_sub)**2, 'm-.', linewidth=1.5)
axes[1, 1].set_title('sin²(x)'); axes[1, 1].grid(True); axes[1, 1].axhline(0, color='k', lw=0.3)

plt.suptitle('Подграфики тригонометрических функций', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("=== ЗАДАНИЕ 5. Решение квадратного уравнения ===")
# Уравнение: 2x² - 3x + 1 = 0
coeff = [2, -3, 1]  # коэффициенты в порядке убывания степени
roots = np.roots(coeff)
print(f"Уравнение: 2x² - 3x + 1 = 0")
print(f"Коэффициенты: {coeff}")
print(f"Корни уравнения: {roots}")
print(f"Проверка: 2·({roots[0]:.3f})² - 3·({roots[0]:.3f}) + 1 = {2*roots[0]**2 - 3*roots[0] + 1:.2e}")
print(f"Проверка: 2·({roots[1]:.3f})² - 3·({roots[1]:.3f}) + 1 = {2*roots[1]**2 - 3*roots[1] + 1:.2e}\n")

print("=== ДОПОЛНИТЕЛЬНО: полезные команды ===")
print("help(np.sin)  # справка по функции")
print("np.__version__  # версия NumPy")
print("plt.style.available  # доступные стили графиков")
print("\n✅ Все задания выполнены успешно!")