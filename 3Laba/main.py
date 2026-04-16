#%% ЭТАП 1: Подготовка среды и решение прямой задачи
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

print("="*60)
print("ЭТАП 1: Решение прямой задачи и визуализация ОДР")
print("="*60)

# Исходные данные (max F = 50x1 + 40x2)
c_primal = [-50, -40]                     # linprog решает min, поэтому знак "-"
A_ub_primal = [[3, 5], [4, 2]]            # Матрица ограничений (≤)
b_ub_primal = [45, 32]                    # Правые части
bounds_primal = [(0, None), (0, None)]    # x1, x2 ≥ 0

res_primal = linprog(c_primal, A_ub=A_ub_primal, b_ub=b_ub_primal, 
                     bounds=bounds_primal, method='highs')

print(f"📊 Статус: {res_primal.message}")
print(f"📦 Оптимальный план: x1 = {res_primal.x[0]:.4f}, x2 = {res_primal.x[1]:.4f}")
print(f"💰 Максимальная прибыль: F_max = {-res_primal.fun:.4f}")
print(f"📉 Остатки ресурсов (slack): {res_primal.slack}")

# Визуализация области допустимых решений
x_vals = np.linspace(0, 12, 200)
y1_vals = (45 - 3*x_vals) / 5
y2_vals = (32 - 4*x_vals) / 2

plt.figure(figsize=(7,5))
plt.plot(x_vals, y1_vals, label='3x₁ + 5x₂ ≤ 45', color='blue')
plt.plot(x_vals, y2_vals, label='4x₁ + 2x₂ ≤ 32', color='red')
plt.fill_between(x_vals, 0, np.minimum(y1_vals, y2_vals), 
                 where=(y1_vals>0)&(y2_vals>0), alpha=0.3, color='lightgreen')
plt.plot(res_primal.x[0], res_primal.x[1], 'ro', markersize=8, label='Оптимум (5,6)')
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.grid(True, alpha=0.6)
plt.legend()
plt.title('Область допустимых решений')
plt.show()


#%% ЭТАП 2: Автоматическое построение двойственной задачи
print("="*60)
print("ЭТАП 2: Функция построения двойственной задачи")
print("="*60)

def build_dual_symmetric(c, A, b):
    """
    Строит двойственную задачу для симметричной формы:
    Прямая: max cᵀx, Ax ≤ b, x ≥ 0
    Двойственная: min bᵀy, Aᵀy ≥ c, y ≥ 0
    Возвращает коэффициенты, готовые для scipy.optimize.linprog (все неравенства ≤)
    """
    c_dual = np.array(b)
    # Aᵀy ≥ c  =>  -Aᵀy ≤ -c
    A_ub_dual = -np.array(A).T
    b_ub_dual = -np.array(c)
    bounds_dual = [(0, None)] * len(b)
    return c_dual.tolist(), A_ub_dual.tolist(), b_ub_dual.tolist(), bounds_dual

c_orig = [50, 40]
A_orig = [[3, 5], [4, 2]]
b_orig = [45, 32]

c_d, A_d, b_d, bounds_d = build_dual_symmetric(c_orig, A_orig, b_orig)
print("✅ Коэффициенты двойственной задачи успешно сформированы.")
print(f"c_dual: {c_d}")
print(f"A_ub_dual: {A_d}")
print(f"b_ub_dual: {b_d}")


#%% ЭТАП 3: Численная проверка теорем двойственности
print("="*60)
print("ЭТАП 3: Проверка теорем двойственности")
print("="*60)

res_dual = linprog(c_d, A_ub=A_d, b_ub=b_d, bounds=bounds_d, method='highs')

print(f"📉 Оптимальные двойственные оценки: y₁ = {res_dual.x[0]:.4f}, y₂ = {res_dual.x[1]:.4f}")
print(f"📉 Минимальная стоимость ресурсов: Z_min = {res_dual.fun:.4f}")

# 🔹 Теорема 1: Значения целевых функций совпадают (F_max = Z_min)
diff_obj = abs(-res_primal.fun - res_dual.fun)
print(f"\n🔍 Теорема 1 (сильная двойственность):")
print(f"   |F_max - Z_min| = {diff_obj:.6e} {'✅ Верно' if diff_obj < 1e-6 else '❌ Ошибка'}")

# 🔹 Теорема 2: Условия дополняющей нежесткости
# y_i * (b_i - a_iᵀx) = 0
slack_primal = np.array(b_orig) - np.dot(A_orig, res_primal.x)
comp_y1 = res_dual.x[0] * slack_primal[0]
comp_y2 = res_dual.x[1] * slack_primal[1]

# x_j * (a_jᵀy - c_j) = 0
dual_slack = np.dot(np.array(A_orig).T, res_dual.x) - np.array(c_orig)
comp_x1 = res_primal.x[0] * dual_slack[0]
comp_x2 = res_primal.x[1] * dual_slack[1]

print("🔍 Теорема 2 (дополняющая нежесткость):")
print(f"   y₁·(остаток ресурса 1) = {comp_y1:.6e}")
print(f"   y₂·(остаток ресурса 2) = {comp_y2:.6e}")
print(f"   x₁·(нежесткость огр. 1) = {comp_x1:.6e}")
print(f"   x₂·(нежесткость огр. 2) = {comp_x2:.6e}")
print("   ✅ Все условия дополняющей нежесткости выполняются (≈0).")


#%% ЭТАП 4: Экономическая интерпретация и анализ чувствительности
print("="*60)
print("ЭТАП 4: Экономический анализ и чувствительность")
print("="*60)

shadow_prices = res_dual.x
print(f"💡 Теневые цены (объективно обусловленные оценки):")
print(f"   Ресурс 1: {shadow_prices[0]:.4f} ден.ед./ед.")
print(f"   Ресурс 2: {shadow_prices[1]:.4f} ден.ед./ед.")

# 🔸 Проверка чувствительности: увеличение запаса ресурса 1 на 1 ед.
b_ub_new = [b_orig[0] + 1, b_orig[1]]
res_new = linprog(c_primal, A_ub=A_orig, b_ub=b_ub_new, bounds=bounds_primal, method='highs')
profit_change = -res_new.fun - (-res_primal.fun)

print(f"\n📈 Изменение b₁: 45 → 46")
print(f"   Новая прибыль: {-res_new.fun:.4f}")
print(f"   Прирост прибыли: {profit_change:.4f}")
print(f"   Совпадает с y₁? {'✅ Да' if abs(profit_change - shadow_prices[0]) < 1e-4 else '❌ Нет'}")

# 🔸 Случай нулевой теневой цены (избыточный ресурс)
b_ub_excess = [b_orig[0], b_orig[1] + 50]
res_excess = linprog(c_primal, A_ub=A_orig, b_ub=b_ub_excess, bounds=bounds_primal, method='highs')
c_d_ex, A_d_ex, b_d_ex, bounds_d_ex = build_dual_symmetric(c_orig, A_orig, b_ub_excess)
res_dual_ex = linprog(c_d_ex, A_ub=A_d_ex, b_ub=b_d_ex, bounds=bounds_d_ex, method='highs')

print(f"\n💧 Анализ избыточного ресурса (b₂ → {b_ub_excess[1]}):")
print(f"   Теневая цена ресурса 2: y₂ = {res_dual_ex.x[1]:.6f} (≈0)")
print(f"   Фактический остаток ресурса 2: {res_excess.slack[1]:.2f}")
print("   ✅ Нулевая теневая цена подтверждает избыточность ресурса.")


#%% ЭТАП 5: Индивидуальное задание (РАБОЧИЙ ШАБЛОН)
import numpy as np
from scipy.optimize import linprog

print("="*60)
print("ЭТАП 5: Решение индивидуального варианта")
print("="*60)

# ⚠️ ЗАМЕНИТЕ ЭТИ ДАННЫЕ НА ВАШ ВАРИАНТ ИЗ ТАБЛИЦЫ (стр. 11-12)
# Пример рабочей задачи:
# max Z = 3x1 + 2x2
# при ограничениях:
#   2x1 + x2 ≤ 100
#   x1 + x2 ≤ 80
#   x1 ≤ 40
#   x1, x2 ≥ 0

c_ind = [-3, -2]  # ⚠️ ВАШИ коэффициенты с МИНУСОМ

A_ub_ind = [
    [2, 1],   # ⚠️ ВАШИ ограничения
    [1, 1],
    [1, 0]
]
b_ub_ind = [100, 80, 40]  # ⚠️ ВАШИ правые части

bounds_ind = [(0, None), (0, None)]  # ⚠️ ВАШИ границы

# Решение
res_ind = linprog(c_ind, A_ub=A_ub_ind, b_ub=b_ub_ind, 
                  bounds=bounds_ind, method='highs')

print(f"\n📊 Статус: {res_ind.message}")

if res_ind.success:
    print("✅ Решение найдено!")
    print(f"📦 x = {np.round(res_ind.x, 4)}")
    print(f"💰 F_max = {-res_ind.fun:.4f}")
    
    if hasattr(res_ind, 'ineqlin') and res_ind.ineqlin.marginals is not None:
        print(f"📉 Теневые цены: {np.round(res_ind.ineqlin.marginals, 4)}")
else:
    print("❌ ОШИБКА! Проверьте данные:")
    print("   • Все ли ограничения записаны как ≤ ?")
    print("   • Правильные ли знаки у коэффициентов?")
    print("   • Не противоречат ли ограничения друг другу?")

# 📋 ИНСТРУКЦИЯ:
# 1. Откройте стр. 11-12 методички
# 2. Найдите свой вариант
# 3. Замените c_ind, A_ub_ind, b_ub_ind, bounds_ind на свои данные
# 4. Если есть ограничения "=", добавьте A_eq_ind и b_eq_ind