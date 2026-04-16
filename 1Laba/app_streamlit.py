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
