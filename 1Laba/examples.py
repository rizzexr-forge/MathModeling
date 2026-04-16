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
