import numpy as np

def sor(A, b, omega, tolerancia=1e-8, max_iter=1000, x0=None):
    """
    Resuelve Ax=b con el método de Sobre-relajación Sucesiva (SOR).
    Entradas:
      - A: matriz de coeficientes.
      - b: vector de términos independientes.
      - omega: factor de relajación (ω).
      - tolerancia: criterio de paro relativo en norma infinito.
      - max_iter: máximo número de iteraciones permitidas.
      - x0: aproximación inicial (opcional). Si no se da, se usa el vector cero.
    Salidas:
      - x: aproximación a la solución.
      - k: número de iteraciones realizadas.
    Si el método no converge en 'max_iter', lanza un ValueError con un mensaje claro.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    n, m = A.shape
    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if b.size != n:
        raise ValueError("Dimensiones incompatibles entre A y b.")
    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("SOR no es aplicable: existen ceros en la diagonal de A.")
    
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).flatten()
    
    for k in range(1, max_iter + 1):
        x_prev = x.copy()
        # Barrido de SOR (combina Gauss-Seidel con relajación)
        for i in range(n):
            suma1 = np.dot(A[i, :i], x[:i])
            suma2 = np.dot(A[i, i+1:], x_prev[i+1:])
            x_gs = (b[i] - suma1 - suma2) / A[i, i]
            x[i] = (1 - omega) * x_prev[i] + omega * x_gs
        
        # Criterio de finalización: ||x - x_prev||_inf / ||x||_inf < tolerancia
        num = np.linalg.norm(x - x_prev, ord=np.inf)
        den = np.linalg.norm(x, ord=np.inf)
        rel = num / den if den > 0 else num
        
        if rel < tolerancia:
            return x, k
    
    raise ValueError(f"SOR no alcanzó la tolerancia en {max_iter} iteraciones.")

def demo_sor():
    """
    Demostración del método de Sobre-relajación Sucesiva (SOR) con:
      - Un sistema que SÍ converge con diferentes valores de omega.
      - Un sistema que NO converge (matriz no definida positiva + omega inadecuado).
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=100)
    
    print("\n" + "="*80)
    print("DEMOSTRACIÓN: MÉTODO DE SOBRE-RELAJACIÓN SUCESIVA (SOR)")
    print("="*80)
    
    # --- Caso 1: CONVERGENTE (diagonal dominante por filas) ---
    A1 = np.array([
        [10, -1,  2,  0],
        [-1, 11, -1,  3],
        [ 2, -1, 10, -1],
        [ 0,  3, -1,  8]
    ], dtype=float)
    b1 = np.array([6, 25, -11, 15], dtype=float)
    
    print("\n" + "-"*80)
    print("CASO 1: CONVERGENTE (Matriz diagonal dominante)")
    print("-"*80)
    print("\nMatriz A:")
    print(A1)
    print("\nVector b:")
    print(b1)
    print("\nSolución exacta (usando np.linalg.solve):")
    sol_exacta = np.linalg.solve(A1, b1)
    print(sol_exacta)
    
    # Probamos con diferentes valores de omega
    omegas = [1.0, 1.1, 1.3, 1.5]
    for omega in omegas:
        try:
            x, k = sor(A1, b1, omega=omega, tolerancia=1e-10, max_iter=500)
            error = np.linalg.norm(x - sol_exacta, ord=np.inf)
            print(f"\nSOR (ω={omega}): converge en {k} iteraciones")
            print(f"  Solución: {x}")
            print(f"  Error vs solución exacta: {error:.2e}")
        except ValueError as e:
            print(f"\nSOR (ω={omega}): {e}")
    
    # --- Caso 2: NO CONVERGENTE (matriz con valores propios problemáticos) ---
    A2 = np.array([
        [ 1,  5],
        [ 5,  1]
    ], dtype=float)
    b2 = np.array([6, 6], dtype=float)
    
    print("\n" + "-"*80)
    print("CASO 2: NO CONVERGENTE (Matriz simétrica no definida positiva)")
    print("-"*80)
    print("\nMatriz A:")
    print(A2)
    print("\nVector b:")
    print(b2)
    print("\nSolución exacta (usando np.linalg.solve):")
    sol_exacta2 = np.linalg.solve(A2, b2)
    print(sol_exacta2)
    print("\nValores propios de A:")
    eigenvalues = np.linalg.eigvals(A2)
    print(eigenvalues)
    print("(Nota: valores propios de signo diferente indican que NO es definida positiva)")
    
    # Probamos con omega > 1 (sobre-relajación) que causa divergencia
    omegas_divergentes = [1.5, 1.8, 1.9]
    for omega in omegas_divergentes:
        try:
            x, k = sor(A2, b2, omega=omega, tolerancia=1e-10, max_iter=100)
            error = np.linalg.norm(x - sol_exacta2, ord=np.inf)
            print(f"\nSOR (ω={omega}): converge en {k} iteraciones")
            print(f"  Solución: {x}")
            print(f"  Error vs solución exacta: {error:.2e}")
        except ValueError as e:
            print(f"\nSOR (ω={omega}): DIVERGE - {e}")
    
    # --- Caso 3: OTRO NO CONVERGENTE (matriz no simétrica con diagonal débil) ---
    A3 = np.array([
        [ 1,  10],
        [ 10, 1]
    ], dtype=float)
    b3 = np.array([11, 11], dtype=float)
    
    print("\n" + "-"*80)
    print("CASO 3: NO CONVERGENTE (Matriz sin diagonal dominante)")
    print("-"*80)
    print("\nMatriz A:")
    print(A3)
    print("\nVector b:")
    print(b3)
    print("\nSolución exacta:")
    sol_exacta3 = np.linalg.solve(A3, b3)
    print(sol_exacta3)
    
    omega = 1.2
    try:
        x, k = sor(A3, b3, omega=omega, tolerancia=1e-10, max_iter=100)
        error = np.linalg.norm(x - sol_exacta3, ord=np.inf)
        print(f"\nSOR (ω={omega}): converge en {k} iteraciones")
        print(f"  Solución: {x}")
        print(f"  Error vs solución exacta: {error:.2e}")
    except ValueError as e:
        print(f"\nSOR (ω={omega}): DIVERGE - {e}")

def demo():
    demo_sor()

demo()