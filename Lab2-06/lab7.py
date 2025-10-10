import numpy as np

def jacobi(A, b, tolerancia=1e-8, max_iter=1000, x0=None):
    """
    Resuelve Ax=b con el método de Jacobi.
    Entradas:
      - A: matriz de coeficientes.
      - b: vector de términos independientes.
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
        raise ValueError("Jacobi no es aplicable: existen ceros en la diagonal de A.")

    # Separación D y R (A = D + R)
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.diag(1.0 / np.diag(D))

    # Inicialización
    x_prev = np.zeros(n) if x0 is None else np.array(x0, dtype=float).flatten()

    for k in range(1, max_iter + 1):
        x = D_inv @ (b - R @ x_prev)

        # Criterio de finalización: ||x - x_prev||_inf / ||x||_inf < tolerancia
        num = np.linalg.norm(x - x_prev, ord=np.inf)
        den = np.linalg.norm(x, ord=np.inf)
        rel = num / den if den > 0 else num
        if rel < tolerancia:
            return x, k

        x_prev = x.copy()

    # Si llegó aquí, no cumplió el criterio dentro de max_iter
    raise ValueError(f"Jacobi no alcanzó la tolerancia en {max_iter} iteraciones.")


def gauss_seidel(A, b, tolerancia=1e-8, max_iter=1000, x0=None):
    """
    Resuelve Ax=b con el método de Gauss-Seidel.
    Entradas y salidas análogas a 'jacobi'. Usa el mismo criterio de paro relativo.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    n, m = A.shape

    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if b.size != n:
        raise ValueError("Dimensiones incompatibles entre A y b.")
    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("Gauss-Seidel no es aplicable: existen ceros en la diagonal de A.")

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float).flatten()

    for k in range(1, max_iter + 1):
        x_prev = x.copy()

        # Barrido de Gauss-Seidel (usa valores nuevos al instante)
        for i in range(n):
            suma1 = np.dot(A[i, :i], x[:i])
            suma2 = np.dot(A[i, i+1:], x_prev[i+1:])
            x[i] = (b[i] - suma1 - suma2) / A[i, i]

        # Criterio de finalización: ||x - x_prev||_inf / ||x||_inf < tolerancia
        num = np.linalg.norm(x - x_prev, ord=np.inf)
        den = np.linalg.norm(x, ord=np.inf)
        rel = num / den if den > 0 else num
        if rel < tolerancia:
            return x, k

    raise ValueError(f"Gauss-Seidel no alcanzó la tolerancia en {max_iter} iteraciones.")


# ============================== DEMOSTRACIÓN: JACOBI ==============================

def demo_jacobi():
    """
    Demostración del método de Jacobi con:
      - Un sistema que SÍ converge (diagonal dominante).
      - Un sistema que NO logra la solución (no converge dentro de max_iter).
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=100)

    print("\n" + "="*80)
    print("DEMOSTRACIÓN: MÉTODO DE JACOBI")
    print("="*80)

    # --- Caso 1: Convergente (diagonal dominante por filas) ---
    A1 = np.array([
        [10, -1,  2,  0],
        [-1, 11, -1,  3],
        [ 2, -1, 10, -1],
        [ 0,  3, -1,  8]
    ], dtype=float)
    b1 = np.array([6, 25, -11, 15], dtype=float)

    print("\nCASO CONVERGENTE")
    print("Matriz A:")
    print(A1)
    print("\nVector b:")
    print(b1)

    try:
        x, k = jacobi(A1, b1, tolerancia=1e-10, max_iter=500)
        print(f"\nJacobi: solución aproximada (en {k} iteraciones):")
        print(x)
    except ValueError as e:
        print(f"\nJacobi: {e}")

    # --- Caso 2: No convergente (no dominante; iteración diverge) ---
    # Sistema: A x = b con A que NO es adecuada para Jacobi
    A2 = np.array([
        [1, 2],
        [2, 1]
    ], dtype=float)
    b2 = np.array([3, 3], dtype=float)

    print("\nCASO NO CONVERGENTE")
    print("Matriz A:")
    print(A2)
    print("\nVector b:")
    print(b2)

    try:
        x, k = jacobi(A2, b2, tolerancia=1e-10, max_iter=50)
        print(f"\nJacobi: solución aproximada (en {k} iteraciones):")
        print(x)
    except ValueError as e:
        print(f"\nJacobi: no se pudo obtener la solución con el criterio dado. {e}")


# =========================== DEMOSTRACIÓN: GAUSS-SEIDEL ==========================

def demo_gauss_seidel():
    """
    Demostración del método de Gauss-Seidel con:
      - Un sistema que SÍ converge (diagonal dominante).
      - Un sistema que NO logra la solución (no converge dentro de max_iter).
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=100)

    print("\n" + "="*80)
    print("DEMOSTRACIÓN: MÉTODO DE GAUSS-SEIDEL")
    print("="*80)

    # --- Caso 1: Convergente (diagonal dominante por filas) ---
    A1 = np.array([
        [10, -1,  2,  0],
        [-1, 11, -1,  3],
        [ 2, -1, 10, -1],
        [ 0,  3, -1,  8]
    ], dtype=float)
    b1 = np.array([6, 25, -11, 15], dtype=float)

    print("\nCASO CONVERGENTE")
    print("Matriz A:")
    print(A1)
    print("\nVector b:")
    print(b1)

    try:
        x, k = gauss_seidel(A1, b1, tolerancia=1e-10, max_iter=500)
        print(f"\nGauss-Seidel: solución aproximada (en {k} iteraciones):")
        print(x)
    except ValueError as e:
        print(f"\nGauss-Seidel: {e}")

    # --- Caso 2: No convergente ---
    # Elegimos una matriz que, en este método, tiende a divergir.
    A2 = np.array([
        [1, 2],
        [2, 1]
    ], dtype=float)
    b2 = np.array([3, 3], dtype=float)

    print("\nCASO NO CONVERGENTE")
    print("Matriz A:")
    print(A2)
    print("\nVector b:")
    print(b2)

    try:
        x, k = gauss_seidel(A2, b2, tolerancia=1e-10, max_iter=50)
        print(f"\nGauss-Seidel: solución aproximada (en {k} iteraciones):")
        print(x)
    except ValueError as e:
        print(f"\nGauss-Seidel: no se pudo obtener la solución con el criterio dado. {e}")


# ============================== EJECUCIÓN DE DEMOS ==============================

def demo():
    demo_jacobi()
    demo_gauss_seidel()

demo()
