"""
Resolución de sistemas lineales Ax = b mediante el método de Gauss-Jordan.
Escriba un programa para calcular la solución de un sistema de n ecuaciones con n incógnitas utilizando el método de Gauss-Jordan.

Considere sólo el caso de solución única, si el sistema no tiene solución única despliegue un mensaje.

Compruebe el funcionamiento verificando la solución de un sistema con n=5 (puede comprobar las soluciones usando la función integrada).

Compruebe el funcionamiento verificando la solución de un sistema con n=10 .

Incluya el archivo con el código así como el archivo con la captura de la ejecución del funcionamiento con los sistemas n=5 y n=10. !!!!!El profesor no va a capturar un sistema para verificar, debe hacerlo usted mismo!!!!!!!!
"""
import numpy as np

def gauss_jordan(A, b, tol=1e-12):
    """
    Resuelve Ax=b mediante Gauss-Jordan (forma reducida por filas).
    Devuelve x si existe solución única; de lo contrario lanza ValueError.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    n, m = A.shape
    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if b.shape != (n, 1):
        raise ValueError("El vector/matriz b debe ser de dimensión (n, 1).")

    # Matriz aumentada
    M = np.hstack([A, b]).astype(float)

    # Gauss-Jordan con pivoteo parcial
    for col in range(n):
        # Encontrar pivote con valor absoluto máximo por estabilidad
        piv_row = np.argmax(np.abs(M[col:, col])) + col
        piv_val = M[piv_row, col]

        # Si el pivote es ~0, no hay solución única (matriz singular)
        if abs(piv_val) < tol:
            raise ValueError("No hay solución única: pivote nulo (matriz singular).")

        # Intercambiar filas si es necesario
        if piv_row != col:
            M[[col, piv_row], :] = M[[piv_row, col], :]

        # Normalizar fila del pivote
        M[col, :] = M[col, :] / M[col, col]

        # Eliminar el resto de la columna
        for row in range(n):
            if row == col:
                continue
            factor = M[row, col]
            M[row, :] = M[row, :] - factor * M[col, :]

    # Tras la reducción, el lado izquierdo debe ser (aprox.) la identidad
    A_red = M[:, :n]
    b_red = M[:, n:]
    if not np.allclose(A_red, np.eye(n), atol=1e-9):
        # Redundante si el algoritmo fue correcto, pero lo chequeamos igualmente
        raise ValueError("No hay solución única: la matriz no se redujo a la identidad.")

    return b_red.reshape(-1)

def probar_sistema(A, b, titulo=""):
    print("="*80)
    if titulo:
        print(titulo)
        print("-"*80)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    n = A.shape[0]
    print(f"Dimensión n = {n}")
    print("A =")
    print(A)
    print("b =")
    print(b)

    try:
        x_gj = gauss_jordan(A, b)
        print("\nSolución por Gauss-Jordan x =")
        print(x_gj)
        # Verificación con función integrada
        x_np = np.linalg.solve(A, b)
        print("\nSolución con np.linalg.solve x* =")
        print(x_np)
        # Comprobación de cercanía
        err = np.linalg.norm(x_gj - x_np, ord=np.inf)
        print(f"\n||x - x*||_∞ = {err:.3e}")
    except Exception as e:
        print("\n[AVISO]", e)

def demo():
    np.set_printoptions(precision=4, suppress=True)

    # Reproducibilidad
    rng = np.random.default_rng(42)

    # Caso n=5 (aseguramos matriz invertible generando una aleatoria y sumando identidad escalada)
    n = 5
    A5 = rng.normal(size=(n, n))
    # Asegurar invertibilidad sumando un múltiplo de la identidad si hace falta
    A5 = A5 + 0.5 * np.eye(n)
    x_true_5 = rng.normal(size=n)
    b5 = A5 @ x_true_5

    probar_sistema(A5, b5, titulo="Prueba con n=5 (sistema con solución única)")

    # Caso n=10
    n = 10
    A10 = rng.normal(size=(n, n))
    A10 = A10 + 0.5 * np.eye(n)
    x_true_10 = rng.normal(size=n)
    b10 = A10 @ x_true_10

    probar_sistema(A10, b10, titulo="Prueba con n=10 (sistema con solución única)")

    # Caso sin solución única (opcional de demostración): matriz singular
    A_sing = np.array([[2., 4.],
                       [1., 2.]])
    b_sing = np.array([1., 0.5])
    probar_sistema(A_sing, b_sing, titulo="Demostración de caso sin solución única (matriz singular)")



demo()
