import numpy as np

def gauss_pivoteo_parcial(A, b, tol=1e-12):
    """
    Resuelve Ax = b mediante eliminación gaussiana con pivoteo parcial.
    En cada paso selecciona el término con mayor valor absoluto como pivote.

    Entradas:
      - A: matriz cuadrada de dimensión n×n.
      - b: vector de términos independientes de dimensión n o (n,1).
      - tol: tolerancia para detectar pivotes nulos (matriz singular).

    Salida:
      - x: vector solución de dimensión n.

    Lanza ValueError si la matriz es singular o no hay solución única.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    n, m = A.shape
    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if b.shape != (n, 1):
        raise ValueError("El vector b debe tener dimensión compatible con A.")

    # Matriz aumentada [A | b]
    M = np.hstack([A, b]).astype(float)

    # Eliminación hacia adelante con pivoteo parcial
    for col in range(n):
        # Buscar el mayor pivote en valor absoluto a partir de la fila 'col'
        piv_row = np.argmax(np.abs(M[col:, col])) + col
        piv_val = M[piv_row, col]

        if abs(piv_val) < tol:
            raise ValueError("No hay solución única: pivote casi nulo (matriz singular).")

        # Intercambiar filas si es necesario
        if piv_row != col:
            M[[col, piv_row], :] = M[[piv_row, col], :]

        # Eliminar en las filas inferiores
        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, :] = M[row, :] - factor * M[col, :]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = M[i, n]  # término independiente
        for j in range(i + 1, n):
            suma -= M[i, j] * x[j]
        if abs(M[i, i]) < tol:
            raise ValueError("No hay solución única en sustitución hacia atrás (pivote nulo).")
        x[i] = suma / M[i, i]

    return x


def F_sistema(x):
    """
    Evalúa el sistema no lineal:
        f1(x1, x2) = x1^2 + 4 x2^2 - 4
        f2(x1, x2) = 4 x1^2 + x2^2 - 4

    Entrada:
      - x: vector (x1, x2).

    Salida:
      - F(x): vector [f1, f2].
    """
    x1, x2 = x
    f1 = x1**2 + 4.0 * x2**2 - 4.0
    f2 = 4.0 * x1**2 + x2**2 - 4.0
    return np.array([f1, f2], dtype=float)


def J_sistema(x):
    """
    Jacobiano del sistema:
        f1 = x1^2 + 4 x2^2 - 4
        f2 = 4 x1^2 + x2^2 - 4

    Entonces:
        df1/dx1 = 2 x1        df1/dx2 = 8 x2
        df2/dx1 = 8 x1        df2/dx2 = 2 x2

    Entrada:
      - x: vector (x1, x2).

    Salida:
      - J(x): matriz 2x2 con las derivadas parciales.
    """
    x1, x2 = x
    j11 = 2.0 * x1
    j12 = 8.0 * x2
    j21 = 8.0 * x1
    j22 = 2.0 * x2
    return np.array([[j11, j12],
                     [j21, j22]], dtype=float)


def newton_no_lineal(F, J, x0, tolerancia=1e-8, max_iter=50):
    """
    Método de Newton para sistemas de ecuaciones no lineales F(x) = 0.

    En cada iteración se resuelve:
        J(x^(k)) * y^(k) = -F(x^(k))
    usando el método de Gauss con pivoteo parcial (sin usar la inversa de J),
    y se actualiza:
        x^(k+1) = x^(k) + y^(k)

    Entradas:
      - F: función que recibe x y devuelve F(x).
      - J: función que recibe x y devuelve la matriz jacobiana J(x).
      - x0: vector inicial.
      - tolerancia: criterio de paro relativo en norma infinito.
      - max_iter: máximo número de iteraciones permitidas.

    Salidas:
      - x: aproximación a la solución.
      - k: número de iteraciones realizadas.

    Lanza ValueError si no se alcanza la tolerancia en max_iter iteraciones.
    """
    x = np.array(x0, dtype=float).flatten()
    if x.size == 0:
        raise ValueError("El vector inicial x0 no puede ser vacío.")

    Fx = np.array(F(x), dtype=float).flatten()
    if Fx.size != x.size:
        raise ValueError("La dimensión de F(x) debe coincidir con la de x.")

    for k in range(1, max_iter + 1):
        Fx = np.array(F(x), dtype=float).flatten()
        Jx = np.array(J(x), dtype=float)

        n = x.size
        if Jx.shape != (n, n):
            raise ValueError("La matriz jacobiana J(x) debe ser cuadrada de dimensión n×n.")

        # Resolver J(x) y = -F(x) con Gauss (sin inversa)
        y = gauss_pivoteo_parcial(Jx, -Fx)

        x_nuevo = x + y

        # Error relativo: ||x_k - x_{k-1}||_inf / ||x_k||_inf
        num = np.linalg.norm(x_nuevo - x, ord=np.inf)
        den = np.linalg.norm(x_nuevo, ord=np.inf)
        rel = num / den if den > 0 else num

        if rel < tolerancia:
            return x_nuevo, k

        x = x_nuevo

    raise ValueError(f"El método de Newton no alcanzó la tolerancia en {max_iter} iteraciones.")


def demo_newton():
    """
    Demostración del método de Newton para el sistema:
        x1^2 + 4 x2^2 = 4
        4 x1^2 + x2^2 = 4

    - Usa Gauss con pivoteo parcial para resolver los sistemas lineales.
    - Imprime cada iteración hasta que el error relativo sea <= 1e-8.
    """
    np.set_printoptions(precision=10, suppress=True, linewidth=120)

    print("\n" + "="*80)
    print("MÉTODO DE NEWTON PARA UN SISTEMA NO LINEAL EN R^2")
    print("="*80)

    print("\nSistema de ecuaciones:")
    print("  1)  x1^2 + 4 x2^2 = 4")
    print("  2)  4 x1^2 + x2^2 = 4")

    # Vector inicial (puedes cambiarlo para explorar otras soluciones)
    x0 = np.array([1.0, 1.0])
    tolerancia = 1e-8
    max_iter = 50

    print(f"\nVector inicial x0 = {x0}")
    print(f"Tolerancia pedida = {tolerancia:.0e}\n")

    print("Iteraciones del método de Newton:")
    print("  k        x1                x2              error_relativo")
    print("------------------------------------------------------------------")

    x = x0.copy()
    # Mostrar k = 0
    print(f"{0:3d}  {x[0]: .12f}  {x[1]: .12f}        ---")

    for k in range(1, max_iter + 1):
        Fx = F_sistema(x)
        Jx = J_sistema(x)
        y = gauss_pivoteo_parcial(Jx, -Fx)

        x_nuevo = x + y

        num = np.linalg.norm(x_nuevo - x, ord=np.inf)
        den = np.linalg.norm(x_nuevo, ord=np.inf)
        rel = num / den if den > 0 else num

        print(f"{k:3d}  {x_nuevo[0]: .12f}  {x_nuevo[1]: .12f}  {rel: .3e}")

        if rel < tolerancia:
            print("\nCriterio de paro alcanzado: error relativo < tolerancia.")
            print(f"Número total de iteraciones: {k}")
            print("\nSolución aproximada x*:")
            print(x_nuevo)

            Fx_sol = F_sistema(x_nuevo)
            norma_fx = np.linalg.norm(Fx_sol, ord=np.inf)
            print("\nValor de F(x*) (para verificar):")
            print(Fx_sol)
            print(f"\n||F(x*)||_inf = {norma_fx:.3e}")
            return

        x = x_nuevo

    # Si no se alcanzó la tolerancia
    raise ValueError(f"El método de Newton no alcanzó la tolerancia en {max_iter} iteraciones.")


def demo():
    demo_newton()

demo()
