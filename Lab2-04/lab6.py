import numpy as np

def factorizacion_lu(A, tol=1e-12):
    """
    Calcula la factorización LU de una matriz A sin intercambios de renglones.
    A = L*U, donde L es triangular inferior con diagonal de unos y U es triangular superior.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    
    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n-1):
        if abs(U[k, k]) < tol:
            raise ValueError(
                f"La factorización LU no es posible: pivote nulo en posición ({k},{k}).\n"
                f"Se requiere pivoteo para esta matriz."
            )
        
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
    
    if abs(U[n-1, n-1]) < tol:
        raise ValueError(
            f"La factorización LU no es posible: pivote nulo en posición ({n-1},{n-1}).\n"
            f"La matriz es singular o requiere pivoteo."
        )
    
    return L, U


def verificar_factorizacion(A, L, U):
    """
    Verifica que L*U = A calculando el error de reconstrucción.
    """
    reconstruccion = L @ U
    error = np.linalg.norm(A - reconstruccion, ord=np.inf)
    return error


def resolver_con_lu(L, U, b):
    """
    Resuelve Ax=b usando la factorización LU.
    Primero resuelve Ly=b (sustitución adelante), luego Ux=y (sustitución atrás).
    """
    b = np.array(b, dtype=float).flatten()
    n = len(b)
    
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        y[i] /= L[i, i]
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i] - np.dot(U[i, i+1:], x[i+1:])
        x[i] /= U[i, i]
    
    return x


def factorizacion_ldlt(A, tol=1e-12):
    """
    Calcula la factorización LDLᵀ de una matriz A simétrica.
    A = L*D*Lᵀ, donde L es triangular inferior con diagonal unitaria y D es diagonal.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape

    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if not np.allclose(A, A.T, atol=tol):
        raise ValueError("La factorización LDLᵀ requiere que A sea simétrica (A ≈ Aᵀ).")

    L = np.eye(n)
    D = np.zeros(n)

    for k in range(n):
        suma = 0.0
        for s in range(k):
            suma += (L[k, s] ** 2) * D[s]
        D[k] = A[k, k] - suma

        if abs(D[k]) < tol:
            raise ValueError(
                f"La factorización LDLᵀ no es posible: pivote nulo en posición ({k},{k})."
            )

        for i in range(k+1, n):
            suma2 = 0.0
            for s in range(k):
                suma2 += L[i, s] * D[s] * L[k, s]
            L[i, k] = (A[i, k] - suma2) / D[k]

    Dmat = np.diag(D)
    return L, Dmat


def verificar_ldlt(A, L, D):
    """
    Verifica que L*D*Lᵀ = A calculando el error de reconstrucción.
    """
    reconstruccion = L @ D @ L.T
    error = np.linalg.norm(A - reconstruccion, ord=np.inf)
    return error


def factorizacion_cholesky(A, tol=1e-12):
    """
    Calcula la factorización de Cholesky (LLᵀ) de una matriz A definida positiva.
    A = L*Lᵀ, donde L es triangular inferior.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape

    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if not np.allclose(A, A.T, atol=tol):
        raise ValueError("La factorización de Cholesky requiere que A sea simétrica (A ≈ Aᵀ).")

    L = np.zeros_like(A)

    for i in range(n):
        suma = np.dot(L[i, :i], L[i, :i])
        diag = A[i, i] - suma
        if diag <= tol:
            raise ValueError("La factorización de Cholesky no es posible: la matriz no es definida positiva.")
        L[i, i] = np.sqrt(diag)
        for j in range(i+1, n):
            suma2 = np.dot(L[j, :i], L[i, :i])
            L[j, i] = (A[j, i] - suma2) / L[i, i]
    
    return L


def verificar_cholesky(A, L):
    """
    Verifica que L*Lᵀ = A calculando el error de reconstrucción.
    """
    reconstruccion = L @ L.T
    error = np.linalg.norm(A - reconstruccion, ord=np.inf)
    return error


def demo():
    """
    Demostración con ejemplos de factorización LU, LDLᵀ y Cholesky.
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=100)
    
    print("="*80)
    print("DEMOSTRACIÓN DE FACTORIZACIONES LU, LDLᵀ Y CHOLESKY")
    print("="*80)
    
    # Ejemplo LU
    print("\n" + "="*80)
    print("EJEMPLO 1: FACTORIZACIÓN LU")
    print("="*80)
    
    A1 = np.array([
        [4, 3, -1],
        [-2, -4, 5],
        [1, 2, 6]
    ], dtype=float)
    
    print("\nMatriz A:")
    print(A1)

    try:
        L1, U1 = factorizacion_lu(A1)
        print("\nMatriz L:")
        print(L1)
        print("\nMatriz U:")
        print(U1)
        error = verificar_factorizacion(A1, L1, U1)
        print(f"\nVerificación: ||A - LU||_∞ = {error:.3e}")
    except ValueError as e:
        print(f"\nERROR: {e}")

    # Ejemplo LDLᵀ
    print("\n" + "="*80)
    print("EJEMPLO 2: FACTORIZACIÓN LDLᵀ")
    print("="*80)

    A2 = np.array([
        [4, 1, 1],
        [1, 2, 0.5],
        [1, 0.5, 1]
    ], dtype=float)
    
    print("\nMatriz A:")
    print(A2)
    
    try:
        L2, D2 = factorizacion_ldlt(A2)
        print("\nMatriz L:")
        print(L2)
        print("\nMatriz D:")
        print(D2)
        error = verificar_ldlt(A2, L2, D2)
        print(f"\nVerificación: ||A - L*D*Lᵀ||_∞ = {error:.3e}")
    except ValueError as e:
        print(f"\nERROR: {e}")

    # Ejemplo Cholesky
    print("\n" + "="*80)
    print("EJEMPLO 3: FACTORIZACIÓN DE CHOLESKY (LLᵀ)")
    print("="*80)
    
    A3 = np.array([
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2]
    ], dtype=float)
    
    print("\nMatriz A:")
    print(A3)
    
    try:
        L3 = factorizacion_cholesky(A3)
        print("\nMatriz L:")
        print(L3)
        error = verificar_cholesky(A3, L3)
        print(f"\nVerificación: ||A - L*Lᵀ||_∞ = {error:.3e}")
    except ValueError as e:
        print(f"\nERROR: {e}")

    print("\n" + "="*80)
    print("FIN DE LA DEMOSTRACIÓN")
    print("="*80)

demo()
