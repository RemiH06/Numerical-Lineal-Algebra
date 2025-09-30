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
    
    # Inicializar L como identidad y U como copia de A
    L = np.eye(n)
    U = A.copy()
    
    # Proceso de eliminación gaussiana
    for k in range(n-1):
        # Verificar que el pivote no sea nulo
        if abs(U[k, k]) < tol:
            raise ValueError(
                f"La factorización LU no es posible: pivote nulo en posición ({k},{k}).\n"
                f"Se requiere pivoteo para esta matriz."
            )
        
        # Calcular multiplicadores y actualizar matrices
        for i in range(k+1, n):
            # Calcular el multiplicador (elemento de L)
            L[i, k] = U[i, k] / U[k, k]
            
            # Actualizar la fila i de U (eliminación)
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
    
    # Verificar el último pivote
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
    
    # Sustitución hacia adelante: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        y[i] /= L[i, i]
    
    # Sustitución hacia atrás: Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = y[i] - np.dot(U[i, i+1:], x[i+1:])
        x[i] /= U[i, i]
    
    return x


def demo():
    """
    Demostración con varios ejemplos de factorización LU.
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=100)
    
    print("="*80)
    print("DEMOSTRACIÓN DE FACTORIZACIÓN LU SIN PIVOTEO")
    print("="*80)
    
    # Ejemplo 1: Matriz 3x3 bien condicionada
    print("\n" + "="*80)
    print("EJEMPLO 1: Matriz 3x3 bien condicionada")
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
        print("\nMatriz L (triangular inferior):")
        print(L1)
        print("\nMatriz U (triangular superior):")
        print(U1)
        
        error = verificar_factorizacion(A1, L1, U1)
        print(f"\nVerificación: ||A - LU||_∞ = {error:.3e}")
        
        # Resolver un sistema
        b1 = np.array([1, 2, 3])
        x1 = resolver_con_lu(L1, U1, b1)
        x_ref = np.linalg.solve(A1, b1)
        print(f"\nSolución de Ax=b con b={b1}:")
        print(f"x (usando LU) = {x1}")
        print(f"Error vs numpy: {np.linalg.norm(x1 - x_ref):.3e}")
        
    except ValueError as e:
        print(f"\nERROR: {e}")
    
    # Ejemplo 2: Matriz 5x5 aleatoria
    print("\n" + "="*80)
    print("EJEMPLO 2: Matriz 5x5 aleatoria")
    print("="*80)
    
    np.random.seed(42)
    A2 = np.random.randn(5, 5)
    A2 = A2 + 3 * np.eye(5)  # Asegurar diagonal dominante
    
    print("\nMatriz A (5x5):")
    print(A2)
    
    try:
        L2, U2 = factorizacion_lu(A2)
        print("\nMatriz L:")
        print(L2)
        print("\nMatriz U:")
        print(U2)
        
        error = verificar_factorizacion(A2, L2, U2)
        print(f"\nVerificación: ||A - LU||_∞ = {error:.3e}")
        
    except ValueError as e:
        print(f"\nERROR: {e}")
    
    # Ejemplo 3: Matriz que requiere pivoteo (fallará)
    print("\n" + "="*80)
    print("EJEMPLO 3: Matriz que requiere pivoteo (esperamos que falle)")
    print("="*80)
    
    A3 = np.array([
        [0, 1, 2],
        [1, 1, 1],
        [2, 1, 4]
    ], dtype=float)
    
    print("\nMatriz A:")
    print(A3)
    
    try:
        L3, U3 = factorizacion_lu(A3)
        print("\nMatriz L:")
        print(L3)
        print("\nMatriz U:")
        print(U3)
        
    except ValueError as e:
        print(f"\n✓ ERROR ESPERADO: {e}")
    
    # Ejemplo 4: Matriz singular
    print("\n" + "="*80)
    print("EJEMPLO 4: Matriz singular (filas linealmente dependientes)")
    print("="*80)
    
    A4 = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]
    ], dtype=float)
    
    print("\nMatriz A:")
    print(A4)
    
    try:
        L4, U4 = factorizacion_lu(A4)
        print("\nMatriz L:")
        print(L4)
        print("\nMatriz U:")
        print(U4)
        
    except ValueError as e:
        print(f"\n✓ ERROR ESPERADO: {e}")
    
    print("\n" + "="*80)
    print("FIN DE LA DEMOSTRACIÓN")
    print("="*80)


if __name__ == "__main__":
    demo()