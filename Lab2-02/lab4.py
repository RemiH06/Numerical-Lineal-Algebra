"""
Modifique su código para el método de Gauss que incluya el pivoteo parcial. En cada paso de eliminación debe seleccionar el término con el mayor valor absoluto para conservarlo como pivote, esto puede requerir intercambio de renglones.

Modifique su código para el método de Gauss que incluya el pivoteo parcial con escalamiento. En cada paso debe normalizar los elementos de cada renglón respecto al término que tenga el mayor valor absoluto. 

Compruebe el funcionamiento de cada caso verificando la solución de un sistema con n=5 (puede comprobar las soluciones usando las funciones de la biblioteca linalg).

Compruebe el funcionamiento verificando la solución de un sistema con n=10 (puede comprobar las soluciones usando las funciones de la biblioteca linalg).
"""
import numpy as np

def gauss(A, b, tol=1e-12):
    """
    Resuelve Ax=b mediante eliminación gaussiana simple (sin pivoteo).
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
        raise ValueError("El vector b debe ser de dimensión (n, 1).")

    # Matriz aumentada
    M = np.hstack([A, b]).astype(float)

    # Eliminación hacia adelante
    for col in range(n):
        piv_val = M[col, col]
        
        # Si el pivote es ~0, no hay solución única
        if abs(piv_val) < tol:
            raise ValueError("No hay solución única: pivote nulo (matriz singular).")
        
        # Normalizar fila del pivote
        M[col, :] = M[col, :] / piv_val
        
        # Eliminar en las filas inferiores
        for row in range(col + 1, n):
            factor = M[row, col]
            M[row, :] = M[row, :] - factor * M[col, :]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = M[i, n]
        for j in range(i+1, n):
            x[i] -= M[i, j] * x[j]
        x[i] /= M[i, i]

    return x

def gauss_pivoteo_parcial(A, b, tol=1e-12):
    """
    Resuelve Ax=b mediante eliminación gaussiana con pivoteo parcial.
    En cada paso selecciona el término con mayor valor absoluto como pivote.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    n, m = A.shape
    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if b.shape != (n, 1):
        raise ValueError("El vector b debe ser de dimensión (n, 1).")

    # Matriz aumentada
    M = np.hstack([A, b]).astype(float)

    # Eliminación con pivoteo parcial
    for col in range(n):
        # Encontrar pivote con valor absoluto máximo en la columna actual
        piv_row = np.argmax(np.abs(M[col:, col])) + col
        piv_val = M[piv_row, col]

        # Si el pivote es ~0, no hay solución única
        if abs(piv_val) < tol:
            raise ValueError("No hay solución única: pivote nulo (matriz singular).")

        # Intercambiar filas si es necesario
        if piv_row != col:
            M[[col, piv_row], :] = M[[piv_row, col], :]

        # Eliminar en las filas inferiores
        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, :] = M[row, :] - factor * M[col, :]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = M[i, n]
        for j in range(i+1, n):
            x[i] -= M[i, j] * x[j]
        x[i] /= M[i, i]

    return x

def gauss_pivoteo_escalamiento(A, b, tol=1e-12):
    """
    Resuelve Ax=b mediante eliminación gaussiana con pivoteo parcial con escalamiento.
    Normaliza los elementos de cada renglón respecto al término con mayor valor absoluto.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    n, m = A.shape
    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if b.shape != (n, 1):
        raise ValueError("El vector b debe ser de dimensión (n, 1).")

    # Matriz aumentada
    M = np.hstack([A, b]).astype(float)
    
    # Calcular factores de escalamiento para cada fila
    s = np.max(np.abs(M[:, :-1]), axis=1)

    # Eliminación con pivoteo por escalamiento
    for col in range(n):
        # Encontrar pivote usando escalamiento
        scaled_ratios = np.abs(M[col:, col]) / s[col:]
        piv_row = np.argmax(scaled_ratios) + col
        piv_val = M[piv_row, col]

        # Si el pivote es ~0, no hay solución única
        if abs(piv_val) < tol:
            raise ValueError("No hay solución única: pivote nulo (matriz singular).")

        # Intercambiar filas si es necesario
        if piv_row != col:
            M[[col, piv_row], :] = M[[piv_row, col], :]
            s[[col, piv_row]] = s[[piv_row, col]]  # También intercambiar factores de escalamiento

        # Eliminar en las filas inferiores
        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, :] = M[row, :] - factor * M[col, :]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = M[i, n]
        for j in range(i+1, n):
            x[i] -= M[i, j] * x[j]
        x[i] /= M[i, i]

    return x

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
        # Solución con numpy (referencia)
        x_np = np.linalg.solve(A, b)
        print("\nSolución de referencia (np.linalg.solve):")
        print(x_np)
        
        # Probar los tres métodos
        print("\n" + "="*50)
        print("COMPARACIÓN DE MÉTODOS")
        print("="*50)
        
        # Método simple
        try:
            x_simple = gauss(A, b)
            err_simple = np.linalg.norm(x_simple - x_np, ord=np.inf)
            print(f"\nGauss Simple:     ||error||_∞ = {err_simple:.3e}")
        except Exception as e:
            print(f"\nGauss Simple:     ERROR - {e}")
        
        # Pivoteo parcial
        try:
            x_parcial = gauss_pivoteo_parcial(A, b)
            err_parcial = np.linalg.norm(x_parcial - x_np, ord=np.inf)
            print(f"Pivoteo Parcial: ||error||_∞ = {err_parcial:.3e}")
        except Exception as e:
            print(f"Pivoteo Parcial: ERROR - {e}")
        
        # Pivoteo con escalamiento
        try:
            x_escala = gauss_pivoteo_escalamiento(A, b)
            err_escala = np.linalg.norm(x_escala - x_np, ord=np.inf)
            print(f"Pivoteo Escala:  ||error||_∞ = {err_escala:.3e}")
        except Exception as e:
            print(f"Pivoteo Escala:  ERROR - {e}")
            
    except Exception as e:
        print(f"\n[ERROR] No se pudo resolver el sistema: {e}")

def demo():
    np.set_printoptions(precision=6, suppress=True)
    rng = np.random.default_rng(42)

    print("DEMOSTRACIÓN DE MÉTODOS DE ELIMINACIÓN GAUSSIANA")
    print("="*80)

    # Caso n=5
    n = 5
    A5 = rng.normal(size=(n, n))
    A5 = A5 + 2.0 * np.eye(n)  # Asegurar invertibilidad
    x_true_5 = rng.normal(size=n)
    b5 = A5 @ x_true_5

    probar_sistema(A5, b5, titulo="PRUEBA CON n=5")

    # Caso n=10
    n = 10
    A10 = rng.normal(size=(n, n))
    A10 = A10 + 2.0 * np.eye(n)
    x_true_10 = rng.normal(size=n)
    b10 = A10 @ x_true_10

    probar_sistema(A10, b10, titulo="PRUEBA CON n=10")

    # Caso con matriz mal condicionada (para mostrar ventajas del pivoteo)
    print("\n" + "="*80)
    print("PRUEBA CON MATRIZ MAL CONDICIONADA")
    print("="*80)
    
    A_mal = np.array([
        [1e-10, 1, 1],
        [1, 1, 1],
        [1, 1, 2]
    ], dtype=float)
    b_mal = np.array([1, 0, 1], dtype=float)
    
    probar_sistema(A_mal, b_mal, titulo="Matriz mal condicionada")



demo()