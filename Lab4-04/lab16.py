import numpy as np

def svd(A):
    """
    Descomposición en Valores Singulares (SVD) de una matriz A.
    Calcula A = U * Σ * V^T
    
    Entradas:
      - A: matriz de tamaño m x n.
    
    Salidas:
      - U: matriz ortogonal m x m (vectores singulares izquierdos).
      - Sigma: array con los valores singulares (ordenados de mayor a menor).
      - VT: matriz ortogonal n x n transpuesta (vectores singulares derechos).
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    # Paso 1: Calcular A^T * A
    ATA = A.T @ A
    
    # Paso 2: Calcular valores y vectores propios de A^T * A
    eigenvalues_ATA, V = np.linalg.eig(ATA)
    
    # Ordenar por valores propios descendentes
    idx = np.argsort(eigenvalues_ATA)[::-1]
    eigenvalues_ATA = eigenvalues_ATA[idx]
    V = V[:, idx]
    
    # Paso 3: Los valores singulares son las raíces cuadradas de los valores propios
    Sigma = np.sqrt(np.maximum(eigenvalues_ATA, 0))  # Evitar negativos por errores numéricos
    
    # Paso 4: Calcular U usando U = A * V * Σ^(-1)
    r = np.sum(Sigma > 1e-10)  # Rango de la matriz
    U = np.zeros((m, m))
    
    for i in range(r):
        U[:, i] = (A @ V[:, i]) / Sigma[i]
    
    # Completar U con vectores ortonormales adicionales si m > r
    if r < m:
        # Usar QR para completar la base ortogonal
        Q, _ = np.linalg.qr(np.random.randn(m, m))
        for i in range(r, m):
            # Ortogonalizar contra las columnas existentes de U
            v = Q[:, i]
            for j in range(i):
                v = v - np.dot(v, U[:, j]) * U[:, j]
            v = v / np.linalg.norm(v)
            U[:, i] = v
    
    # VT es la transpuesta de V
    VT = V.T
    
    return U, Sigma, VT

def demo_svd():
    """
    Demostración de la descomposición en valores singulares con una matriz 8x5.
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    
    print("\n" + "="*80)
    print("DEMOSTRACIÓN: DESCOMPOSICIÓN EN VALORES SINGULARES (SVD)")
    print("="*80)
    
    # Crear una matriz 8x5
    np.random.seed(42)
    A = np.random.rand(8, 5) * 10
    
    print("\nMatriz A (8 x 5):")
    print(A)
    print(f"\nDimensiones: {A.shape[0]} x {A.shape[1]}")
    
    # Aplicar SVD
    U, Sigma, VT = svd(A)
    
    print("\n" + "-"*80)
    print("RESULTADOS DE LA DESCOMPOSICIÓN SVD")
    print("-"*80)
    
    print("\nMatriz U (vectores singulares izquierdos) - 8 x 8:")
    print(U)
    print(f"Dimensiones de U: {U.shape}")
    
    print("\nValores singulares (Σ):")
    print(Sigma)
    print(f"Número de valores singulares: {len(Sigma)}")
    
    print("\nMatriz V^T (vectores singulares derechos transpuesta) - 5 x 5:")
    print(VT)
    print(f"Dimensiones de V^T: {VT.shape}")
    
    # Reconstruir la matriz A
    print("\n" + "-"*80)
    print("RECONSTRUCCIÓN DE LA MATRIZ A")
    print("-"*80)
    
    # Crear la matriz diagonal Sigma_matriz (8x5)
    Sigma_matriz = np.zeros((U.shape[0], VT.shape[0]))
    for i in range(min(len(Sigma), Sigma_matriz.shape[0], Sigma_matriz.shape[1])):
        Sigma_matriz[i, i] = Sigma[i]
    
    print("\nMatriz diagonal Σ (8 x 5):")
    print(Sigma_matriz)
    
    # Reconstruir A = U * Σ * V^T
    A_reconstructed = U @ Sigma_matriz @ VT
    
    print("\nMatriz A reconstruida (U * Σ * V^T):")
    print(A_reconstructed)
    
    # Calcular el error de reconstrucción
    error = np.linalg.norm(A - A_reconstructed)
    print(f"\nError de reconstrucción ||A - U*Σ*V^T||: {error:.2e}")
    
    # Verificar ortogonalidad de U
    print("\n" + "-"*80)
    print("VERIFICACIÓN DE PROPIEDADES")
    print("-"*80)
    
    UTU = U.T @ U
    print("\nU^T * U (debe ser identidad):")
    print(UTU)
    error_U = np.linalg.norm(UTU - np.eye(U.shape[1]))
    print(f"Error ||U^T*U - I||: {error_U:.2e}")
    
    # Verificar ortogonalidad de V
    V = VT.T
    VTV = V.T @ V
    print("\nV^T * V (debe ser identidad):")
    print(VTV)
    error_V = np.linalg.norm(VTV - np.eye(V.shape[1]))
    print(f"Error ||V^T*V - I||: {error_V:.2e}")


def demo():
    demo_svd()

demo()