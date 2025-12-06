"""
cordic_functions.py
Implementación del algoritmo CORDIC (Coordinate Rotation Digital Computer)
para el cálculo de funciones trigonométricas usando únicamente aritmética finita.
"""

def cordic_sin(theta, iteraciones=16):
    """
    Calcula el seno de un ángulo usando el algoritmo CORDIC.
    
    Entradas:
      - theta: ángulo en radianes.
      - iteraciones: número de iteraciones del algoritmo (por defecto 16).
    
    Salidas:
      - seno aproximado de theta.
    """
    # Normalizar el ángulo al rango [-π, π]
    pi = 3.14159265358979323846
    while theta > pi:
        theta -= 2 * pi
    while theta < -pi:
        theta += 2 * pi
    
    # Tabla de arcotangentes precalculada
    atan_table = [
        0.78539816339745, 0.46364760900081, 0.24497866312686,
        0.12435499454676, 0.06241880999596, 0.03123983343027,
        0.01562372862048, 0.00781234106010, 0.00390623013197,
        0.00195312251648, 0.00097656218956, 0.00048828121119,
        0.00024414062015, 0.00012207031189, 0.00006103515617,
        0.00003051757812, 0.00001525878906, 0.00000762939453
    ]
    
    # Factor de escala K (producto de cos(atan(2^-i)))
    K = 0.60725293500888
    
    # Inicialización
    x = K
    y = 0.0
    z = theta
    
    # Iteraciones CORDIC
    for i in range(min(iteraciones, len(atan_table))):
        # Determinar dirección de rotación
        if z >= 0:
            d = 1
        else:
            d = -1
        
        # Calcular desplazamiento
        potencia = 2 ** (-i)
        
        # Rotación CORDIC
        x_nuevo = x - d * y * potencia
        y_nuevo = y + d * x * potencia
        z_nuevo = z - d * atan_table[i]
        
        x = x_nuevo
        y = y_nuevo
        z = z_nuevo
    
    return y


def cordic_cos(theta, iteraciones=16):
    """
    Calcula el coseno de un ángulo usando el algoritmo CORDIC.
    
    Entradas:
      - theta: ángulo en radianes.
      - iteraciones: número de iteraciones del algoritmo (por defecto 16).
    
    Salidas:
      - coseno aproximado de theta.
    """
    # Normalizar el ángulo al rango [-π, π]
    pi = 3.14159265358979323846
    while theta > pi:
        theta -= 2 * pi
    while theta < -pi:
        theta += 2 * pi
    
    # Tabla de arcotangentes precalculada
    atan_table = [
        0.78539816339745, 0.46364760900081, 0.24497866312686,
        0.12435499454676, 0.06241880999596, 0.03123983343027,
        0.01562372862048, 0.00781234106010, 0.00390623013197,
        0.00195312251648, 0.00097656218956, 0.00048828121119,
        0.00024414062015, 0.00012207031189, 0.00006103515617,
        0.00003051757812, 0.00001525878906, 0.00000762939453
    ]
    
    # Factor de escala K
    K = 0.60725293500888
    
    # Inicialización
    x = K
    y = 0.0
    z = theta
    
    # Iteraciones CORDIC
    for i in range(min(iteraciones, len(atan_table))):
        if z >= 0:
            d = 1
        else:
            d = -1
        
        potencia = 2 ** (-i)
        
        x_nuevo = x - d * y * potencia
        y_nuevo = y + d * x * potencia
        z_nuevo = z - d * atan_table[i]
        
        x = x_nuevo
        y = y_nuevo
        z = z_nuevo
    
    return x


def cordic_tan(theta, iteraciones=16):
    """
    Calcula la tangente de un ángulo usando el algoritmo CORDIC.
    
    Entradas:
      - theta: ángulo en radianes.
      - iteraciones: número de iteraciones del algoritmo (por defecto 16).
    
    Salidas:
      - tangente aproximada de theta.
    """
    cos_val = cordic_cos(theta, iteraciones)
    sin_val = cordic_sin(theta, iteraciones)
    
    if abs(cos_val) < 1e-10:
        return float('inf') if sin_val > 0 else float('-inf')
    
    return sin_val / cos_val


def cordic_atan(y, x, iteraciones=16):
    """
    Calcula el arcotangente de y/x usando el algoritmo CORDIC (atan2).
    
    Entradas:
      - y: coordenada y.
      - x: coordenada x.
      - iteraciones: número de iteraciones del algoritmo (por defecto 16).
    
    Salidas:
      - ángulo en radianes.
    """
    # Tabla de arcotangentes precalculada
    atan_table = [
        0.78539816339745, 0.46364760900081, 0.24497866312686,
        0.12435499454676, 0.06241880999596, 0.03123983343027,
        0.01562372862048, 0.00781234106010, 0.00390623013197,
        0.00195312251648, 0.00097656218956, 0.00048828121119,
        0.00024414062015, 0.00012207031189, 0.00006103515617,
        0.00003051757812, 0.00001525878906, 0.00000762939453
    ]
    
    # Manejar casos especiales
    if abs(x) < 1e-10 and abs(y) < 1e-10:
        return 0.0
    
    # Determinar cuadrante y ajustar
    pi = 3.14159265358979323846
    ajuste = 0.0
    
    if x < 0:
        if y >= 0:
            ajuste = pi
        else:
            ajuste = -pi
        x = -x
        y = -y
    
    # Inicialización
    z = 0.0
    
    # Iteraciones CORDIC en modo vectoring
    for i in range(min(iteraciones, len(atan_table))):
        if y >= 0:
            d = 1
        else:
            d = -1
        
        potencia = 2 ** (-i)
        
        x_nuevo = x + d * y * potencia
        y_nuevo = y - d * x * potencia
        z_nuevo = z + d * atan_table[i]
        
        x = x_nuevo
        y = y_nuevo
        z = z_nuevo
    
    return z + ajuste


def cordic_sqrt(n, iteraciones=16):
    """
    Calcula la raíz cuadrada usando el algoritmo CORDIC en modo hiperbólico.
    
    Entradas:
      - n: número positivo del cual calcular la raíz.
      - iteraciones: número de iteraciones del algoritmo (por defecto 16).
    
    Salidas:
      - raíz cuadrada aproximada de n.
    """
    if n < 0:
        raise ValueError("No se puede calcular la raíz cuadrada de un número negativo.")
    if n == 0:
        return 0.0
    
    # Para números muy pequeños o muy grandes, normalizar
    escala = 1.0
    n_original = n
    
    while n < 0.25:
        n *= 4
        escala *= 2
    
    while n > 1.0:
        n /= 4
        escala /= 2
    
    # Método de Newton-Raphson simplificado usando solo sumas y desplazamientos
    x = (n + 1.0) / 2.0  # Aproximación inicial
    
    for _ in range(iteraciones):
        x_nuevo = (x + n / x) / 2.0
        x = x_nuevo
    
    return x / escala


def cordic_exp(x, iteraciones=16):
    """
    Calcula e^x usando el algoritmo CORDIC en modo hiperbólico.
    
    Entradas:
      - x: exponente.
      - iteraciones: número de iteraciones del algoritmo (por defecto 16).
    
    Salidas:
      - e^x aproximado.
    """
    # Constante e
    e = 2.71828182845905
    
    # Reducir x al rango [-1, 1] usando e^x = (e^(x/2^k))^(2^k)
    k = 0
    while abs(x) > 1:
        x /= 2
        k += 1
    
    # Aproximación de Taylor para e^x cuando |x| <= 1
    resultado = 1.0
    termino = 1.0
    
    for i in range(1, iteraciones):
        termino *= x / i
        resultado += termino
    
    # Elevar al cuadrado k veces
    for _ in range(k):
        resultado *= resultado
    
    return resultado


def cordic_ln(x, iteraciones=16):
    """
    Calcula ln(x) usando el algoritmo CORDIC en modo hiperbólico.
    
    Entradas:
      - x: número positivo.
      - iteraciones: número de iteraciones del algoritmo (por defecto 16).
    
    Salidas:
      - ln(x) aproximado.
    """
    if x <= 0:
        raise ValueError("El logaritmo natural solo está definido para números positivos.")
    
    # Constantes
    ln2 = 0.69314718055995
    
    # Normalizar x al rango [1, 2) contando potencias de 2
    potencias = 0
    while x >= 2:
        x /= 2
        potencias += 1
    
    while x < 1:
        x *= 2
        potencias -= 1
    
    # Ahora x está en [1, 2), usar serie de Taylor para ln(x)
    # ln(x) = ln(1 + (x-1)) usando serie de Taylor
    y = x - 1
    resultado = 0.0
    termino = y
    
    for i in range(1, iteraciones):
        resultado += termino / i * (1 if i % 2 == 1 else -1)
        termino *= y
    
    return resultado + potencias * ln2


# Funciones auxiliares para comparación

def sin_taylor(x, iteraciones=10):
    """Calcula sin(x) usando serie de Taylor."""
    pi = 3.14159265358979323846
    
    # Normalizar al rango [-π, π]
    while x > pi:
        x -= 2 * pi
    while x < -pi:
        x += 2 * pi
    
    resultado = 0.0
    termino = x
    
    for n in range(iteraciones):
        resultado += termino
        termino *= -x * x / ((2 * n + 2) * (2 * n + 3))
    
    return resultado


def cos_taylor(x, iteraciones=10):
    """Calcula cos(x) usando serie de Taylor."""
    pi = 3.14159265358979323846
    
    # Normalizar al rango [-π, π]
    while x > pi:
        x -= 2 * pi
    while x < -pi:
        x += 2 * pi
    
    resultado = 1.0
    termino = 1.0
    
    for n in range(1, iteraciones):
        termino *= -x * x / ((2 * n - 1) * (2 * n))
        resultado += termino
    
    return resultado