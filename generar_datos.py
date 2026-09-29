"""
Genera un dataset FICTICIO de una empresa de repuestos con varias sucursales.
Está sucio a propósito (nulos, duplicados, textos inconsistentes, atípicos,
fechas en distintos formatos) para que practiques limpieza con pandas.

Uso:
    python generar_datos.py

Crea la carpeta datos_crudos/ con 4 archivos CSV:
    sucursales.csv, productos.csv, ventas.csv, stock.csv
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # semilla fija: siempre salen los mismos datos
os.makedirs("datos_crudos", exist_ok=True)

# ---------------------------------------------------------------- sucursales
sucursales = pd.DataFrame({
    "id_sucursal": [1, 2, 3, 4, 5, 6],
    "nombre": ["Santiago Centro", "Providencia", "Maipú",
               "Puente Alto", "Viña del Mar", "Concepción"],
    "region": ["RM", "RM", "RM", "RM", "Valparaíso", "Biobío"],
})

# ------------------------------------------------------------------ productos
categorias = {
    "Frenos": ["Pastillas de freno", "Disco de freno", "Líquido de frenos"],
    "Motor": ["Filtro de aceite", "Filtro de aire", "Bujía", "Correa de distribución"],
    "Suspensión": ["Amortiguador", "Rótula", "Barra estabilizadora"],
    "Eléctrico": ["Batería 60Ah", "Alternador", "Ampolleta H4"],
    "Lubricantes": ["Aceite 5W30 1L", "Aceite 10W40 1L", "Refrigerante 1L"],
}
filas = []
pid = 100
for cat, items in categorias.items():
    for item in items:
        pid += 1
        filas.append({
            "id_producto": pid,
            "producto": item,
            "categoria": cat,
            "precio_unitario": int(rng.integers(3, 120) * 1000 / 10),
        })
productos = pd.DataFrame(filas)

# --------------------------------------------------------------------- ventas
# Cada sucursal tiene una "demanda" distinta por producto, para que existan
# desbalances reales (sobra en unas, falta en otras).
fechas = pd.date_range("2025-10-01", "2026-09-15", freq="D")
factor_sucursal = {1: 1.4, 2: 1.2, 3: 1.0, 4: 0.9, 5: 0.7, 6: 0.5}
afinidad = {(s, p): rng.uniform(0.3, 1.8)
            for s in sucursales["id_sucursal"] for p in productos["id_producto"]}

registros = []
for f in fechas:
    for s in sucursales["id_sucursal"]:
        for p in productos["id_producto"]:
            lam = 0.35 * factor_sucursal[s] * afinidad[(s, p)]
            q = rng.poisson(lam)
            if q > 0:
                registros.append((f, s, p, q))
ventas = pd.DataFrame(registros, columns=["fecha", "id_sucursal", "id_producto", "cantidad"])
ventas.insert(0, "id_venta", range(1, len(ventas) + 1))
ventas = ventas.merge(productos[["id_producto", "precio_unitario"]], on="id_producto")
ventas["total"] = ventas["cantidad"] * ventas["precio_unitario"]
ventas = ventas.drop(columns="precio_unitario").sort_values("id_venta").reset_index(drop=True)

# ---------------------------------------------------------------------- stock
# Stock actual por sucursal y producto: unas sobrestockeadas, otras casi en cero.
stock_rows = []
demanda_dia = ventas.groupby(["id_sucursal", "id_producto"])["cantidad"].sum() / len(fechas)
for (s, p), d in demanda_dia.items():
    dias_cobertura = rng.choice([2, 5, 15, 30, 60, 120], p=[.12, .18, .25, .2, .15, .10])
    stock_rows.append((s, p, int(round(d * dias_cobertura))))
stock = pd.DataFrame(stock_rows, columns=["id_sucursal", "id_producto", "stock_actual"])

# =========================================================== ENSUCIAR DATOS
# 1) Nombres de sucursal inconsistentes
variantes = {
    "Santiago Centro": ["Santiago Centro", "santiago centro", "STGO CENTRO", "Santiago Centro "],
    "Providencia": ["Providencia", "providencia", "Providencia "],
    "Maipú": ["Maipú", "Maipu", "MAIPU"],
    "Puente Alto": ["Puente Alto", "puente alto", "P. Alto"],
    "Viña del Mar": ["Viña del Mar", "Vina del Mar", "viña del mar"],
    "Concepción": ["Concepción", "Concepcion", "concepción"],
}
sucursales["nombre"] = sucursales["nombre"].apply(lambda n: rng.choice(variantes[n]))

# 2) Precios como texto con símbolo en algunos productos
productos["precio_unitario"] = productos["precio_unitario"].astype(object)
idx = rng.choice(productos.index, size=5, replace=False)
productos.loc[idx, "precio_unitario"] = productos.loc[idx, "precio_unitario"].apply(
    lambda x: f"${int(x):,}".replace(",", "."))

# 3) Ventas: nulos, atípicos, negativos, fechas mixtas, duplicados
n = len(ventas)
ventas["fecha"] = ventas["fecha"].dt.strftime("%Y-%m-%d")
mixtas = rng.choice(ventas.index, size=int(n * 0.03), replace=False)
ventas.loc[mixtas, "fecha"] = pd.to_datetime(ventas.loc[mixtas, "fecha"]).dt.strftime("%d/%m/%Y")

ventas.loc[rng.choice(ventas.index, size=int(n * 0.02), replace=False), "cantidad"] = np.nan
ventas.loc[rng.choice(ventas.index, size=int(n * 0.01), replace=False), "id_sucursal"] = np.nan
ventas.loc[rng.choice(ventas.index, size=12, replace=False), "cantidad"] = 9999       # atípicos
ventas.loc[rng.choice(ventas.index, size=8, replace=False), "cantidad"] = -3          # negativos
ventas = pd.concat([ventas, ventas.sample(int(n * 0.015), random_state=1)], ignore_index=True)  # duplicados
ventas = ventas.sample(frac=1, random_state=7).reset_index(drop=True)

# 4) Stock: nulos y algunos valores absurdos
stock.loc[rng.choice(stock.index, size=4, replace=False), "stock_actual"] = np.nan
stock.loc[rng.choice(stock.index, size=3, replace=False), "stock_actual"] = -50

# ------------------------------------------------------------------- guardar
sucursales.to_csv("datos_crudos/sucursales.csv", index=False)
productos.to_csv("datos_crudos/productos.csv", index=False)
ventas.to_csv("datos_crudos/ventas.csv", index=False)
stock.to_csv("datos_crudos/stock.csv", index=False)

print("Listo. Archivos creados en datos_crudos/")
print(f"  sucursales: {len(sucursales)} filas")
print(f"  productos:  {len(productos)} filas")
print(f"  ventas:     {len(ventas)} filas")
print(f"  stock:      {len(stock)} filas")
