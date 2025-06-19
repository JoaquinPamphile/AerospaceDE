# AerospaceDE

# Hoja de ruta – Data/ML Engineer → Industria Espacial  
*(16 semanas · ≈ 10 h/semana · Python + Spark + AWS)*  

Cada módulo: **concepto físico → impacto “data” → snippet Python**.  
Las semanas son orientativas; ajusta la carga a tu ritmo.

---

## 0 · Preparación (Semana 0)

| Paso | Acción | Resultado |
|------|--------|-----------|
| 0.1 | Crear entorno | `conda create -n space python=3.12` → `conda activate space` |
| 0.2 | Instalar librerías | `pip install sgp4 poliastro astropy pyspark airflow pandas tensorflow prophet` |
| 0.3 | Infra mínima | bucket **s3://space-lake** · IAM EMR · API-key Space-Track |
| 0.4 | Datasets | TLE (CelesTrak) · Sentinel-2 (AWS ODC) · SatNOGS |

---

## 1 · Índice general

| Fase | Semanas | Foco |
|------|---------|----------------------------------------------|
| **A** | 1-6  | Mecánica orbital · dinámica interplanetaria |
| **B** | 7-10 | ETL · frames · SGP4 · lakehouse |
| **C** | 11-13| Predicción BSTAR · anomalías telemetría |
| **D** | 14-16| Proyectos finales · MLOps · entrevistas |

---

## A · Fundamentos Físicos (Sem 1-6)

| Nº | Módulo | Clave física | Mini-snippet |
|----|--------|--------------|--------------|
| 1 | Gravitación & órbita circ. | \(v_c=\sqrt{μ/r}\) | `(MU/(6378+700))**0.5` |
| 2 | Kepler + Vis-Viva | \(v^{2}=μ\,(2/r−1/a)\) | `visviva(7000,8000)` |
| 3 | COE clásicos | (a,e,i,Ω,ω,M) ↔ (r,v) | poliastro demo |
| 4 | Ecuación Kepler | \(M=E−e\sin E\) | `kepler_newton()` |
| 5 | Regiones especiales | Hill · Lagrange · Roche | `hill_radius()` |
| 6 | Dinámica interplanetaria | Hohmann · Lambert · GA | poliastro ΔV |

### 1.1 Teoría & ejemplos

#### 1 · Gravitación & órbita circular
Convención: μ = 3.986 × 10<sup>5</sup> km³/s² (Tierra)

Imagina que atas una bola a una cuerda y la haces girar.
La cuerda “tira” hacia tu mano con la misma fuerza con la que la bola “quiere” escapar por la tangente.
En órbita ocurre lo mismo:  la gravedad es la cuerda; la inercia de la nave intenta escapar por la tangente.


Ecuaciones clave

$$
F_{\text{grav}} = \frac{G\,m\,M}{r^{2}}, \qquad
F_{\text{cent}} = \frac{m\,v^{2}}{r}, \qquad
$$

Al igualarlas se obtiene la velocidad de equilibrio

$$
v_{c} = \sqrt{\frac{\mu}{r}}, \qquad
$$

y su periodo

$$
P = 2\pi \sqrt{\frac{r^{3}}{\mu}}.
$$
 
Caso real: a 400 km de altitud (ISS) r ≈ 6778 km → v ≈ 7.67 km/s y
P ≈ 92 min. Cualquier simulación de telemetría que muestre 9 km/s en LEO está claramente mal calibrada.

#### Uso en datos
Una regla de saneamiento sencilla, Filtra efemérides con arrastre atmosférico extremo o TLE vencido:
```sql
WHERE ABS(v_mag - SQRT(mu/r)) < 0.15
```

MU = 3.986004418e5
r = 6378.136 + 700
v_c = (MU/r)**0.5 → 7.55 km/s

Fuerza centrífuga = fuerza gravitatoria ⇒ fórmula.

Usa esto para un test Spark que verifique |v| ± 2 %. Si |v| difiere de 
v_c > 2 %, el TLE está viejo o hubo maniobra.
```python
MU = 3.986004418e5      # km^3/s^2
r  = 6378.136 + 700     # km
v_c = (MU / r)**0.5     # km/s
P   = 2*3.14159265*(r**3 / MU)**0.5 / 60  # min
print(v_c, P)
```

## 2 · Kepler & Vis-Viva
| Ley | Formulación                      | Idea clave                | Uso Práctico
| --- | -------------------------------- | ------------------------- |--------------
| 1.ª | Trayectoria = cónica             | Foco en el cuerpo central |Diferenciar órbitas elípticas vs hiperbólicas 
| 2.ª | Áreas iguales en tiempos iguales  $(h=r^{2}θ˙=cte)$| Conserva momento angular  | Conserva momento angular ⇒ define ground-track
| 3.ª | $P^{2}=4\pi^{2}a^{3}/\mu$        | Periodo ↔ semieje mayor   | Verificar períodos a partir de semieje

Analogía “línea del tren”
Piensa en un tren elíptico: avanza despacio en las curvas amplias (afelio) y acelera cuando entra en la curva cerrada (perigeo) para “no caerse de la vía”.
La segunda ley (áreas iguales) garantiza que compensa la lentitud con la longitud del tramo.

Vis-Viva traduce esa historia a energía:
$$
v^{2} \;=\; \mu \!\left( \frac{2}{r} - \frac{1}{a} \right)
$$

Permite calcular la velocidad en cualquier punto sin integrar la órbita.

Regímenes típicos LEO:

a≈6700–7150km,e<0.01,v≈7–8km/s.

Si r = a → vuelve a salir la velocidad circular.

Si r → ∞ pero a > 0 → v → 0 (órbita ligada).

Si a < 0 (hipérbola) el término −1/a cambia de signo y v nunca se cancela (escape).

Caso de uso – cross-check rápido
Al propagar con SGP4 obtienes r,v. Aplicar Vis-Viva con el a del mismo TLE debería devolver la misma v con < 1 m/s de error. Si se desvía decenas de m/s, tu propagación entró en región de fuerte drag o la nave realizó un impulso.

```python
from math import sqrt
def visviva(r, a, mu=3.986e5):
    return sqrt(mu * (2/r - 1/a))
print(visviva(7000, 8000))
```
Valida la velocidad que devuelve tu propagador:
```python
def visviva(r, a, mu=3.986e5):
    from math import sqrt
    return sqrt(mu*(2/r - 1/a))
```
## 3 · COE clásicos

| Símbolo  | Nombre                 | Rango                        |
| -------- | ---------------------- | ---------------------------- |
| $a$      | Semieje mayor          | $>0$ elipse                  |
| $e$      | Excentricidad          | $0$ (círculo) → 1 (parabola) |
| $i$      | Inclinación            | $0–180°$                     |
| $\Omega$ | RAAN (nodo ascendente) | $0–360°$                     |
| $\omega$ | Arg. perigeo           | $0–360°$                     |
| $M$      | Anomalía media         | $0–360°$                     |

Los COE son las “coordenadas GPS” de una órbita:
#### interpretación geométrica:
 - Plano:

        𝑖 (tilt): define el plano, como la inclinación de una pista de avión respecto al ecuador; 
        Ω (longitud del nodo ascendente):  su orientación respecto al meridiano de Greenwich, el meridiano por donde la pista cruza hacia el norte;
- Forma:

        a y e: longitud y “achatamiento” de la pista elíptica.

- Orientación dentro del plano

        𝜔 coloca el perigeo dentro del plano; 
        𝑀 dice “dónde” está el satélite en la elipse en 𝑡0.

- Conversión cartesiano ↔ COE: usa $h=r×v,n=k×h,e=1/μ[(v^{2} − μ/r)r−(r⋅v)v].$

Los tres primeros fijan la forma + plano; los tres últimos, la orientación + posición.
```python
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
import astropy.units as u
r = [-6045, -3490, 2500] * u.km
v = [-3.457,  6.618, 2.533] * u.km/u.s
print(Orbit.from_vectors(Earth, r, v).classical())
```
## 4 · Solver de Kepler

Analogía “reloj de arena”
El reloj da siempre el mismo tiempo de caída (anomalía media M crece lineal), pero cada grano (satélite) tarda más o menos en atravesar el estrechamiento según el “tapón” (excentricidad).
Calcular E es averiguar cuánto lleva el grano atravesando el cuello.

Métodos

- Newton-Raphson: rápido si la órbita no es casi parabólica.

- Halley / Householder: añaden curvatura (2ª o 3ª derivada) y convergen en 2 pasos para e > 0.9.

Ellíptica ( e<1 ): 
$$
M \;=\; E \;-\; e \,\sin E
$$
Hipérbola ( e>1 ): 
$$
M=esinhH−H
$$
Derivada para Newton–Raphson::
$$
\frac{dM}{dE} \;=\; 1 \;-\; e \,\cos E
$$


Solver Newton

```python
import numpy as np
def kepler_newton(M, e, tol=1e-12):
    M = np.mod(M, 2*np.pi)
    E = M if e < .8 else np.pi
    while True:
        dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        E -= dE
        if abs(dE) < tol:
            return E
```
Con 5 iteraciones resuelve e ≤ 0.9 con error < 1 × 10⁻¹² rad.


## 5 · Hill-sphere - Regiones Gravitacionales Especiales

| Concepto           | Expresión                                      | Relevancia                                                          |
| ------------------ | ---------------------------------------------- | ------------------------------------------------------------------- |
| **Hill-sphere**    | $r_H \simeq a\left(\dfrac{m}{3M}\right)^{1/3}$ | Límite donde un cuerpo retiene satélites (≈ 1.5 Gm para la Tierra). |
| **Lagrange L1-L5** | Soluciones de 3-cuerpos                        | L1/L2 para observatorios (SOHO, JWST). L4/L5 estables → “Troyanos”. |
| **Roche limit**    | $r_R \approx 2.44\,R_p (\rho_p/\rho_s)^{1/3}$  | Dentro de él un satélite fluvial se fragmenta (anillos de Saturno). |

| Analogía                                                                            | Uso satelital                                     |
| ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| “Zona wifi” de la Tierra; fuera ya predomina el router (Sol)                        | Elegir propagador: SGP4 vs efemérides planetarias |
| Vórtices en un río: el agua (fuerzas) gira pero el corcho (satélite) queda atrapado | Ubicar observatorios solares (SOHO, JWST)         |
| Limite de “marea” donde la gravita-tía te fragmenta                                 | Orbitadores de cometas: mantener > 2 r *Roche*    |


Aplicación data: elegir qué efemérides (planetarias vs satelitales) usar según si 𝑟 está fuera o dentro de la esfera de Hill. Regla práctica: si r > r_H usa efemérides solares, no SGP4.

```python
MU_E, MU_S = 3.986e5, 1.327e11   # km^3/s^2
AU = 1.496e8                      # km
hill = AU * (MU_E / (3*MU_S))**(1/3)
print(hill / 1e3, "Mm")
```
## 6 ·Dinámica Interplanetaria -  ΔV Hohmann GEO

| Maniobra           | Fórmula clave                                                                                                                                                    | Orden de magnitud                         |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **Hohmann** ΔV     | $\Delta v_1 = \sqrt{\frac{\mu}{r_1}}\bigl(\sqrt{\frac{2r_2}{r_1+r_2}}-1\bigr)$<br>$\Delta v_2 = \sqrt{\frac{\mu}{r_2}}\bigl(1-\sqrt{\frac{2r_1}{r_1+r_2}}\bigr)$ | LEO→GEO ≈ 3.9 km/s                        |
| **Lambert**        | Resuelve órbita que pasa por $r_1,r_2$ en Δt                                                                                                                     | Planificación de encuentros (rendez-vous) |
| **Gravity Assist** | $\Delta v_{\infty} = 2 v_{\infty}\sin\frac{\delta}{2}$ donde $\delta$ depende de $e$ hiperbólica                                                                 | Voyager ganó > 10 km/s                    |

- Patched-conics: combina tramos 2-cuerpos enlazados en puntos de esfera de influencia.

- Ingeniería de datos: calcular ΔV y tiempos de vuelo permite etiquetar segmentos de telemetría como coasting vs burns automáticamente.

Hohmann como cambio de “nivel”
Subir de piso en un edificio:

1. Tomas la escalera rolante (ΔV₁) que te lleva al entre-piso (elipse de transferencia).

2. Bajas en la nueva planta y das otro impulso (ΔV₂) para igualar el ritmo de la cinta de ese piso (órbita final).

Para planos coplanares es el mínimo ΔV posible.

Lambert

El “GPS inverso”: conoces punto A, punto B y el tiempo de viaje → ¿qué órbita pasa por ambos?
Fundamental para interceptar satélites (rendez-vous) y para reconstruir trayectorias históricas (OD inversa).

Gravity assist

Patín sobre hielo: te abalanzas sobre un compañero y, sin empujarlo, te catapultas porque le robas parte de su momento.
Permite ganar 10–15 km/s sin combustible (Voyager, BepiColombo).

Caso de uso ML
Etiquetar épocas de “coasting” vs “burn” en telemetría:

- Deriva lenta de energía ⇒ fase de transferencia.

- Spike de ΔV medido ⇒ maniobra (vector thrust ≠ 0).

Aplicaciones “data” rápidas
 - Marcado de maniobras: si 
∣v∣−vvisviva∣>50m/s en un paso, etiqueta como burn.

- Clasificación de órbitas: e < 0.01 y i ∈ [51°, 56°] ⇒ probable satélite ISS-visita.

- Detectar decay: derivada negativa sostenida de semimayor a > 1 km/día.


```python
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
leo = Orbit.circular(Earth, 300*u.km)
geo = Orbit.circular(Earth, 35786*u.km)
print(Maneuver.hohmann(leo, geo).get_total_cost())
```
## B · Canal de Datos (Sem 7-10)

| Sem | Tema                            | Entregable            |
| --- | ------------------------------- | --------------------- |
| 7   | Parseo TLE + checksum → Parquet | **raw\_tle**          |
| 8   | Airflow ingest Space-Track      | DAG + Grafana         |
| 9   | Spark SGP4 (1 min)              | **ephemeris\_bronze** |
| 10  | TEME → lat/lon                  | **ephemeris\_silver** |
| 11  | GIS & Datos Geo‑espaciales                  | geo_lake |

## 7 · Parseo TLE + checksum → Parquet  

| Campo | Fórmula/valor | Interpretación física |
|-------|---------------|-----------------------|
| **n** (col 53-63 L2) | $$ n = \sqrt{\mu/a^{3}}\;[\,\text{rev/d}\,] $$ | Velocidad angular media (Kepler) |
| **e** (27-33 L2) | $$ r_p = a(1-e) $$ | Forma de la elipse |
| **i** | ángulo plano-ecuador | Plano orbital |
| **BSTAR** |  $ B^{*} = \dfrac{C_D\,A}{m}\,\rho_0 $ | Se precisa para drag (densidad $\approx 10^{-12}\,\text{kg}\,\text{m}^{-3}$ en LEO) |
  |

La Tierra no es una esfera; su abultamiento produce precesiones seculares.

$$
n' = n\,\Bigl( 1 +
  \tfrac{3}{2}J_2\Bigl(\tfrac{R_e}{a}\Bigr)^2 \sqrt{1-e^2}\,(1-\tfrac{3}{2}\sin^2 i)
\Bigr)
$$

Piense en el TLE como una **tarjeta de embarque**:  
– n, e, i = destino y tipo de vuelo;  
– BSTAR = cuánta gasolina perderá por fricción (drag) en el trayecto.

su checksum es el código QR que avisa si hay un dígito mal.


**Estructura de un TLE**  
```text
Line-1 cols: 01  03································68
            1 25544U 98067A   24196.52780015  .00023198 … 0  9990
            ^ ^     ^           ^
            | |     |           |__ Epoch 2024-196.52780015 (UTC)
            | |     |____________ Intl. designator (year 1998, launch 067, piece A)
            | |__________________ Catalog number 25544 (ISS)
            |____________________ Line number (always 1)
```
**Checksum** 

suma de todos los dígitos + “−” (cuenta como 1) mod 10 ⇒ último carácter.

chk = $\displaystyle \left(\sum_{i=1}^{67} d_i\right)\bmod 10$



```python
import re, pandas as pd, requests
URL = "https://celestrak.org/NORAD/elements/stations.txt"
lines = requests.get(URL, timeout=10).text.splitlines()

def checksum_ok(l):
    s = sum(int(c) if c.isdigit() else 1 for c in l[:-1])
    return (s % 10) == int(l[-1])

records = []
for i in range(0, len(lines), 3):
    name, l1, l2 = lines[i:i+3]
    if checksum_ok(l1) and checksum_ok(l2):
        records.append((name.strip(), l1, l2))
df = pd.DataFrame(records, columns=["name", "l1", "l2"])
df.to_parquet("raw_tle/2024-07-15.parquet")
```

## 8 · Ingesta Space-Track con Airflow

Un TLE «envejece»: el error in-track crece
≈ 1 km / día en LEO. Actualizar cada ~12 h mantiene el error < 2 km.

- DAG (Directed Acyclic Graph) = plano de vuelo – cada tarea un eslabón; el scheduler es el “ATC” que inicia vuelos.

- Planificación: Expresión CRON “12,42 * * * *” = descargar a hh:12 y hh:42, justo después de
que Space-Track actualiza su catálogo.

- Observabilidad: Métricas Prometheus = “telemetría” del pipeline: latencia, n.º TLE, fallos chk.

```python
from airflow import DAG
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3, gzip, io, pandas as pd

def upload_to_s3(ti, bucket, key):
    text = ti.xcom_pull(task_ids="get_tle")
    df   = pd.DataFrame(text.splitlines())
    buf  = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="w") as f:
        f.write(text.encode())
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

with DAG("tle_ingest",
         start_date=datetime(2024,7,15),
         schedule="12,42 * * * *",
         catchup=False) as dag:

    get_tle = SimpleHttpOperator(
        task_id="get_tle",
        http_conn_id="space_track",
        endpoint="/basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/tle",
        method="GET",
    )

    to_s3 = PythonOperator(
        task_id="to_s3",
        python_callable=upload_to_s3,
        op_kwargs={"bucket":"space-lake","key":"raw_tle/{{ ds_nodash }}.gz"},
    )

    get_tle >> to_s3
```

## 9 · Propagación masiva con Spark + SGP4

Spark = “repartir el cálculo entre muchas CPUs”.
Cada ejecutor procesa unos cientos de TLE → llama SGP4 → devuelve filas
(sat_id, ts, x,y,z,vx,vy,vz).

| Tipo                   | Efecto principal | Término dominante            |
| ---------------------- | ---------------- | ---------------------------- |
| J₂ oblato              | Precesión Ω, ω   | Ecuación de arriba           |
| Arrastre               | Reducción a, e   | $\dot a \propto -B^{\ast}\rho$
 |
| Resonancia 24 h / 12 h | GEO, GPS         | Corrección al n              |
| Luni-solar medio       | Larga escala     | Pequeños ∆i, ∆e              |

Ventana útil: LEO ± 3 d, GEO ± 30 d.

```python
from sgp4.api import Satrec, jday
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType

schema = StructType([StructField(c, DoubleType()) for c in
                    ("x","y","z","vx","vy","vz")])

@pandas_udf(schema)
def propagate(l1, l2, ts):
    import pandas as pd
    out = []
    for a, b, t in zip(l1, l2, ts):
        sat = Satrec.twoline2rv(a, b)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second)
        e, r, v = sat.sgp4(jd, fr)
        out.append((*r, *v) if e == 0 else (None,)*6)
    return pd.DataFrame(out)
```
Resultado diario → ephemeris_bronze particionado por
sat_id y date.

## 10 · Conversión TEME → lat / lon / alt

1. ECI → ECEF
Rotación alrededor del eje Z por el ángulo de Greenwich
$$\theta_{\mathrm{GST}}(t)$$  
$$\mathbf{r}_{\mathrm{ECEF}} = R_{z}\bigl(\theta_{\mathrm{GST}}\bigr)\,\mathbf{r}_{\mathrm{ECI}}$$  
 
2. ECEF → geodésicas (WGS-84)
$$\lambda = \operatorname{arctan2}(y,\,x)$$  
$$p = \sqrt{x^{2} + y^{2}}$$  
$$\phi = \operatorname{arctan2}\bigl(z,\;p\,(1 - e^{2})\bigr)\quad(\text{iterativo})$$  
$$h = p\,\cos\phi \;-\; N(\phi)$$  
$$N(\phi) = \frac{a}{\sqrt{1 - e^{2}\,\sin^{2}\phi}}$$  

 
Giramos un globo terráqueo virtual hasta que el meridiano que pasaba
por Greenwich en aquel instante se alinea con el eje Y del ordenador.
Luego le clavamos un alfiler y leemos la etiqueta de latitud/longitud.

```python
from astropy.coordinates import TEME, ITRS
from astropy.time import Time
from astropy import units as u

def teme2geodetic(x, y, z, t_iso):
    teme = TEME([x, y, z] * u.km,
                obstime=Time(t_iso, scale='utc'))
    itrs = teme.transform_to(ITRS(obstime=teme.obstime))
    geo  = itrs.earth_location.to_geodetic()
    return geo.lat.deg, geo.lon.deg, geo.height.km
```
## Resumen de buenas prácticas
- Actualizar TLE < 48 h en LEO para evitar errores > 2 km.

- Propagar en clúster (SGP4 analítico ≈ 0.05 ms/epoch CPU).

- Rotar a WGS-84 inmediatamente si vas a unir con imágenes o GIS.

- Versionar el lago (raw → bronze → silver) para trazabilidad ML. 

## 11 · Gestión de datos geo‑espaciales

- Almacenamiento: PostGIS/TileDB para datos 4D (x,y,z,t)

- Procesado: GeoPandas / Rasterio para capas vectoriales/ráster

- Visualización: Kepler.gl / deck.gl para órbitas interactivas

```python
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

df = pd.read_parquet("ephemeris_silver/")
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(lon, lat) for lon, lat in zip(df.lon, df.lat)],
    crs="EPSG:4326"
)
gdf.to_file("s3://space-lake/geo/iss_tracks.geojson", driver="GeoJSON")
```

## C · Analítica & ML (Sem 11-13)
| Sem | Proyecto            | Dataset        | Métrica      |
| --- | ------------------- | -------------- | ------------ |
| 11  | LSTM BSTAR          | TLE + F10.7/Kp | MAE < 1 e-5  |
| 12  | Autoencoder SatNOGS | IQ frames      | Prec > 0.90  |
| 13  | Prophet drift error | ephem vs SPICE | RMSE < 10 km |


## 11 · Predicción de BSTAR con LSTM  

### 11.1 Background físico / matemático
**BSTAR** empaqueta el producto $\tfrac12\,C_D\,A/m$ con una densidad de referencia $\rho_0$.  

El semieje mayor decrece casi linealmente:  
$$\dot a(t)\simeq -\tfrac{2}{3}\,a\,B^{*}\,\rho_0\,e^{-\,(h - h_0)/H}$$  

**Índices exógenos**  
- *F10.7* (flujo UV/EUV) calienta la termosfera ⇒ ↑ρ.  
- *Kp* (geomagnético) aumenta ρ durante tormentas.  

**Serie temporal multivariada**  
$$\mathbf{x}(t) = \bigl[B^{*}(t),\;F10.7(t),\;Kp(t)\bigr]$$  
LSTM capta dependencias no lineales + lags > 1 órbita (≈ 90 min).

### 11.2 Casos de uso reales
| Rol | Decisión asistida por  \( \hat{B}^{\*}_{+24h} \) |
|-----|--------------------------------------------------|
| Control orbital LEO | Ajustar frecuencia de re-boost (drag compensation). |
| Predicción de rentrada | Estimar ventana de caída de sat. inactivo (± 12 h). |
| Gestión de catálogos | Priorizar TLE freshening si MAE↑. |

### 11.3 Ejemplo Python (Keras)
```python
import pandas as pd, numpy as np, tensorflow as tf
df = pd.read_parquet("raw_tle/*.parquet")          # epoch, bstar, f107, kp
df = df.set_index("epoch").resample("1H").interpolate()
X, y = [], []
WIN, HOR = 72, 24          # 72 h ventana → 24 h horizonte
for i in range(len(df)-WIN-HOR):
    X.append(df[['bstar','f107','kp']].iloc[i:i+WIN].values)
    y.append(df['bstar'].iloc[i+WIN+HOR])
X, y = np.array(X), np.array(y)
model = tf.keras.Sequential([
    tf.keras.layers.Input((WIN,3)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)])
model.compile('adam','mae')
model.fit(X, y, epochs=20, batch_size=256)
```
```python
import pandas as pd
import numpy as np
import tensorflow as tf

# Carga y preparación de datos
df = pd.read_parquet("raw_tle/*.parquet")  # epoch, bstar, f107, kp

df = df.set_index("epoch").resample("1H").interpolate()

# Construcción de ventanas deslizantes\ nX, y = [], []
WIN, HOR = 72, 24  # 72 h ventana → 24 h horizonte
for i in range(len(df) - WIN - HOR):
    X.append(df[['bstar', 'f107', 'kp']].iloc[i : i + WIN].values)
    y.append(df['bstar'].iloc[i + WIN + HOR])
X, y = np.array(X), np.array(y)

# Definición y entrenamiento del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input((WIN, 3)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mae')
model.fit(X, y, epochs=20, batch_size=256)
```
## 12 · Auto-detector de anomalías SatNOGS (Autoencoder)

| Escenario             | Acción disparada al detectar anomalía      |
| --------------------- | ------------------------------------------ |
| Ruido de portadora ↑  | Cambiar ancho de banda y re-trasmitir.     |
| Sobre-temperatura EPS | Conmutar a modo safe + reducir duty-cycle. |
| Glitch en IQ          | Volver a pedir frame (ARQ).                |

```python
import torch, torch.nn as nn, numpy as np
X = torch.tensor(df_norm.values, dtype=torch.float32)     # N×n
class AE(nn.Module):
    def __init__(self, n): super().__init__()
    self.enc = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 8))
    self.dec = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, n))
    def forward(self,x): return self.dec(self.enc(x))
ae = AE(X.shape[1]); opt = torch.optim.Adam(ae.parameters(), 1e-3)
for _ in range(200):
    opt.zero_grad(); loss = ((ae(X)-X)**2).mean(); loss.backward(); opt.step()
err = ((ae(X)-X)**2).sum(1).sqrt().detach().numpy()
anomaly_idx = np.where(err > err.mean() + 3*err.std())
```

```python
import torch
import torch.nn as nn
import numpy as np

# Conversión a tensor y normalización
X = torch.tensor(df_norm.values, dtype=torch.float32)  # tamaño N×n

# Definición del autoencoder
class AE(nn.Module):
    def __init__(self, n_features):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        self.dec = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, n_features)
        )

    def forward(self, x):
        return self.dec(self.enc(x))

# Instanciar y entrenar
ae = AE(X.shape[1])
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
for epoch in range(200):
    optimizer.zero_grad()
    recon = ae(X)
    loss = nn.functional.mse_loss(recon, X)
    loss.backward()
    optimizer.step()

# Cálculo de errores y detección de anomalías
err = torch.sqrt(((ae(X) - X) ** 2).sum(dim=1)).detach().numpy()
anomaly_idx = np.where(err > err.mean() + 3 * err.std())
```

## 13 · Pronóstico del error de efemérides (Prohet)

| Usuario                  | Decide                                               |
| ------------------------ | ---------------------------------------------------- |
| Catalog mgmt             | Cuándo refrescar TLE para mantener error < 5 km.     |
| Planificador de imágenes | Retrasar captura si predicho footprint ≫ pixel size. |

```python
from prophet import Prophet
dx = ephem.merge(spice, on='t', suffixes=('','_ref'))
dx['err'] = np.linalg.norm(dx[['x','y','z']].sub(dx[['x_ref','y_ref','z_ref']]).values, axis=1)
train = dx[['t','err']].rename(columns={'t':'ds','err':'y'})
m = Prophet(daily_seasonality=True, yearly_seasonality=False)
m.fit(train)
future = m.make_future_dataframe(periods=48, freq='H')
pred = m.predict(future)[['ds','yhat']]
```

```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Input((72, 3)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])
model.compile('adam', 'mae')
```
## D · Portfolio & DevOps (Sem 14-16)
| Proyecto              | Stack                        | KPI objetivo         | Descripción breve                                                                                      |
| --------------------- | ---------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------ |
| API “Where-is-my-sat” | Lambda · API GW · DynamoDB   | Error < 5 km @ 24 h  | REST service: NORAD ID → SGP4 → GeoJSON `{lat,lon,alt}`; caché TTL 10 s y CORS.                        |
| Space-Lakehouse       | S3 · Spark (EMR) · Athena    | Coste < 2 USD/mes    | ETL diario de TLE → efemérides 1 min → Parquet particionado; consultas SQL en Athena.                  |
| Conjunción alert ML   | PySpark · XGBoost · SNS      | FPR < 5 % @ TPR 90 % | Predice riesgo de conjunción 24 h con features orbitales; envía notificaciones SNS.                    |
| Re-entry predictor    | Poliastro · XGB · Step Fn    | RMSE ± 1 día         | Integra órbitas LEO + drag; modelo ML estima fecha/lugar de re-entrada; orquestado con Step Functions. |
| Planet-look AI        | SageMaker · CNN · Sentinel-2 | F1 incendios > 0.85  | Une footprint satélite + Sentinel-2; CNN detecta fuegos/inundaciones casi en tiempo real.              |
| Integración con APIs Satelitales        | Python·REST·Terraform | Lat/lon vs API < 1 km  | Ingestión automática de Planet, Sentinel Hub, SatNOGS; normalización CRS/timestamp              |



## Integración con APIs Satelitales Reales

- Proveedores: Planet (PSScene), Sentinel Hub (L1C/L2A), SatNOGS

- Pipeline:

    1. Autenticación (API-key/OAuth)

    2. Pull de metadatos (órbita, footprint)

    3. Descarga tile/scene con control de cuota

    4. Normalización CRS y timestamp

```python
resource "aws_lambda_function" "planet_ingest" {
  filename = "planet_ingest.zip"
  handler  = "ingest.handler"
  runtime  = "python3.12"
  environment { variables = { PLANET_API_KEY = var.planet_api_key } }
}
import requests

url = "https://api.planet.com/data/v1/quick-search"
headers = {"Content-Type":"application/json"}
payload = {"item_types":["PSScene4Band"],"filter":{...}}
resp = requests.post(url, auth=("api-key", PLANET_API_KEY), headers=headers, json=payload)
items = resp.json()["features"]
for feat in items:
    rec = {"id":feat["id"],"acquired":feat["properties"]["acquired"],"footprint":feat["geometry"]}
    # guardar en S3/Dynamo
```


## Bibliografía esencial
- Vallado — Fundamentals of Astrodynamics & Applications

- NASA — Basics of Space Flight (Ch 4)

- CCSDS 502.0-B-3 — Orbit Data Messages

- ESA OPS Blog — Lambert solver (Izzo)

- CelesTrak — Guide to TLE

- SatNOGS — Wiki & API

## Ejemplo Spark — ISS ± 48 h

```python
import datetime as dt, requests, pandas as pd
from sgp4.api import Satrec, jday
from pyspark.sql import SparkSession, functions as F

txt = requests.get('https://celestrak.org/NORAD/elements/stations.txt').text
L1, L2 = [l for l in txt.splitlines() if l.startswith(('1 ', '2 '))][:2]
sat = Satrec.twoline2rv(L1, L2)

base = dt.datetime.utcnow()
rows = []
for k in range(-48*4, 48*4 + 1):
    t = base + dt.timedelta(minutes=15*k)
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
    e, r, v = sat.sgp4(jd, fr)
    if e == 0:
        rows.append([t.isoformat(), *r, *v])

df = pd.DataFrame(rows, columns=['utc', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

spark = SparkSession.builder.appName('iss').getOrCreate()
(spark.createDataFrame(df.astype(str))
      .withColumn('date', F.to_date('utc'))
      .write.mode('overwrite')
      .partitionBy('date')
      .parquet('s3://YOUR_BUCKET/iss_ephem/'))
spark.stop()
```
