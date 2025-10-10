Mortalidad Materna - Dash App

Estructura:
- app.py: aplicación Dash
- requirements.txt: dependencias (incluye gunicorn)
- Procfile: para despliegue en Heroku (web: gunicorn app:server)

Instrucciones rápidas:
1. Crear entorno virtual e instalar dependencias:
   python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt
2. Ejecutar localmente:
   set PORT=8050; python app.py

Notas:
- La app usa el archivo de datos en ../Geo/Mortalidad_Materna.xlsx y el shapefile en jose/coordenadas/COLOMBIA/COLOMBIA.shp
- Si el shapefile no contiene población, la RMM mostrará un mensaje indicando que se requiere población para un cálculo fiable.
