import pandas as pd
from fastapi import FastAPI, HTTPException
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
generos_contenido = pd.read_csv('dfx_steam_games.csv')
usuarios_tiempo_juego = pd.read_csv('dfx_user_items.csv')

app = FastAPI()

@app.get('/playtimeGenre/{genero}')

def playtimeGenre(genero: str):
    # Filtra los datos de género específico
    datos_genero = generos_contenido[generos_contenido['Genero'] == genero]
    
    if datos_genero.empty:
        return {"Mensaje": "Género no encontrado"}

   
    datos_completos = usuarios_tiempo_juego.merge(datos_genero, left_on='item_id', right_on='Id')

    # Agrupa los datos por año y suma el tiempo de juego total
    datos_agrupados = datos_completos.groupby('Lanzamiento')['tiempo_Juego_total'].sum().reset_index()

    # Encuentra el año con más horas jugadas
    max_hours_year = datos_agrupados[datos_agrupados['tiempo_Juego_total'] == datos_agrupados['tiempo_Juego_total'].max()]

    if not max_hours_year.empty:
        max_year = max_hours_year.iloc[0]['Lanzamiento']
        return {"El  género":  genero},{"Año con más horas jugadas ":  max_year}

    return {"Mensaje": "No se encontraron datos de tiempo de juego para el género especificado"}

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.get('/userforgenre/{genero}')
def userforgenre(genero: str):
    # Filtra los datos de género específico
    datos_genero = generos_contenido[generos_contenido['Genero'] == genero]

    if datos_genero.empty:
        return {"Mensaje": "Género no encontrado"}

    # Combina los datos de usuarios y tiempo de juego con los datos de género
    
    datos_completos = usuarios_tiempo_juego.merge(datos_genero, left_on='item_id', right_on='Id')

    if datos_completos.empty:
        return {"Mensaje": "No se encontraron datos de tiempo de juego para el género especificado"}

    # Agrupa los datos por usuario y año y suma el tiempo de juego total
    datos_agrupados = datos_completos.groupby(['user_id', 'Lanzamiento'])['tiempo_Juego_total'].sum().reset_index()

    # Encuentra el usuario con más horas jugadas para el género
    max_hours_user = datos_agrupados[datos_agrupados.groupby('user_id')['tiempo_Juego_total'].transform('max') == datos_agrupados['tiempo_Juego_total']]
    if not max_hours_user.empty:
        max_user = max_hours_user.iloc[0]['user_id']

        # Calcula la acumulación de horas jugadas por año
        horas_por_anio = datos_agrupados.groupby('Lanzamiento')['tiempo_Juego_total'].sum().reset_index()
        horas_por_anio = horas_por_anio.to_dict(orient='records')

        return {
            "Usuario con más horas jugadas para género": max_user,
           "Horas jugadas": [{"Año": int(row['Lanzamiento']), "Horas": int(row['tiempo_Juego_total'])} for row in horas_por_anio]
           

        }

    return {"Mensaje": "No se encontraron datos de tiempo de juego para el género especificado"}



#/////////////////////////////////////////////////////////////////////////////////////////////////////////////
dataframe1 = pd.read_csv('dfx_sentimiento.csv')
dataframe2 = pd.read_csv('dfx_steam_games.csv')

@app.get('/usersrecommend/{anio}')

def usersrecommend(anio: int):
    # Filtra el DataFrame1 para obtener revisiones recomendadas y comentarios positivos/neutrales
    filtro_df1 = dataframe1[(dataframe1['Recomendacion'] == True)]
    dataframe2['Lanzamiento'] = dataframe2['Lanzamiento'].astype(int)
    
    # Filtra el DataFrame2 para el año dado
    filtro_df2 = dataframe2[dataframe2['Lanzamiento'] == anio]
    

    if filtro_df1.empty or filtro_df2.empty:
        return {"Mensaje": "No se encontraron juegos recomendados para el año especificado"}

    # Combina los DataFrames en función de 'Item_id' y cuenta las revisiones recomendadas
    juegos_recomendados = filtro_df1.merge(filtro_df2, left_on='Item_id', right_on='Id')
    top_juegos = juegos_recomendados.groupby('Nombre_del_contenido')['Recomendacion'].count().reset_index()

    # Ordena los juegos por la cantidad de recomendaciones en orden descendente y toma los 3 primeros
    top_juegos = top_juegos.sort_values(by='Recomendacion', ascending=False).head(3)

    # Formatea el resultado en el formato deseado
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(top_juegos['Nombre_del_contenido'])]

    return resultado

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.get('/usersnotrecommend/{anio}')
def usersnotrecommend(anio: int):
    # Filtra el DataFrame1 para obtener revisiones no recomendadas y comentarios negativos
    filtro_df1 = dataframe1[(dataframe1['Recomendacion'] == False)]

    # Filtra el DataFrame2 para el año dado
    filtro_df2 = dataframe2[dataframe2['Lanzamiento'] == anio]

    if filtro_df1.empty or filtro_df2.empty:
        return {"Mensaje": "No se encontraron juegos no recomendados para el año especificado"}

    # Combina los DataFrames en función de 'Item_id' y cuenta las revisiones no recomendadas
    juegos_no_recomendados = filtro_df1.merge(filtro_df2, left_on='Item_id', right_on='Id')
    top_juegos = juegos_no_recomendados.groupby('Nombre_del_contenido')['Recomendacion'].count().reset_index()

    # Ordena los juegos por la cantidad de no recomendaciones en orden descendente y toma los 3 primeros
    top_juegos = top_juegos.sort_values(by='Recomendacion', ascending=False).head(3)

    # Formatea el resultado en el formato deseado
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(top_juegos['Nombre_del_contenido'])]

    return resultado

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////

