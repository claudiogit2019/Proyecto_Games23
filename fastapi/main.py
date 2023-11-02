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
        # Filtro
    datos_genero = generos_contenido[generos_contenido['Genero'] == genero]
    
    if datos_genero.empty:
        return {"Mensaje": "Género no encontrado"}

   
    datos_completos = usuarios_tiempo_juego.merge(datos_genero, left_on='item_id', right_on='Id')

        # Agrupar
    datos_agrupados = datos_completos.groupby('Lanzamiento')['tiempo_Juego_total'].sum().reset_index()

        # Busqueda
    max_hours_year = datos_agrupados[datos_agrupados['tiempo_Juego_total'] == datos_agrupados['tiempo_Juego_total'].max()]

    if not max_hours_year.empty:
        max_year = max_hours_year.iloc[0]['Lanzamiento']
        return {"El  género":  genero},{"Año con más horas jugadas ":  max_year}

    return {"Mensaje": "No se encontraron datos de tiempo de juego para el género especificado"}

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.get('/userforgenre/{genero}')
def userforgenre(genero: str):
        # Filtro
    datos_genero = generos_contenido[generos_contenido['Genero'] == genero]

    if datos_genero.empty:
        return {"Mensaje": "Género no encontrado"}

        # Combinacion
    
    datos_completos = usuarios_tiempo_juego.merge(datos_genero, left_on='item_id', right_on='Id')

    if datos_completos.empty:
        return {"Mensaje": "No se encontraron datos de tiempo de juego para el género especificado"}

        # Agrupar
    datos_agrupados = datos_completos.groupby(['user_id', 'Lanzamiento'])['tiempo_Juego_total'].sum().reset_index()

        # Encuentra el usuario en relacion  tiempo de Juego
    max_hours_user = datos_agrupados[datos_agrupados.groupby('user_id')['tiempo_Juego_total'].transform('max') == datos_agrupados['tiempo_Juego_total']]
    if not max_hours_user.empty:
        max_user = max_hours_user.iloc[0]['user_id']

        # Calculo
        horas_por_anio = datos_agrupados.groupby('Lanzamiento')['tiempo_Juego_total'].sum().reset_index()
        horas_por_anio = horas_por_anio.to_dict(orient='records')

        return {
            "Usuario con más horas jugadas para género": max_user,
           "Horas jugadas": [{"Año": int(row['Lanzamiento']), "Horas": int(row['tiempo_Juego_total'])} for row in horas_por_anio]
           

        }

    return {"Mensaje": "No se encontraron datos de tiempo de juego para el género especificado"}



#/////////////////////////////////////////////////////////////////////////////////////////////////////////////
dataframe = pd.read_csv('dfx_sentimiento.csv')

@app.get('/usersrecommend/{anio}')
def usersrecommend(anio: int):
        # Filtro
    filtro_df = dataframe[(dataframe['Recomendacion'] == True) & (dataframe['Lanzamiento'] == anio)]

    if filtro_df.empty:
        return {"Mensaje": f"No se encontraron juegos recomendados para el año {anio}"}

        # Agrupar los juegos 
    top_juegos = filtro_df.groupby('Nombre_del_contenido')['Recomendacion'].count().reset_index()

        # Ordenar los juegos 
    top_juegos = top_juegos.sort_values(by='Recomendacion', ascending=False).head(3)

        # Salida
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(top_juegos['Nombre_del_contenido'])]

    return resultado


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.get('/usersnotrecommend/{anio}')
def usersrecommend(anio: int):
        # Filtro
    filtro_df = dataframe[(dataframe['Recomendacion'] == False) & (dataframe['Lanzamiento'] == anio)]

    if filtro_df.empty:
        return {"Mensaje": f"No se encontraron juegos recomendados para el año {anio}"}

        # Agrupar los juegos 
    top_juegos = filtro_df.groupby('Nombre_del_contenido')['Recomendacion'].count().reset_index()

        # Ordenar los juegos 
    top_juegos = top_juegos.sort_values(by='Recomendacion', ascending=False).head(3)

        # Formatear el resultado en el formato deseado
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(top_juegos['Nombre_del_contenido'])]

    return resultado
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////

df = pd.read_csv('dfx_sentimiento.csv')

@app.get("/sentiment_analysis/{anio}")

def get_sentiment_analysis(año: int):
    
        # Filtro segun lanzamiento
    filtered_df = df[df['Lanzamiento'] == año]

        # Calculo la cantidad de registros 
    sentiment_counts = filtered_df['Sentimiento'].value_counts()

        # Salida
    result = {key: value for key, value in zip(sentiment_counts.index, sentiment_counts)}
    
    return result

#///////////////////////////////////////////////////////////////////////////////////////////////////
