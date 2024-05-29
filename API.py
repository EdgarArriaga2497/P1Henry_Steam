import streamlit as st
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

st.title('Funcion 1')

# Cargar los datos
df_games = pd.read_parquet('dfgamesrecomendacionAPI.parquet', engine='pyarrow')

# Añadir columna 'is_free' y convertir 'developer' a string
df_games['is_free'] = df_games['price'] == 0.0
df_games['developer'] = df_games['developer'].astype(str)

# Función para obtener los datos requeridos
def get_developer_data(developer: str):
    df_dev = df_games[df_games['developer'].str.contains(developer, case=False, na=False)]
    result = df_dev.groupby('release_date').agg(
        cantidad_items=('item_id', 'count'),
        contenido_free=('is_free', lambda x: 100 * x.sum() / x.count())
    ).reset_index().rename(columns={'release_date': 'Año', 'cantidad_items': 'Cantidad de Items', 'contenido_free': 'Contenido Free'})
    return result

st.markdown('''
+ def **developer( *`desarrollador` : str* )**:
    `Cantidad` de items y `porcentaje` de contenido Free por año según empresa desarrolladora. 
Ejemplo de retorno:

| Año  | Cantidad de Items | Contenido Free  |
|------|-------------------|------------------|
| 2023 | 50                | 27%              |
| 2022 | 45                | 25%              |
| xxxx | xx                | xx%              |
''')

# Input para el nombre del desarrollador
developer_name = st.text_input('Introduce el nombre del desarrollador Ejemplo (XLGAMES):')

# Mostrar resultados cuando se proporciona un nombre de desarrollador
if developer_name:
    developer_data = get_developer_data(developer_name)
    st.write(developer_data)

st.title('Funcion 2')
st.markdown('''
+ def **userdata( *`User_id` : str* )**:
    Debe devolver `cantidad` de dinero gastado por el usuario, el `porcentaje` de recomendación en base a reviews.recommend y `cantidad de items`.

Ejemplo de retorno: {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}
''')
# Cargar los datos
df_games = pd.read_parquet('dfgamesrecomendacionAPI.parquet', engine='pyarrow')
df_user_reviews = pd.read_parquet('dfreviewsAPI.parquet', engine='pyarrow')

# Convertir las columnas 'item_id' a tipo str
df_games['item_id'] = df_games['item_id'].astype('str')
df_user_reviews['item_id'] = df_user_reviews['item_id'].astype('str')

# Función para obtener los datos del usuario
def userdata(user_id: str) -> dict:
    # Unir los DataFrames en base a la columna 'item_id'
    merged_df = pd.merge(df_user_reviews, df_games, on='item_id')
    
    # Filtrar las filas relevantes para el 'user_id' proporcionado
    user_data = merged_df[merged_df['user_id'] == user_id]
    
    # Calcular la cantidad de dinero gastado
    total_money_spent = user_data['price'].sum()
    
    # Calcular el porcentaje de recomendación
    total_reviews = user_data.shape[0]
    recommendation_percentage = (user_data['recommend'].sum() / total_reviews) * 100 if total_reviews > 0 else 0
    
    # Calcular la cantidad de items
    total_items = user_data['item_id'].nunique()
    
    # Crear el diccionario de retorno
    result = {
        "Usuario": user_id,
        "Dinero gastado": f"{total_money_spent:.2f} USD",
        "% de recomendación": f"{recommendation_percentage:.2f}%",
        "cantidad de items": total_items
    }
    
    return result

# Input para el nombre del usuario
user_id = st.text_input('Introduce el ID del usuario Ejemplo (evcentric):')

# Mostrar resultados cuando se proporciona un ID de usuario
if user_id:
    user_data = userdata(user_id)
    st.write(user_data)

st.title('Funcion 3')
st.markdown('''
+ def **UserForGenre( *`genero` : str* )**:
    Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.

Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf,
			     "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
''')

ruta_archivo = 'tabla_user2.parquet'
tabla_user2 = pd.read_parquet(ruta_archivo, engine='pyarrow')

# Función para obtener el usuario con más horas jugadas para un género específico
def UserForGenre(genero):
    try:
        usuario = tabla_user2[tabla_user2["genres"] == genero]["user_id"].iloc[0]  # obtengo usuario
        historial = tabla_user2[(tabla_user2['user_id'] == usuario) & (tabla_user2['genres'] == genero)]  # filtro por el género y usuario
        historial2 = historial[['release_date', 'Horas jugadas']].copy()  # me quedo con las columnas necesarias
        historial3 = historial2.to_dict(orient="records")
        return {"Usuario": usuario, "con más horas jugadas para": genero, "Historial acumulado": historial3}
    except IndexError:
        return {"Error": "No se encontró el género especificado."}

# Input para el género
genero = st.text_input('Introduce el género del juego Ejemplo (Action):')

# Mostrar resultados cuando se proporciona un género
if genero:
    resultado = UserForGenre(genero)
    st.write(resultado)


st.title('Funcion 4')
st.markdown('''
+ def **best_developer_year( *`año` : int* )**:
   Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
''')
def best_developer_year(year: int) -> list:

    df_games = pd.read_parquet(
    'dfgamesrecomendacionAPI.parquet', 
    columns=['item_id', 'developer'],
    engine='pyarrow'
)
    df_user_reviews = pd.read_parquet(
    'dfreviewsAPI.parquet', 
    columns=['item_id', 'year', 'recommend', 'sentiment_analisis'],
    engine='pyarrow'
)
    # Convertir las columnas 'item_id' a tipo int64
    df_games['item_id'] = df_games['item_id'].astype('int64')
    df_user_reviews['item_id'] = df_user_reviews['item_id'].astype('int64')
    df_user_reviews['year'] = df_user_reviews['year'].astype('int64')  # Convertir la columna year a entero
    
    # Unir los DataFrames en base a la columna 'item_id'
    merged_df = pd.merge(df_user_reviews, df_games, on='item_id')
    
    # Filtrar las filas relevantes para el año proporcionado
    year_data = merged_df[merged_df['year'] == year]
    
    # Filtrar las reviews recomendadas y con comentarios positivos
    positive_recommendations = year_data[(year_data['recommend'] == True) & (year_data['sentiment_analisis'] > 0)]
    
    # Contar las recomendaciones por cada desarrollador
    developer_recommendations = positive_recommendations['developer'].value_counts()
    
    # Seleccionar los top 3 desarrolladores
    top_3_developers = developer_recommendations.head(3).index.tolist()
    
    # Crear la lista de resultados
    result = []
    if len(top_3_developers) > 0:
        result.append({"Puesto 1": top_3_developers[0]})
    if len(top_3_developers) > 1:
        result.append({"Puesto 2": top_3_developers[1]})
    if len(top_3_developers) > 2:
        result.append({"Puesto 3": top_3_developers[2]})
    
    return result

# Input para el año
año = st.text_input('Introduce el año: Ejemplo (2014):')

# Convertir el input a entero y mostrar resultados cuando se proporciona un año
if año:
    try:
        año = int(año)  # Convertir a entero
        resultado = best_developer_year(año)
        st.write(resultado)
    except ValueError:
        st.write("Por favor, introduce un año válido.")

st.title('Funcion 5')
st.markdown('''
+ def **developer_reviews_analysis( *`desarrolladora` : str* )**:
    Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo. 

Ejemplo de retorno: {'Valve' : [Negative = 182, Positive = 278]}
''')

def developer_reviews_analysis(developer: str) -> dict:
    # Leer los DataFrames desde los archivos parquet
    df_games = pd.read_parquet(
        'dfgamesrecomendacionAPI.parquet', 
        columns=['item_id', 'developer'],
        engine='pyarrow'
    )
    df_user_reviews = pd.read_parquet(
        'dfreviewsAPI.parquet', 
        columns=['user_id', 'item_id', 'sentiment_analisis'],
        engine='pyarrow'
    )
    
    # Convertir las columnas 'item_id' a tipo int64
    df_games['item_id'] = df_games['item_id'].astype('int64')
    df_user_reviews['item_id'] = df_user_reviews['item_id'].astype('int64')
    
    # Unir los DataFrames en base a la columna 'item_id'
    merged_df = pd.merge(df_user_reviews, df_games, on='item_id')
    
    # Filtrar las filas relevantes para el desarrollador proporcionado
    developer_data = merged_df[merged_df['developer'] == developer]
    
    # Contar las reseñas con análisis de sentimiento positivo y negativo
    positive_reviews = developer_data[developer_data['sentiment_analisis'] > 0].shape[0]
    negative_reviews = developer_data[developer_data['sentiment_analisis'] <= 0].shape[0]
    
    # Crear el diccionario de retorno
    result = {developer: {'Negative': negative_reviews, 'Positive': positive_reviews}}
    
    return result

# Input para el desarrollador
developer = st.text_input('Introduce el nombre del desarrollador: Ejemplo(Ubisoft)')

# Mostrar resultados cuando se proporciona un desarrollador
if developer:
    resultado = developer_reviews_analysis(developer)
    st.write(resultado)



st.title('Modelo de recomendaciones')


ruta_archivo = 'modelo_render.parquet'
modelo_render = pd.read_parquet(ruta_archivo, engine='pyarrow')
def recomendacion_juego(item_id): #este si funciono tal cual
    
    game = modelo_render[modelo_render['item_id'] == item_id]
    
    if game.empty:
        return("El juego '{item_id}' no posee registros.")
    
    # Obtiene el índice del juego dado
    idx = game.index[0]

    # Toma una muestra aleatoria del DataFrame df_games
    sample_size = 100  # Define el tamaño de la muestra (ajusta según sea necesario)
    df_sample = modelo_render.sample(n=sample_size, random_state=42)  # Ajusta la semilla aleatoria según sea necesario

    # Calcula la similitud de contenido solo para el juego dado y la muestra
    sim_scores = cosine_similarity([modelo_render.iloc[idx, 3:]], df_sample.iloc[:, 3:])

    # Obtiene las puntuaciones de similitud del juego dado con otros juegos
    sim_scores = sim_scores[0]

    # Ordena los juegos por similitud en orden descendente
    similar_games = [(i, sim_scores[i]) for i in range(len(sim_scores)) if i != idx]
    similar_games = sorted(similar_games, key=lambda x: x[1], reverse=True)

    # Obtiene los 5 juegos más similares
    similar_game_indices = [i[0] for i in similar_games[:5]]

    # Lista de juegos similares (solo nombres)
    similar_game_names = df_sample['app_name'].iloc[similar_game_indices].tolist()

    return {"similar_games": similar_game_names}

# Widget de entrada para el ID del juego
item_id = st.text_input("Ingrese el ID del juego: Ejemplo (2360 o 20)")

# Botón para obtener recomendaciones de juegos
if st.button("Obtener recomendaciones de juegos"):
    if item_id:
        try:
            # Intenta convertir a entero, en caso de que el ID deba ser numérico
            item_id = int(item_id)
            resultado = recomendacion_juego(item_id)
            st.write(resultado)
        except ValueError:
            st.error("Por favor, ingrese un ID de juego válido.")
    else:
        st.error("Por favor, ingrese el ID de un juego.")



