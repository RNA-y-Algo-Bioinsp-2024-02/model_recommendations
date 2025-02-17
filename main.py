from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Variables globales para compartir datos entre funciones
df = None
interpreter = None
combined_embeddings = None
id_to_name = None
name_to_id = None
prod_weights = None
main_weights = None
sub_weights = None
num_products = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, interpreter, combined_embeddings, id_to_name, name_to_id
    global prod_weights, main_weights, sub_weights, num_products

    print("Iniciando startup...")
    try:
        # 1. Cargar CSV y preprocesar datos
        df = pd.read_csv("productos.csv")
        df['discount_price'] = df['discount_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
        df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)

        # Codificar variables
        name_enc = LabelEncoder()
        df['name_encoded'] = name_enc.fit_transform(df['name'])

        main_cat_enc = LabelEncoder()
        df['main_category_encoded'] = main_cat_enc.fit_transform(df['main_category'])

        sub_cat_enc = LabelEncoder()
        df['sub_category_encoded'] = sub_cat_enc.fit_transform(df['sub_category'])
        print("Variables codificadas.")

        # Normalización de precios
        df['discount_price'] = df['discount_price'] / df['discount_price'].max()
        df['actual_price'] = df['actual_price'] / df['actual_price'].max()
        print("Preprocesamiento completado.")

        # Crear mapeos para convertir entre nombre e ID
        temp_id_to_name = df[['name_encoded', 'name']].drop_duplicates().set_index('name_encoded')['name'].to_dict()
        id_to_name = {int(k): v for k, v in temp_id_to_name.items()}
        name_to_id = {v: int(k) for k, v in id_to_name.items()}
        print("Mapeos creados.")

        # 2. Cargar el modelo TFLite
        interpreter = tf.lite.Interpreter(model_path="recomendacion.tflite")
        interpreter.allocate_tensors()
        print("Modelo TFLite cargado y tensores asignados.")

        # Función para extraer los pesos del embedding de manera más precisa
        def get_embedding_weights(layer_name):
            candidates = []
            for tensor_detail in interpreter.get_tensor_details():
                # Filtrar por el nombre y descartar operaciones internas (por ejemplo, strided_slice)
                if layer_name in tensor_detail["name"] and "strided_slice" not in tensor_detail["name"]:
                    tensor = interpreter.get_tensor(tensor_detail["index"])
                    # Suponemos que la matriz de embedding tiene al menos dos dimensiones
                    if len(tensor.shape) >= 2:
                        candidates.append((tensor_detail, tensor))
            if not candidates:
                raise ValueError(f"No se encontró tensor adecuado para la capa '{layer_name}'")
            # Seleccionar el primer candidato (o podrías elegir el que cumpla mejor el criterio)
            selected_detail, selected_tensor = candidates[0]
            print(f"Tensor para '{layer_name}' seleccionado: {selected_detail['name']} con shape {selected_tensor.shape}")
            return selected_tensor

        # 3. Extraer pesos de los embeddings
        prod_weights = get_embedding_weights("product_embedding")
        main_weights = get_embedding_weights("main_cat_embedding")
        sub_weights = get_embedding_weights("sub_cat_embedding")
        num_products = prod_weights.shape[0]
        print("Pesos de embeddings extraídos.")

        # 4. Calcular los vectores combinados para cada producto
        def get_vector_combinado(product_id: int):
            row = df[df['name_encoded'] == product_id].iloc[0]
            main_cat = int(row['main_category_encoded'])
            sub_cat = int(row['sub_category_encoded'])
            vec_prod = prod_weights[product_id]
            vec_main = main_weights[main_cat]
            vec_sub = sub_weights[sub_cat]
            return np.concatenate([vec_prod, vec_main, vec_sub])
        
        combined_embeddings = np.array([get_vector_combinado(pid) for pid in range(num_products)])
        print("Vectores combinados calculados.")

    except Exception as e:
        print("Error en startup:", e)
        raise e

    print("Startup completado. La API ya está lista.")
    yield
    print("Shutdown de la API.")

# Definir la aplicación FastAPI usando el lifespan event handler
app = FastAPI(lifespan=lifespan, title="API de Recomendaciones de Productos")

# Agregar middleware de CORS para permitir solicitudes de cualquier origen
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def recomendar_productos_combinados(product_id: int, top_n: int = 5):
    """
    Recomienda productos basados en la similitud del vector combinado.
    """
    if product_id < 0 or product_id >= num_products:
        return []
    query_vec = combined_embeddings[product_id].reshape(1, -1)
    similitudes = cosine_similarity(query_vec, combined_embeddings)[0]
    indices_similares = np.argsort(-similitudes)
    # Excluir el propio producto
    indices_similares = [i for i in indices_similares if i != product_id]
    return [int(x) for x in indices_similares[:top_n]]

@app.get("/recomendaciones")
def get_recomendaciones(product_name: str, top_n: int = 5):
    """
    Endpoint que, dado el nombre de un producto, devuelve recomendaciones.
    Ejemplo:
      GET /recomendaciones?product_name=Camisa%20Roja&top_n=5
    """
    if product_name not in name_to_id:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    product_id = name_to_id[product_name]
    recomendados_ids = recomendar_productos_combinados(product_id, top_n)
    recomendaciones = [
        {"id": int(pid), "name": id_to_name.get(int(pid), f"ID {pid}")}
        for pid in recomendados_ids
    ]
    return {"producto": product_name, "recomendaciones": recomendaciones}
