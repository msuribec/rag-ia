import base64
import io
import os
import time
from typing import List, Tuple

import numpy as np
import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings


APP_TITLE = "RAG Playground - Taller 03"
DEFAULT_STANDARD_CHUNK_SIZE = 600
DEFAULT_STANDARD_TOP_K = 4
DEFAULT_STANDARD_TEMPERATURE = 0.2
DEFAULT_VISION_MODEL = "llama-3.2-11b-vision-preview"

PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
]

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    denom = (np.linalg.norm(a_np) * np.linalg.norm(b_np)) + 1e-10
    return float(np.dot(a_np, b_np) / denom)


def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


@st.cache_data(show_spinner=False, ttl=300)
def list_groq_models(api_key: str) -> List[str]:
    client = groq_client(api_key)
    models = client.models.list()
    ids = [model.id for model in models.data]
    return sorted(set(ids))


def extract_text_from_image(file_bytes: bytes, api_key: str, model_id: str) -> str:
    image_b64 = base64.b64encode(file_bytes).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extrae todo el texto de la imagen. Devuelve solo el texto, sin comentarios.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }
    ]
    client = groq_client(api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def build_vectorstore(text: str, chunk_size: int) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(200, max(0, int(chunk_size * 0.2))),
    )
    docs = splitter.create_documents([text])
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)


def answer_with_groq(
    api_key: str,
    model: str,
    question: str,
    temperature: float,
    context: str | None = None,
    force_no_se: bool = False,
) -> Tuple[str, float]:
    system_prompt = None
    if context is not None:
        system_prompt = (
            "Eres un asistente que responde SOLO con el contexto dado. "
            "Si la respuesta no esta en el contexto, di exactamente: No se."
            if force_no_se
            else "Responde usando solo el contexto dado. Si falta informacion, dilo."
        )
        user_content = f"Contexto:\n{context}\n\nPregunta:\n{question}"
    else:
        user_content = question

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    client = groq_client(api_key)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    elapsed = time.perf_counter() - start
    return response.choices[0].message.content.strip(), elapsed


def format_seconds(value: float) -> str:
    return f"{value:.2f} s"


def get_vectorstore_cached(text: str, chunk_size: int) -> FAISS:
    cache = st.session_state.setdefault("vectorstores", {})
    key = f"{hash(text)}::{chunk_size}"
    if key not in cache:
        cache[key] = build_vectorstore(text, chunk_size)
    return cache[key]


def get_default_model(available_models: List[str]) -> str:
    for model in PREFERRED_MODELS:
        if model in available_models:
            return model
    return available_models[0] if available_models else FALLBACK_MODELS[0]


with st.sidebar:
    st.header("Configuracion")
    api_key_input = st.text_input("Groq API Key", type="password")
    api_key = api_key_input.strip() if api_key_input else os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        st.warning("Ingresa tu Groq API Key para habilitar modelos y consultas.")
        available_models = FALLBACK_MODELS
    else:
        try:
            available_models = list_groq_models(api_key)
            if not available_models:
                st.warning("Groq no devolvio modelos; usando lista de respaldo.")
                available_models = FALLBACK_MODELS
            else:
                st.success(f"Modelos disponibles: {len(available_models)}")
        except Exception as exc:
            st.error(f"No se pudo consultar modelos de Groq: {exc}")
            available_models = FALLBACK_MODELS

    default_model = get_default_model(available_models)
    model = st.selectbox("Model Select", available_models, index=available_models.index(default_model))
    with st.expander("Ver modelos disponibles"):
        st.code("\n".join(available_models))
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_STANDARD_TEMPERATURE, 0.05)
    chunk_size = st.slider("Chunk Size (aprox. caracteres)", 20, 2000, DEFAULT_STANDARD_CHUNK_SIZE, 10)
    top_k = st.slider("Top-K", 1, 10, DEFAULT_STANDARD_TOP_K, 1)
    force_no_se = st.checkbox('Forzar "No se" si no hay evidencia', value=False)

    vision_model = DEFAULT_VISION_MODEL
    if api_key and available_models and DEFAULT_VISION_MODEL not in available_models:
        st.info(
            f"Modelo vision '{DEFAULT_VISION_MODEL}' no esta disponible en tu cuenta. "
            "Si subes imagenes, ajusta este modelo manualmente."
        )
        vision_model = st.text_input("Modelo OCR (vision)", value=DEFAULT_VISION_MODEL)


uploaded = st.file_uploader("Sube un PDF o una imagen con texto", type=["pdf", "png", "jpg", "jpeg"])
question = st.text_input("Pregunta", placeholder="Escribe tu pregunta sobre el documento")

extracted_text = ""
if uploaded:
    file_bytes = uploaded.getvalue()
    is_pdf = uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf")
    if is_pdf:
        with st.spinner("Extrayendo texto del PDF..."):
            try:
                extracted_text = read_pdf(file_bytes)
            except Exception as exc:
                st.error(f"No se pudo leer el PDF: {exc}")
    else:
        if not api_key:
            st.error("Para OCR necesitas ingresar la Groq API Key.")
        else:
            with st.spinner("Extrayendo texto con modelo vision..."):
                try:
                    extracted_text = extract_text_from_image(file_bytes, api_key, vision_model)
                except Exception as exc:
                    st.error(f"No se pudo extraer texto de la imagen: {exc}")

if extracted_text:
    st.caption(f"Texto cargado. Longitud aproximada: {len(extracted_text)} caracteres.")
    with st.expander("Ver texto extraido"):
        st.write(extracted_text)


if api_key and extracted_text and question:
    embeddings = get_embeddings()

    with st.spinner("Preparando indices..."):
        vs_standard = get_vectorstore_cached(extracted_text, DEFAULT_STANDARD_CHUNK_SIZE)
        vs_custom = get_vectorstore_cached(extracted_text, chunk_size)

    def run_rag(
        vectorstore: FAISS,
        use_custom_prompt: bool,
        k_value: int,
        temp_value: float,
    ) -> Tuple[str, float, float]:
        docs = vectorstore.similarity_search(question, k=k_value)
        context = "\n\n".join([f"[Chunk {idx+1}]\n{doc.page_content}" for idx, doc in enumerate(docs)])
        answer, elapsed = answer_with_groq(
            api_key=api_key,
            model=model,
            question=question,
            temperature=temp_value,
            context=context,
            force_no_se=use_custom_prompt,
        )
        if docs:
            query_vec = embeddings.embed_query(question)
            doc_vec = embeddings.embed_documents([docs[0].page_content])[0]
            similarity = cosine_similarity(query_vec, doc_vec)
        else:
            similarity = 0.0
        return answer, elapsed, similarity

    cols = st.columns(3)

    with cols[0]:
        st.subheader("LLM Simple")
        with st.spinner("Consultando LLM..."):
            try:
                answer_simple, time_simple = answer_with_groq(
                    api_key=api_key,
                    model=model,
                    question=question,
                    temperature=temperature,
                )
                st.write(answer_simple)
                st.caption(f"Tiempo de respuesta: {format_seconds(time_simple)}")
                st.caption("Similitud de Coseno: N/A")
            except Exception as exc:
                st.error(f"Error en LLM Simple: {exc}")

    with cols[1]:
        st.subheader("RAG Estandar")
        with st.spinner("Ejecutando RAG estandar..."):
            try:
                answer_std, time_std, sim_std = run_rag(
                    vs_standard,
                    use_custom_prompt=False,
                    k_value=DEFAULT_STANDARD_TOP_K,
                    temp_value=DEFAULT_STANDARD_TEMPERATURE,
                )
                st.write(answer_std)
                st.caption(f"Tiempo de respuesta: {format_seconds(time_std)}")
                st.caption(f"Similitud de Coseno: {sim_std:.3f}")
            except Exception as exc:
                st.error(f"Error en RAG estandar: {exc}")

    with cols[2]:
        st.subheader("RAG Optimizado")
        with st.spinner("Ejecutando RAG optimizado..."):
            try:
                answer_opt, time_opt, sim_opt = run_rag(
                    vs_custom,
                    use_custom_prompt=force_no_se,
                    k_value=top_k,
                    temp_value=temperature,
                )
                st.write(answer_opt)
                st.caption(f"Tiempo de respuesta: {format_seconds(time_opt)}")
                st.caption(f"Similitud de Coseno: {sim_opt:.3f}")
            except Exception as exc:
                st.error(f"Error en RAG optimizado: {exc}")

elif not api_key:
    st.info("Ingresa tu Groq API Key en la barra lateral para iniciar.")
elif not uploaded:
    st.info("Sube un archivo para comenzar.")
elif not question:
    st.info("Escribe una pregunta para comparar resultados.")


with st.expander("Fase 4: Analisis (completa en tu informe o aqui)"):
    st.text_area(
        "1. Alucinacion: En que casos el LLM Simple invento datos que si estaban en el documento?",
        height=80,
    )
    st.text_area(
        '2. Inyeccion de Contexto: Como cambia la respuesta si inyectamos un System Prompt que obligue al modelo a decir "No se" si la respuesta no esta en el RAG?',
        height=80,
    )
    st.text_area(
        "3. Fine-Tuning vs RAG: Por que para este ejercicio el RAG es superior a intentar un Fine-tuning del modelo?",
        height=80,
    )
    st.text_area(
        "4. Transformer vs No-Transformer: Explica brevemente si los embeddings generados dependen de una arquitectura Transformer.",
        height=80,
    )
