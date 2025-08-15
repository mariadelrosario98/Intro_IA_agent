import os
import streamlit as st

# LangChain + Groq
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

# Herramienta gratuita de búsqueda
from langchain_community.tools import DuckDuckGoSearchRun

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="Agente (LangChain + Groq)", page_icon="🧠", layout="wide")
st.title("🧠 Agente LangChain con Groq (llama3-8b-8192)")
st.caption("Usando Streamlit Secrets para manejar credenciales de forma segura")

# =========================
# Cargar credenciales desde secrets
# =========================
def get_groq_key() -> str:
    """
    Obtiene la clave de Groq desde Streamlit Secrets.
    Estructura esperada en secrets.toml:
    [groq]
    api_key = "TU_API_KEY"
    """
    try:
        return st.secrets["groq"]["api_key"]
    except Exception:
        # Fallback por si el usuario exportó la variable de entorno (opcional)
        return os.getenv("GROQ_API_KEY", "")

GROQ_API_KEY = get_groq_key()
if not GROQ_API_KEY:
    st.error(
        "No se encontró `groq.api_key` en `.streamlit/secrets.toml` "
        "ni la variable de entorno `GROQ_API_KEY`. "
        "Configura tus credenciales y recarga la app."
    )
    st.stop()

# =========================
# Sidebar: Configuración
# =========================
st.sidebar.header("🔧 Configuración del LLM")
model_name = st.sidebar.selectbox(
    "Modelo Groq",
    options=["llama3-8b-8192", "llama3-70b-8192"],
    index=0
)
temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.2, 0.1)
use_search = st.sidebar.checkbox("Habilitar búsqueda web (DuckDuckGo)", value=True)
if st.sidebar.button("🗑️ Limpiar conversación"):
    st.session_state.clear()

# =========================
# Estado de sesión
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# =========================
# Construcción del agente
# =========================
@st.cache_resource(show_spinner=False)
def build_agent(api_key: str, model: str, temp: float, enable_search: bool):
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=temp,
    )

    tools = []
    if enable_search:
        tools.append(DuckDuckGoSearchRun(name="web-search"))

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=st.session_state.memory,
        handle_parsing_errors=True,
    )
    return agent

st.session_state.agent = build_agent(GROQ_API_KEY, model_name, temperature, use_search)

# =========================
# UI de chat
# =========================
# Historial
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# Entrada
user_prompt = st.chat_input("Escribe tu mensaje para el agente...")
if user_prompt:
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = st.session_state.agent.run(user_prompt)
            except Exception as e:
                response = f"Lo siento, ocurrió un error: {e}"
        st.markdown(response)
        st.session_state.messages.append(AIMessage(content=response))

# =========================
# Panel de depuración
# =========================
with st.expander("🧠 Memoria de conversación (debug)"):
    buf = getattr(st.session_state.memory, "buffer_as_str", "")
    st.write(buf if buf else "—")
