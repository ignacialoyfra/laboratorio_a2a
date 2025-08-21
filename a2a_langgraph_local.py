import asyncio
import threading
from typing import TypedDict, List
from dataclasses import dataclass

# -------- A2A (python-a2a): agentes y cliente --------
from python_a2a import (
    A2AServer,
    run_server,
    TaskStatus,
    TaskState,
    A2AClient,
)

# -------- LangGraph --------
from langgraph.graph import StateGraph, END


# =========================
# 1) Agente CLASSIFIER A2A
# =========================
class ClassifierAgent(A2AServer):
    """
    Clasifica el texto en:
      - QUESTION si termina en '?'
      - LONG si largo > 40
      - SHORT en otro caso
    Devuelve texto plano, p.ej. "label=QUESTION"
    """
    def handle_task(self, task):
        message = task.message or {}
        content = message.get("content", {})
        text = content.get("text", "") if isinstance(content, dict) else str(content)

        text = (text or "").strip()

        if text.endswith("?"):
            label = "QUESTION"
        elif len(text) > 40:
            label = "LONG"
        else:
            label = "SHORT"

        # Respuesta A2A: artifacts con part de tipo "text"
        task.artifacts = [{"parts": [{"type": "text", "text": f"label={label}"}]}]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task


# ==========================
# 2) Agente TRANSFORMER A2A
# ==========================
class TransformerAgent(A2AServer):
    """
    Transforma el texto según 'style' (en headers no lo exige A2A aquí,
    así que lo pasamos en el prompt por simplicidad):
      - "upper": MAYÚSCULAS
      - "title": Title Case
      - default : "[ok]" al final
    """
    def handle_task(self, task):
        message = task.message or {}
        content = message.get("content", {})
        text = content.get("text", "") if isinstance(content, dict) else str(content)

        # Mini-protocolo: si el texto llega como "style=upper::hola"
        style = "raw"
        if text.startswith("style=") and "::" in text:
            prefix, rest = text.split("::", 1)
            style = prefix.split("=", 1)[-1].strip()
            text = rest

        if style == "upper":
            out = text.upper()
        elif style == "title":
            out = text.title()
        else:
            out = text + " [ok]"

        task.artifacts = [{"parts": [{"type": "text", "text": out}]}]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task


# ==================================================
# 3) Levantar ambos agentes A2A en hilos de fondo
# ==================================================
def start_agent_server(agent_cls, port: int):
    def _runner():
        run_server(agent_cls(), port=port)
    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    return th


# ================================
# 4) Orquestador con LangGraph
# ================================
class OrchestratorState(TypedDict):
    query: str
    classification: str
    transformed: str
    log: List[str]


@dataclass
class Orchestrator:
    classifier_url: str
    transformer_url: str

    async def route(self, state: OrchestratorState) -> OrchestratorState:
        log = state.get("log", [])
        q = state["query"]

        # Clientes A2A contra cada agente
        classifier = A2AClient(self.classifier_url)
        transformer = A2AClient(self.transformer_url)

        # --- 1) Clasificar ---
        # A2AClient.ask(...) devuelve texto (conforme a ejemplos del README)
        cls_out = await asyncio.to_thread(classifier.ask, q)
        log.append(f"classifier -> {cls_out}")

        label = cls_out.split("=", 1)[-1].strip()

        # --- 2) Elegir estilo para el transformer ---
        if label == "QUESTION":
            style = "title"
        elif label == "LONG":
            style = "upper"
        else:
            style = "raw"

        # --- 3) Transformar (pasamos style en el prompt simple) ---
        t_in = f"style={style}::{q}"
        tr_out = await asyncio.to_thread(transformer.ask, t_in)
        log.append(f"transformer(style={style}) -> {tr_out}")

        # Actualizar estado
        state["classification"] = label
        state["transformed"] = tr_out
        state["log"] = log
        return state


def build_graph(orchestrator: Orchestrator):
    g = StateGraph(OrchestratorState)
    g.add_node("route", orchestrator.route)
    g.set_entry_point("route")
    g.add_edge("route", END)
    return g.compile()


# ==========================
# 5) Demo end-to-end
# ==========================
async def main():
    # Puertos locales para los dos agentes
    PORT_CLASSIFIER = 7001
    PORT_TRANSFORMER = 7002

    # Levantar servidores A2A en background (mismo script)
    start_agent_server(ClassifierAgent, PORT_CLASSIFIER)
    start_agent_server(TransformerAgent, PORT_TRANSFORMER)

    # Construir orquestador (LangGraph)
    orch = Orchestrator(
        classifier_url=f"http://localhost:{PORT_CLASSIFIER}",
        transformer_url=f"http://localhost:{PORT_TRANSFORMER}",
    )
    app = build_graph(orch)

    # Ejemplos
    tests = [
        "Hola mundo",
        "Este es un texto bastante más largo para probar el flujo",
        "¿Cómo integro A2A con LangGraph?"
    ]

    for q in tests:
        state: OrchestratorState = {
            "query": q,
            "classification": "",
            "transformed": "",
            "log": [],
        }
        result = await app.ainvoke(state)
        print("\n=== INPUT ==================")
        print(q)
        print("=== OUTPUT =================")
        print("classification:", result["classification"])
        print("transformed   :", result["transformed"])
        print("log           :", " | ".join(result["log"]))


if __name__ == "__main__":
    asyncio.run(main())
