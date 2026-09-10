import numpy as np
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

app = FastAPI(title="Agent IA - Espace d'États")

BETA = 0.65
ALPHA = 0.30
GAMMA = 0.80
DELTA = 0.50

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_GGUF = "qwen2.5:3b"

INITIAL_STATE = np.array([2.5, 0.1, 0.7])
sessions = {}


class ChatInput(BaseModel):
    message: str
    session_id: str = "default"


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {"state": INITIAL_STATE.copy(), "history": []}
    return sessions[session_id]


def requete_gguf(prompt, system_prompt="", json_mode=False):
    payload = {
        "model": MODEL_GGUF,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9}
    }
    if json_mode:
        payload["format"] = "json"
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout Ollama")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama non accessible")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur GGUF : {str(e)}")


def extraire_u_utilisateur(replique_texte):
    prompt = f"""Analyse la réplique suivante dans une négociation et quantifie l'action 'u' entre -1.0 et 1.0.

- u proche de -1.0 : Agression, ultimatum, refus hostile.
- u proche de 0.0 : Fait neutre, description purement professionnelle.
- u proche de +1.0 : Concession, écoute active, esprit de compromis.

Réplique : "{replique_texte}"

Renvoie UNIQUEMENT un JSON sous cette forme exacte : {{"u": <float>}}"""

    reponse_json = requete_gguf(prompt, json_mode=True)
    try:
        data = json.loads(reponse_json)
        return float(min(1.0, max(-1.0, float(data.get("u", 0.0)))))
    except (ValueError, KeyError, json.JSONDecodeError, TypeError):
        return 0.0


def transition_modele(x, u_user, u_ia):
    x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])
    u_net = (u_user + u_ia) / 2.0
    sat_x1 = np.tanh(x1)
    phi_u = max(0.0, u_net)
    attaque_u = max(0.0, -u_net)

    x1_next = x1 + (1.0 - x3) * sat_x1 - (1.0 - x2) * (1.0 + x3) * phi_u
    x2_next = min(1.0, max(0.0,
        BETA * x2 + ALPHA * abs(x1) * (1.0 - x3) + GAMMA * attaque_u
    ))
    x3_next = float(np.tanh(x3 + DELTA * (1.0 - x2) * u_net))

    return np.array([x1_next, x2_next, x3_next])


@app.post("/chat")
def chat_endpoint(input_data: ChatInput):
    session = get_session(input_data.session_id)
    state = session["state"]
    user_text = input_data.message

    u_user = extraire_u_utilisateur(user_text)

    tension = float(state[1])
    if tension > 0.6:
        u_ia = 0.8
        consigne = "La tension est critique. Empathie, validation, désescalade."
    elif tension > 0.3:
        u_ia = 0.5
        consigne = "La tension est modérée. Reconnais les contraintes, propose un compromis."
    else:
        u_ia = 0.3
        consigne = "La tension est sous contrôle. Avance calmement sur le fond."

    old_state = state.copy()
    new_state = transition_modele(state, u_user, u_ia)
    session["state"] = new_state

    system_prompt = f"""Tu es un négociateur professionnel.
État psychologique :
- Écart sémantique (x1) : {new_state[0]:.2f}
- Tension (x2) : {new_state[1]:.2f}
- Posture (x3) : {new_state[2]:.2f}
Directive : {consigne}
Ne parle jamais des variables mathématiques. Incarne la posture requise. Réponds en français."""

    ia_text = requete_gguf(user_text, system_prompt=system_prompt)

    session["history"].append({
        "user": user_text,
        "ia": ia_text,
        "u_user": u_user,
        "u_ia": u_ia,
        "state": new_state.tolist()
    })

    return {
        "ia_response": ia_text,
        "extraction_u_user": u_user,
        "loi_commande_u_ia": u_ia,
        "ancien_etat": {"x1": float(old_state[0]), "x2": float(old_state[1]), "x3": float(old_state[2])},
        "nouvel_etat": {"x1": float(new_state[0]), "x2": float(new_state[1]), "x3": float(new_state[2])}
    }


@app.post("/reset")
def reset(session_id: str = "default"):
    sessions[session_id] = {"state": INITIAL_STATE.copy(), "history": []}
    return {"status": "reset", "etat": INITIAL_STATE.tolist()}


@app.get("/history")
def history(session_id: str = "default"):
    session = get_session(session_id)
    return {"history": session["history"]}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent GGUF Supervisé</title>
        <meta charset="utf-8">
        <style>
            body { font-family: sans-serif; max-width: 900px; margin: 40px auto; background: #0f172a; color: #e2e8f0; padding: 20px; }
            h2 { color: #38bdf8; }
            #chat { background: #1e293b; height: 400px; overflow-y: scroll; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }
            .msg-user { color: #fbbf24; margin: 8px 0; }
            .msg-ia { color: #4ade80; margin: 8px 0; }
            #input-row { display: flex; gap: 10px; }
            input { flex: 1; padding: 12px; background: #334155; color: white; border: none; border-radius: 4px; font-size: 14px; }
            button { padding: 12px 20px; background: #0284c7; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
            button.reset { background: #dc2626; }
            pre { background: #020617; color: #4ade80; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 12px; }
        </style>
    </head>
    <body>
        <h2>Négociation Assistée par GGUF & Espace d'États</h2>
        <div id="chat"></div>
        <div id="input-row">
            <input type="text" id="msg" placeholder="Discuter avec l'IA..." onkeydown="if(event.key==='Enter') envoyer()">
            <button onclick="envoyer()">Envoyer</button>
            <button class="reset" onclick="resetSession()">Reset</button>
        </div>
        <h3>Métriques</h3>
        <pre id="metrics">En attente...</pre>
        <script>
            const sessionId = "session-" + Math.random().toString(36).substring(2, 10);
            async function envoyer() {
                const input = document.getElementById('msg');
                const m = input.value.trim();
                if (!m) return;
                const chat = document.getElementById('chat');
                chat.innerHTML += `<div class="msg-user"><b>Vous :</b> ${escapeHtml(m)}</div>`;
                input.value = '';
                chat.scrollTop = chat.scrollHeight;
                const r = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: m, session_id: sessionId})
                });
                const d = await r.json();
                if (d.detail) {
                    chat.innerHTML += `<div class="msg-ia" style="color:#f87171"><b>Erreur :</b> ${escapeHtml(d.detail)}</div>`;
                } else {
                    chat.innerHTML += `<div class="msg-ia"><b>IA :</b> ${escapeHtml(d.ia_response)}</div>`;
                    document.getElementById('metrics').innerText = JSON.stringify(d, null, 2);
                }
                chat.scrollTop = chat.scrollHeight;
            }
            async function resetSession() {
                await fetch('/reset?session_id=' + sessionId, {method: 'POST'});
                document.getElementById('chat').innerHTML = '';
                document.getElementById('metrics').innerText = 'Session réinitialisée.';
            }
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
        </script>
    </body>
    </html>
    
