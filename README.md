
Conversational Dynamics Layer

An exploratory dynamical systems layer for conversational regulation.

This is an IMPERFECT DRAFT. It proposes a three-variable model (semantic gap, tension, posture) with coupled transition equations, a Lyapunov function, and a control policy. The model runs as an external layer to an LLM served by Ollama.

Status

Exploratory draft. Not validated. Open to criticism.

## Requirements

- Python 3.10+
- Ollama running locally with a model (e.g. qwen2.5:3b)

## Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/TON_USERNAME/conversational-dynamics-layer.git
cd conversational-dynamics-layer
pip install -r requirements.txt
```

## Ollama setup

Make sure Ollama is running and pull a model.

```bash
ollama serve
ollama pull qwen2.5:3b
```

If you want to use a different model, change the value of MODEL_GGUF in app.py.

## Run

```bash
python -m uvicorn app:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

## What to test

1. Extractor fidelity
Send 20 varied sentences (aggression, concession, neutral) and compare the extracted u with your own human coding. If the correlation is above 0.7, the extractor is usable.

2. Generator stability
Fix the state to a given value and generate 10 responses. Check if the posture is respected consistently.

3. Full trajectory
Simulate a 10-turn negotiation and observe the evolution of x1, x2, x3. Check if convergence or divergence matches your intuition.

4. Noise robustness
Inject an ambiguous sentence mid-conversation and observe whether the system recovers or diverges.

## Model summary

State space:

- x1 in R: semantic gap between positions. Perfect agreement is at x1 = 0.
- x2 in [0, 1]: emotional tension. 0 = calm, 1 = saturation.
- x3 in [-1, 1]: cooperation posture. +1 = collaboration, -1 = obstruction.

Nominal equilibrium point: x* = [0, 0, 1].

Transition equations:

```
x1(k+1) = x1(k) + (1 - x3(k)) * sat1(x1(k)) - (1 - x2(k)) * (1 + x3(k)) * phi(u(k))
x2(k+1) = sat01( beta * x2(k) + alpha * |x1(k)| * (1 - x3(k)) + gamma * max(0, -u(k)) )
x3(k+1) = tanh( x3(k) + delta * (1 - x2(k)) * u(k) )
```

Where:

- sat01(z) = min(1, max(0, z))
- sat1(z) = tanh(z)
- phi(u) = max(0, u)
- beta in ]0, 1[: natural tension dissipation
- alpha > 0: sensitivity to disagreement
- gamma > 0: destructive impact of an attack
- delta > 0: posture flexibility

Lyapunov function:

```
V(x) = 0.5 * x1^2 + 0.5 * x2^2 + 0.5 * (1 - x3)^2
```

Convergence condition:

If u(k) > 0 is maintained and if alpha < (1 - beta) / 2, the system converges asymptotically to x*.

## Architecture

- FastAPI serves the API and the web interface.
- Ollama hosts the GGUF model locally.
- The dynamic engine runs in Python and maintains the conversation state.
- The LLM acts as an extractor (text to u_user) and as a generator (state to text).

Two LLM calls are made per turn: one for extraction, one for generation.

## Endpoints

POST /chat
Send a message and receive the agent response with the updated state.

POST /reset
Reset the conversation state for a given session.

GET /history
Retrieve the full trajectory for a given session.

GET /
Web interface for interactive testing.

## Limitations

- The model does not capture power.
- The model does not capture time.
- The model does not capture reputation.
- Noise is not modeled.
- Bifurcation analysis remains to be completed.
- Empirical validation is preliminary.

## Open questions

- What is the exact critical value of alpha?
- How to model noise (misunderstanding)?
- How to integrate power, time, reputation?
- How to validate on a larger corpus?

## Invitation

This model is an exploratory draft. Researchers and practitioners are invited to test it, to criticize it, and to propose extensions.

If you test this model, if you criticize it, if you extend it, feedback is essential to refine this draft.

## License

MIT
