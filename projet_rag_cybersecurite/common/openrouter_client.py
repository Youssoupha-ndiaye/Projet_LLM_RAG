"""
common/openrouter_client.py — appelle les LLM d'OpenRouter.

OpenRouter expose une API compatible OpenAI :
    POST https://openrouter.ai/api/v1/chat/completions
    Authorization: Bearer <VOTRE_CLE_API>

Ce module gère :
  * l'appel d'un modèle (avec une liste de modèles de secours),
  * les erreurs courantes (clé invalide, crédits, limite de débit 429...),
  * la liste des modèles gratuits actuellement disponibles.
La clé API n'est JAMAIS écrite sur le disque par ce code.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


class OpenRouterError(RuntimeError):
    """Erreur lisible renvoyée à l'interface / au script appelant."""


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # En-têtes optionnels recommandés par OpenRouter (identifient l'appli).
        "HTTP-Referer": "http://localhost",
        "X-Title": "RAG Cybersecurite (projet LLM)",
    }


def list_free_models(api_key: str | None = None, timeout: int = 30) -> list[str]:
    """Retourne les identifiants des modèles dont prompt ET completion coûtent 0.

    La liste des modèles gratuits change souvent : c'est pour ça qu'on la
    redemande à l'API au lieu de la figer dans le code.
    """
    headers = _headers(api_key.strip()) if api_key and api_key.strip() else {}
    try:
        r = requests.get(f"{config.OPENROUTER_BASE_URL}/models", headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise OpenRouterError(f"Impossible de joindre OpenRouter : {e}") from e
    if r.status_code != 200:
        raise OpenRouterError(f"Impossible de lister les modèles (HTTP {r.status_code}).")
    free = []
    for m in r.json().get("data", []):
        p = m.get("pricing") or {}
        try:
            if float(p.get("prompt", 1)) == 0 and float(p.get("completion", 1)) == 0:
                free.append(m["id"])
        except (TypeError, ValueError):
            continue
    free = sorted(set(free))
    # Le routeur gratuit en premier : c'est le choix par défaut.
    if "openrouter/free" in free:
        free.remove("openrouter/free")
    return ["openrouter/free"] + free


def chat(
    messages: list[dict],
    api_key: str,
    model: str | None = None,
    fallbacks: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    timeout: int = 90,
    retries_per_model: int = 2,
) -> dict:
    """Envoie une conversation au LLM.

    Retourne {"content": str, "model": str}, où "model" est le modèle qui a
    réellement répondu (utile avec le routeur openrouter/free).
    Lève OpenRouterError si tous les modèles échouent.
    """
    if not api_key or not api_key.strip():
        raise OpenRouterError("Aucune clé API OpenRouter fournie.")
    api_key = api_key.strip()
    candidates = [model or config.DEFAULT_MODEL] + list(fallbacks or config.FALLBACK_MODELS)
    last_error = "inconnue"

    for mdl in candidates:
        for attempt in range(retries_per_model + 1):
            try:
                r = requests.post(
                    f"{config.OPENROUTER_BASE_URL}/chat/completions",
                    headers=_headers(api_key),
                    json={
                        "model": mdl,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=timeout,
                )
            except requests.RequestException as e:
                last_error = f"réseau : {e}"
                time.sleep(2)
                continue

            if r.status_code == 401:
                raise OpenRouterError("Clé API refusée (401). Vérifiez qu'elle est correcte et active.")
            if r.status_code == 402:
                raise OpenRouterError("Crédits insuffisants (402) : choisissez un modèle ':free' ou rechargez.")
            if r.status_code == 429:  # limite de débit : on attend puis on réessaie
                last_error = f"{mdl} : limite de requêtes atteinte (429)"
                try:
                    wait = min(float(r.headers.get("Retry-After", 5)), 20)
                except ValueError:
                    wait = 5
                if attempt < retries_per_model:
                    time.sleep(wait)
                continue
            if r.status_code != 200:
                last_error = f"{mdl} : HTTP {r.status_code} {r.text[:200]}"
                break  # modèle indisponible -> modèle suivant

            data = r.json()
            if "error" in data:  # OpenRouter peut renvoyer 200 + un champ error
                last_error = f"{mdl} : {data['error']}"
                break
            try:
                content = (data["choices"][0]["message"].get("content") or "").strip()
            except (KeyError, IndexError):
                last_error = f"{mdl} : réponse mal formée"
                break
            if not content:  # certains modèles "raisonnement" renvoient du vide
                last_error = f"{mdl} : réponse vide"
                break
            return {"content": content, "model": data.get("model", mdl)}

    raise OpenRouterError(f"Tous les modèles ont échoué. Dernière erreur : {last_error}")
