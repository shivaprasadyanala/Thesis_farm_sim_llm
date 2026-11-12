"""HTTP layer - the only way this harness touches the game."""

import logging

from . import runtime

logger = logging.getLogger(__name__)


def api_get(path, **params):
    response = runtime.session.get(f"{runtime.base_url}{path}", params=params or None, timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path, payload, include_state=False):
    """POST one action. `include_state=1` echoes the fresh observation back."""
    params = {"include_state": 1} if include_state else None
    response = runtime.session.post(f"{runtime.base_url}{path}", json=payload,
                                    params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def refresh_state(events=6):
    runtime.state = api_get("/api/v1/state", events=events)
    return runtime.state


def describe_server(health):
    """One line that proves *which* server this run is talking to."""
    return (f"server: {health.get('version', '?')} preset={health.get('preset')} "
            f"seed={health.get('seed')} episode={health.get('episode')} "
            f"tick={health.get('tick')} browser clients={health.get('clients', 0)}")
