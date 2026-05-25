"""
automata.py — Estrutura do AFD para visualização (grafo δ)
Exporta nós, arestas e layout didático do autômato léxico.
"""

from lexer import TRANSITION_TABLE, STATES

ACCEPT_STATES = {
    'ACCEPT_IDENT', 'ACCEPT_NUMBER', 'ACCEPT_FLOAT',
    'ACCEPT_STRING', 'ACCEPT_DSTRING',
    'ACCEPT_SINGLE', 'ACCEPT_NEQ', 'ACCEPT_LTE', 'ACCEPT_GTE',
}

# Posições fixas para layout SVG (x, y) — agrupamento didático
STATE_LAYOUT = {
    'START': (40, 200),
    'IN_IDENT': (180, 80),
    'IN_NUMBER': (180, 160),
    'IN_NUMBER_DOT': (300, 160),
    'IN_FLOAT': (420, 160),
    'IN_STRING': (180, 240),
    'IN_DSTRING': (180, 300),
    'IN_LT': (180, 360),
    'IN_GT': (180, 420),
    'IN_BANG': (180, 480),
    'ACCEPT_IDENT': (560, 80),
    'ACCEPT_NUMBER': (560, 160),
    'ACCEPT_FLOAT': (560, 200),
    'ACCEPT_STRING': (560, 260),
    'ACCEPT_DSTRING': (560, 300),
    'ACCEPT_SINGLE': (560, 360),
    'ACCEPT_NEQ': (560, 400),
    'ACCEPT_LTE': (560, 440),
    'ACCEPT_GTE': (560, 480),
    'ERROR': (300, 520),
}


def get_afd_graph() -> dict:
    """Retorna o grafo completo do AFD para renderização no frontend."""
    edges = []
    seen = set()
    for (src, label), dst in TRANSITION_TABLE.items():
        key = (src, label, dst)
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "from": src,
            "to": dst,
            "label": label,
        })

    nodes = []
    for state in sorted(STATES):
        layout = STATE_LAYOUT.get(state, (0, 0))
        nodes.append({
            "id": state,
            "x": layout[0],
            "y": layout[1],
            "is_accept": state in ACCEPT_STATES,
            "is_start": state == 'START',
            "is_error": state == 'ERROR',
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "start": "START",
        "accept_states": sorted(ACCEPT_STATES),
    }
