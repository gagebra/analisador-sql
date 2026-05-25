"""
app.py — Servidor Flask
Integra Lexer (AFD) + Parser (PDA) e expõe a API para o frontend.
"""

from flask import Flask, render_template, request, jsonify
from lexer import tokenize, LexerError
from parser import parse, ParseError
from automata import get_afd_graph
import os

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    sql = data.get("sql", "").strip()

    if not sql:
        return jsonify({"error": "Nenhuma query fornecida."}), 400

    result = {
        "sql": sql,
        "tokens": [],
        "afd_trace": [],
        "afd_graph": get_afd_graph(),
        "pda_trace": [],
        "ast": None,
        "derivation": [],
        "valid": False,
        "error": None,
        "error_position": None,
    }

    # ── Fase 1: Análise Léxica (AFD) ──
    try:
        tokens, trace = tokenize(sql)
        result["tokens"] = [
            {"type": t.type, "value": t.value, "position": t.position}
            for t in tokens if t.type != "EOF"
        ]
        result["afd_trace"] = [
            {
                "state": s["state"],
                "char": s["char"] if s["char"] != "\x00" else "EOF",
                "char_class": s["char_class"],
                "next_state": s["next_state"],
                "buffer": s["buffer"],
                "pos": s.get("pos", 0),
            }
            for s in trace[:300]
        ]
    except LexerError as e:
        result["error"] = f"Erro léxico: {str(e)}"
        result["error_position"] = e.position
        trace = getattr(e, "trace", []) or []
        result["afd_trace"] = [
            {
                "state": s["state"],
                "char": s["char"] if s["char"] != "\x00" else "EOF",
                "char_class": s["char_class"],
                "next_state": s["next_state"],
                "buffer": s["buffer"],
                "pos": s.get("pos", 0),
            }
            for s in trace[:300]
        ]
        return jsonify(result)

    # ── Fase 2: Análise Sintática (PDA + GLC) ──
    try:
        ast, derivation, pda_trace = parse(tokens)
        result["ast"] = ast.to_dict()
        result["derivation"] = derivation
        result["pda_trace"] = pda_trace[:400]
        result["valid"] = True
    except ParseError as e:
        result["error"] = f"Erro sintático: {str(e)}"
        result["error_position"] = e.position
        result["pda_trace"] = e.pda_trace[:400]
        return jsonify(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))