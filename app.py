"""
app.py — Servidor Flask
Integra Lexer (AFD) + Parser (PDA) e expõe a API para o frontend.
"""

from flask import Flask, render_template, request, jsonify
from lexer import tokenize, LexerError
from parser import parse, ParseError

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
            }
            for s in trace[:150]  # limitar para não explodir o JSON
        ]
    except LexerError as e:
        result["error"] = f"Erro léxico: {str(e)}"
        result["error_position"] = e.position
        return jsonify(result)

    # ── Fase 2: Análise Sintática (PDA + GLC) ──
    try:
        ast, derivation = parse(tokens)
        result["ast"] = ast.to_dict()
        result["derivation"] = derivation
        result["valid"] = True
    except ParseError as e:
        result["error"] = f"Erro sintático: {str(e)}"
        result["error_position"] = e.position
        return jsonify(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
