"""
lexer.py — Análise Léxica via Autômato Finito Determinístico (AFD)

O AFD é representado explicitamente via tabela de transições.
Estados e transições são mapeados como dicionários, refletindo
diretamente a definição formal: M = (Q, Σ, δ, q0, F)
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────
# Tipos de tokens reconhecidos pelo AFD
# ─────────────────────────────────────────────
TOKEN_TYPES = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT",
    "INSERT", "INTO", "VALUES",
    "UPDATE", "SET",
    "DELETE",
    "CREATE", "TABLE", "DROP",
    "JOIN", "INNER", "LEFT", "RIGHT", "ON",
    "ORDER", "BY", "GROUP", "HAVING",
    "LIMIT", "OFFSET",
    "AS", "DISTINCT", "ALL",
    "NULL", "IS", "IN", "LIKE", "BETWEEN",
    "ASC", "DESC",
    "INT", "VARCHAR", "TEXT", "FLOAT", "BOOLEAN", "DATE",
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "DEFAULT",
    # literais e operadores
    "IDENTIFIER", "NUMBER", "STRING",
    "STAR",           # *
    "COMMA",          # ,
    "SEMICOLON",      # ;
    "LPAREN",         # (
    "RPAREN",         # )
    "DOT",            # .
    "EQ",             # =
    "NEQ",            # != ou <>
    "LT",             # <
    "GT",             # >
    "LTE",            # <=
    "GTE",            # >=
    "PLUS",           # +
    "MINUS",          # -
    "SLASH",          # /
    "EOF",
}

KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT",
    "INSERT", "INTO", "VALUES",
    "UPDATE", "SET",
    "DELETE",
    "CREATE", "TABLE", "DROP",
    "JOIN", "INNER", "LEFT", "RIGHT", "ON",
    "ORDER", "BY", "GROUP", "HAVING",
    "LIMIT", "OFFSET",
    "AS", "DISTINCT", "ALL",
    "NULL", "IS", "IN", "LIKE", "BETWEEN",
    "ASC", "DESC",
    "INT", "VARCHAR", "TEXT", "FLOAT", "BOOLEAN", "DATE",
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "DEFAULT",
}


@dataclass
class Token:
    type: str
    value: str
    position: int  # posição na string original

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, pos={self.position})"


# ─────────────────────────────────────────────
# AFD — definição formal explícita
# Q  = conjunto de estados
# Σ  = alfabeto (categorias de caracteres)
# δ  = função de transição (tabela abaixo)
# q0 = estado inicial
# F  = estados de aceitação
# ─────────────────────────────────────────────

# Categorias de caracteres (Σ)
def char_class(c: str) -> str:
    if c.isalpha() or c == '_':    return 'ALPHA'
    if c.isdigit():                return 'DIGIT'
    if c == "'":                   return 'QUOTE'
    if c == '"':                   return 'DQUOTE'
    if c in ' \t\n\r':            return 'WS'
    if c == '*':                   return 'STAR'
    if c == ',':                   return 'COMMA'
    if c == ';':                   return 'SEMI'
    if c == '(':                   return 'LPAREN'
    if c == ')':                   return 'RPAREN'
    if c == '.':                   return 'DOT'
    if c == '=':                   return 'EQ'
    if c == '!':                   return 'BANG'
    if c == '<':                   return 'LT'
    if c == '>':                   return 'GT'
    if c == '+':                   return 'PLUS'
    if c == '-':                   return 'MINUS'
    if c == '/':                   return 'SLASH'
    return 'OTHER'

# Estados (Q)
STATES = {
    'START', 'IN_IDENT', 'IN_NUMBER', 'IN_NUMBER_DOT', 'IN_FLOAT',
    'IN_STRING', 'IN_STRING_ESC', 'IN_DSTRING', 'IN_DSTRING_ESC',
    'IN_LT', 'IN_GT', 'IN_BANG',
    'ACCEPT_IDENT', 'ACCEPT_NUMBER', 'ACCEPT_FLOAT',
    'ACCEPT_STRING', 'ACCEPT_DSTRING',
    'ACCEPT_SINGLE', 'ACCEPT_NEQ', 'ACCEPT_LTE', 'ACCEPT_GTE',
    'ERROR',
}

# δ — Tabela de transições: (estado_atual, classe_char) → próximo_estado
TRANSITION_TABLE = {
    # Estado inicial
    ('START', 'ALPHA'):  'IN_IDENT',
    ('START', 'DIGIT'):  'IN_NUMBER',
    ('START', 'QUOTE'):  'IN_STRING',
    ('START', 'DQUOTE'): 'IN_DSTRING',
    ('START', 'WS'):     'START',
    ('START', 'STAR'):   'ACCEPT_SINGLE',
    ('START', 'COMMA'):  'ACCEPT_SINGLE',
    ('START', 'SEMI'):   'ACCEPT_SINGLE',
    ('START', 'LPAREN'): 'ACCEPT_SINGLE',
    ('START', 'RPAREN'): 'ACCEPT_SINGLE',
    ('START', 'DOT'):    'ACCEPT_SINGLE',
    ('START', 'EQ'):     'ACCEPT_SINGLE',
    ('START', 'LT'):     'IN_LT',
    ('START', 'GT'):     'IN_GT',
    ('START', 'BANG'):   'IN_BANG',
    ('START', 'PLUS'):   'ACCEPT_SINGLE',
    ('START', 'MINUS'):  'ACCEPT_SINGLE',
    ('START', 'SLASH'):  'ACCEPT_SINGLE',

    # Identificadores / palavras-chave
    ('IN_IDENT', 'ALPHA'):  'IN_IDENT',
    ('IN_IDENT', 'DIGIT'):  'IN_IDENT',
    ('IN_IDENT', 'OTHER'):  'ACCEPT_IDENT',  # reconsume

    # Números inteiros
    ('IN_NUMBER', 'DIGIT'): 'IN_NUMBER',
    ('IN_NUMBER', 'DOT'):   'IN_NUMBER_DOT',
    ('IN_NUMBER', 'OTHER'): 'ACCEPT_NUMBER',

    # Float
    ('IN_NUMBER_DOT', 'DIGIT'): 'IN_FLOAT',
    ('IN_FLOAT', 'DIGIT'):      'IN_FLOAT',
    ('IN_FLOAT', 'OTHER'):      'ACCEPT_FLOAT',

    # Strings com aspas simples
    ('IN_STRING', 'QUOTE'):  'ACCEPT_STRING',
    ('IN_STRING', 'OTHER'):  'IN_STRING',
    ('IN_STRING', 'ALPHA'):  'IN_STRING',
    ('IN_STRING', 'DIGIT'):  'IN_STRING',
    ('IN_STRING', 'WS'):     'IN_STRING',

    # Strings com aspas duplas (identificadores entre aspas)
    ('IN_DSTRING', 'DQUOTE'): 'ACCEPT_DSTRING',
    ('IN_DSTRING', 'OTHER'):  'IN_DSTRING',
    ('IN_DSTRING', 'ALPHA'):  'IN_DSTRING',
    ('IN_DSTRING', 'DIGIT'):  'IN_DSTRING',
    ('IN_DSTRING', 'WS'):     'IN_DSTRING',

    # Operadores compostos
    ('IN_LT', 'EQ'):    'ACCEPT_NEQ',   # <=
    ('IN_LT', 'GT'):    'ACCEPT_NEQ',   # <>
    ('IN_LT', 'OTHER'): 'ACCEPT_SINGLE', # <  (reconsume)
    ('IN_GT', 'EQ'):    'ACCEPT_GTE',
    ('IN_GT', 'OTHER'): 'ACCEPT_SINGLE',
    ('IN_BANG', 'EQ'):  'ACCEPT_NEQ',   # !=
}


class LexerError(Exception):
    def __init__(self, message: str, position: int, trace: Optional[list] = None):
        super().__init__(message)
        self.position = position
        self.trace = trace or []


class Lexer:
    """
    Implementa o AFD para análise léxica de SQL simplificado.
    M = (Q, Σ, δ, START, F)
    F = {ACCEPT_IDENT, ACCEPT_NUMBER, ACCEPT_FLOAT, ACCEPT_STRING,
         ACCEPT_DSTRING, ACCEPT_SINGLE, ACCEPT_NEQ, ACCEPT_LTE, ACCEPT_GTE}
    """

    ACCEPT_STATES = {
        'ACCEPT_IDENT', 'ACCEPT_NUMBER', 'ACCEPT_FLOAT',
        'ACCEPT_STRING', 'ACCEPT_DSTRING',
        'ACCEPT_SINGLE', 'ACCEPT_NEQ', 'ACCEPT_LTE', 'ACCEPT_GTE',
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.tokens: List[Token] = []
        self.trace: List[dict] = []   # rastro de estados para documentação

    def _next_char(self) -> str:
        if self.pos < len(self.source):
            return self.source[self.pos]
        return '\x00'

    def _map_single_char_token(self, ch: str) -> str:
        mapping = {
            '*': 'STAR', ',': 'COMMA', ';': 'SEMICOLON',
            '(': 'LPAREN', ')': 'RPAREN', '.': 'DOT',
            '=': 'EQ', '+': 'PLUS', '-': 'MINUS', '/': 'SLASH',
            '<': 'LT', '>': 'GT',
        }
        return mapping.get(ch, 'UNKNOWN')

    def tokenize(self) -> List[Token]:
        state = 'START'
        buf = ''
        tok_start = 0
        reconsume = False

        while self.pos <= len(self.source):
            ch = self._next_char() if not reconsume else ch
            reconsume = False

            cc = char_class(ch) if ch != '\x00' else 'OTHER'

            # Consulta δ
            next_state = TRANSITION_TABLE.get((state, cc))

            # Se não há transição, tratar como "OTHER" (reconsume boundary)
            if next_state is None:
                next_state = TRANSITION_TABLE.get((state, 'OTHER'), 'ERROR')
                reconsume = True

            self.trace.append({
                'state': state,
                'char': ch,
                'char_class': cc,
                'next_state': next_state,
                'buffer': buf,
                'pos': min(self.pos, len(self.source)),
            })

            # ── Estados de aceitação ──
            if next_state in self.ACCEPT_STATES:
                if not reconsume:
                    buf += ch
                    self.pos += 1

                token = self._emit_token(next_state, buf, tok_start)
                if token:
                    self.tokens.append(token)

                buf = ''
                state = 'START'
                tok_start = self.pos
                continue

            elif next_state == 'ERROR':
                if ch == '\x00':
                    break
                raise LexerError(
                    f"Caractere inesperado '{ch}' na posição {self.pos}",
                    self.pos,
                    list(self.trace),
                )

            else:
                if not reconsume:
                    if state != 'START' or ch not in ' \t\n\r':
                        buf += ch
                    self.pos += 1
                state = next_state

        self.tokens.append(Token('EOF', '', len(self.source)))
        return self.tokens

    def _emit_token(self, accept_state: str, buf: str, pos: int) -> Token | None:
        buf = buf.strip()
        if not buf:
            return None

        if accept_state == 'ACCEPT_IDENT':
            upper = buf.upper()
            ttype = upper if upper in KEYWORDS else 'IDENTIFIER'
            return Token(ttype, buf, pos)

        if accept_state == 'ACCEPT_NUMBER':
            return Token('NUMBER', buf, pos)

        if accept_state == 'ACCEPT_FLOAT':
            return Token('NUMBER', buf, pos)

        if accept_state in ('ACCEPT_STRING', 'ACCEPT_DSTRING'):
            return Token('STRING', buf, pos)

        if accept_state == 'ACCEPT_SINGLE':
            ch = buf[-1] if buf else ''
            return Token(self._map_single_char_token(ch), buf, pos)

        if accept_state == 'ACCEPT_NEQ':
            return Token('NEQ', buf, pos)

        if accept_state == 'ACCEPT_LTE':
            return Token('LTE', buf, pos)

        if accept_state == 'ACCEPT_GTE':
            return Token('GTE', buf, pos)

        return None


def tokenize(sql: str) -> Tuple[List[Token], List[dict]]:
    """Ponto de entrada público."""
    lexer = Lexer(sql)
    try:
        tokens = lexer.tokenize()
        return tokens, lexer.trace
    except LexerError as e:
        e.trace = e.trace or list(lexer.trace)
        raise
