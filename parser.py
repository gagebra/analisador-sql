"""
parser.py — Análise Sintática via Autômato com Pilha (PDA)
          + Gramática Livre de Contexto (GLC)

Gramática SQL simplificada (subconjunto):

S → select_stmt | insert_stmt | update_stmt | delete_stmt | create_stmt | drop_stmt

select_stmt  → SELECT col_list FROM table_ref [join_clause] [where_clause]
               [group_clause] [having_clause] [order_clause] [limit_clause] SEMI?

insert_stmt  → INSERT INTO IDENTIFIER [LPAREN col_list RPAREN]
               VALUES LPAREN value_list RPAREN SEMI?

update_stmt  → UPDATE IDENTIFIER SET assignment_list [where_clause] SEMI?

delete_stmt  → DELETE FROM IDENTIFIER [where_clause] SEMI?

create_stmt  → CREATE TABLE IDENTIFIER LPAREN col_def_list RPAREN SEMI?

drop_stmt    → DROP TABLE IDENTIFIER SEMI?

col_list     → expr (COMMA expr)*  |  STAR
value_list   → value (COMMA value)*
assignment_list → assignment (COMMA assignment)*
assignment   → IDENTIFIER EQ expr

expr         → term ((PLUS | MINUS) term)*
term         → factor ((STAR | SLASH) factor)*
factor       → NUMBER | STRING | IDENTIFIER (DOT IDENTIFIER)? | LPAREN expr RPAREN | NULL
             | IDENTIFIER LPAREN [expr_list] RPAREN   ← chamada de função

cond         → expr (EQ | NEQ | LT | GT | LTE | GTE) expr
             | expr IS [NOT] NULL
             | expr [NOT] IN LPAREN value_list RPAREN
             | expr [NOT] LIKE STRING
             | NOT cond
             | LPAREN condition RPAREN

condition    → cond ((AND | OR) cond)*

where_clause  → WHERE condition
group_clause  → GROUP BY col_list
having_clause → HAVING condition
order_clause  → ORDER BY order_list
order_list    → (expr [ASC|DESC]) (COMMA expr [ASC|DESC])*
limit_clause  → LIMIT NUMBER [OFFSET NUMBER]

join_clause   → (INNER|LEFT|RIGHT)? JOIN table_ref ON condition
table_ref     → IDENTIFIER [AS IDENTIFIER] | LPAREN select_stmt RPAREN AS IDENTIFIER

col_def_list  → col_def (COMMA col_def)*
col_def       → IDENTIFIER type_def [col_constraints]
type_def      → INT | VARCHAR (LPAREN NUMBER RPAREN)? | TEXT | FLOAT | BOOLEAN | DATE
col_constraints → (PRIMARY KEY | NOT NULL | UNIQUE | DEFAULT value)*

O PDA é implementado como um parser recursivo-descendente onde a pilha
de chamadas da linguagem Python representa a pilha do autômato (P).
Cada função ↔ não-terminal da gramática.
Cada consume() ↔ transição do PDA lendo um símbolo da entrada.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from lexer import Token


# ─────────────────────────────────────────────
# Nós da Árvore Sintática (AST)
# ─────────────────────────────────────────────
@dataclass
class ASTNode:
    type: str
    children: List[Any] = field(default_factory=list)
    value: Optional[str] = None

    def to_dict(self):
        d = {"type": self.type}
        if self.value is not None:
            d["value"] = self.value
        if self.children:
            d["children"] = [
                c.to_dict() if isinstance(c, ASTNode) else str(c)
                for c in self.children
            ]
        return d


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(message)
        self.token = token
        self.position = token.position


# ─────────────────────────────────────────────
# Parser (PDA recursivo-descendente)
# ─────────────────────────────────────────────
class Parser:
    """
    PDA = (Q, Σ, Γ, δ, q0, Z0, F)
    Q  = estados representados pelas funções de parse
    Σ  = tokens produzidos pelo Lexer
    Γ  = pilha de chamadas Python (implícita)
    q0 = parse_statement()
    F  = retorno bem-sucedido com EOF
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type != 'EOF'] + \
                      [t for t in tokens if t.type == 'EOF']
        self.pos = 0
        self.derivation: List[str] = []   # passos de derivação (para documentação)

    # ── Utilitários do PDA ──────────────────────

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, offset=0) -> Token:
        idx = self.pos + offset
        return self.tokens[min(idx, len(self.tokens) - 1)]

    def _is(self, *types) -> bool:
        return self._current().type in types

    def consume(self, *expected_types) -> Token:
        """Lê um token da entrada (transição do PDA)."""
        tok = self._current()
        if expected_types and tok.type not in expected_types:
            exp = " ou ".join(expected_types)
            raise ParseError(
                f"Esperado {exp}, encontrado '{tok.value}' ({tok.type})",
                tok
            )
        self.pos += 1
        return tok

    def _log(self, rule: str):
        self.derivation.append(rule)

    # ── Ponto de entrada ────────────────────────

    def parse(self) -> ASTNode:
        self._log("S → statement")
        node = self.parse_statement()
        if not self._is('EOF'):
            tok = self._current()
            raise ParseError(
                f"Token inesperado '{tok.value}' após fim da instrução",
                tok
            )
        return node

    def parse_statement(self) -> ASTNode:
        t = self._current().type
        if t == 'SELECT': return self.parse_select()
        if t == 'INSERT': return self.parse_insert()
        if t == 'UPDATE': return self.parse_update()
        if t == 'DELETE': return self.parse_delete()
        if t == 'CREATE': return self.parse_create()
        if t == 'DROP':   return self.parse_drop()
        raise ParseError(
            f"Instrução SQL desconhecida: '{self._current().value}'",
            self._current()
        )

    # ── SELECT ──────────────────────────────────

    def parse_select(self) -> ASTNode:
        self._log("select_stmt → SELECT col_list FROM table_ref ...")
        node = ASTNode("SELECT")
        self.consume('SELECT')

        # DISTINCT opcional
        if self._is('DISTINCT'):
            node.children.append(ASTNode("DISTINCT"))
            self.consume('DISTINCT')

        node.children.append(self.parse_col_list())
        self.consume('FROM')
        node.children.append(self.parse_table_ref())

        # JOIN opcional (múltiplos)
        while self._is('JOIN', 'INNER', 'LEFT', 'RIGHT'):
            node.children.append(self.parse_join())

        if self._is('WHERE'):
            node.children.append(self.parse_where())

        if self._is('GROUP'):
            node.children.append(self.parse_group())

        if self._is('HAVING'):
            node.children.append(self.parse_having())

        if self._is('ORDER'):
            node.children.append(self.parse_order())

        if self._is('LIMIT'):
            node.children.append(self.parse_limit())

        if self._is('SEMICOLON'):
            self.consume('SEMICOLON')

        return node

    # ── INSERT ──────────────────────────────────

    def parse_insert(self) -> ASTNode:
        self._log("insert_stmt → INSERT INTO IDENTIFIER ...")
        node = ASTNode("INSERT")
        self.consume('INSERT')
        self.consume('INTO')
        node.children.append(ASTNode("TABLE", value=self.consume('IDENTIFIER').value))

        if self._is('LPAREN'):
            self.consume('LPAREN')
            node.children.append(self.parse_col_list())
            self.consume('RPAREN')

        self.consume('VALUES')
        self.consume('LPAREN')
        node.children.append(ASTNode("VALUES", children=self.parse_value_list()))
        self.consume('RPAREN')

        if self._is('SEMICOLON'): self.consume('SEMICOLON')
        return node

    # ── UPDATE ──────────────────────────────────

    def parse_update(self) -> ASTNode:
        self._log("update_stmt → UPDATE IDENTIFIER SET assignment_list ...")
        node = ASTNode("UPDATE")
        self.consume('UPDATE')
        node.children.append(ASTNode("TABLE", value=self.consume('IDENTIFIER').value))
        self.consume('SET')
        node.children.append(ASTNode("SET", children=self.parse_assignment_list()))
        if self._is('WHERE'):
            node.children.append(self.parse_where())
        if self._is('SEMICOLON'): self.consume('SEMICOLON')
        return node

    # ── DELETE ──────────────────────────────────

    def parse_delete(self) -> ASTNode:
        self._log("delete_stmt → DELETE FROM IDENTIFIER ...")
        node = ASTNode("DELETE")
        self.consume('DELETE')
        self.consume('FROM')
        node.children.append(ASTNode("TABLE", value=self.consume('IDENTIFIER').value))
        if self._is('WHERE'):
            node.children.append(self.parse_where())
        if self._is('SEMICOLON'): self.consume('SEMICOLON')
        return node

    # ── CREATE TABLE ────────────────────────────

    def parse_create(self) -> ASTNode:
        self._log("create_stmt → CREATE TABLE IDENTIFIER LPAREN col_def_list RPAREN")
        node = ASTNode("CREATE_TABLE")
        self.consume('CREATE')
        self.consume('TABLE')
        node.children.append(ASTNode("TABLE", value=self.consume('IDENTIFIER').value))
        self.consume('LPAREN')
        node.children.append(ASTNode("COLUMNS", children=self.parse_col_def_list()))
        self.consume('RPAREN')
        if self._is('SEMICOLON'): self.consume('SEMICOLON')
        return node

    # ── DROP TABLE ──────────────────────────────

    def parse_drop(self) -> ASTNode:
        self._log("drop_stmt → DROP TABLE IDENTIFIER")
        node = ASTNode("DROP_TABLE")
        self.consume('DROP')
        self.consume('TABLE')
        node.children.append(ASTNode("TABLE", value=self.consume('IDENTIFIER').value))
        if self._is('SEMICOLON'): self.consume('SEMICOLON')
        return node

    # ── Colunas / expressões ────────────────────

    def parse_col_list(self) -> ASTNode:
        self._log("col_list → expr (COMMA expr)* | STAR")
        node = ASTNode("COL_LIST")
        if self._is('STAR'):
            node.children.append(ASTNode("STAR", value="*"))
            self.consume('STAR')
        else:
            node.children.append(self.parse_aliased_expr())
            while self._is('COMMA'):
                self.consume('COMMA')
                node.children.append(self.parse_aliased_expr())
        return node

    def parse_aliased_expr(self) -> ASTNode:
        expr = self.parse_expr()
        if self._is('AS'):
            self.consume('AS')
            alias = self.consume('IDENTIFIER').value
            return ASTNode("ALIAS", children=[expr], value=alias)
        return expr

    def parse_expr(self) -> ASTNode:
        self._log("expr → term ((PLUS|MINUS) term)*")
        left = self.parse_term()
        while self._is('PLUS', 'MINUS'):
            op = self.consume().value
            right = self.parse_term()
            left = ASTNode("BINOP", children=[left, right], value=op)
        return left

    def parse_term(self) -> ASTNode:
        self._log("term → factor ((STAR|SLASH) factor)*")
        left = self.parse_factor()
        while self._is('STAR', 'SLASH'):
            op = self.consume().value
            right = self.parse_factor()
            left = ASTNode("BINOP", children=[left, right], value=op)
        return left

    def parse_factor(self) -> ASTNode:
        self._log("factor → NUMBER | STRING | IDENTIFIER | LPAREN expr RPAREN | NULL")
        tok = self._current()

        if tok.type == 'NUMBER':
            self.consume('NUMBER')
            return ASTNode("NUMBER", value=tok.value)

        if tok.type == 'STRING':
            self.consume('STRING')
            return ASTNode("STRING", value=tok.value)

        if tok.type == 'NULL':
            self.consume('NULL')
            return ASTNode("NULL")

        if tok.type == 'IDENTIFIER':
            self.consume('IDENTIFIER')
            # função: IDENTIFIER(...)
            if self._is('LPAREN'):
                self.consume('LPAREN')
                args = []
                if not self._is('RPAREN'):
                    args.append(self.parse_expr())
                    while self._is('COMMA'):
                        self.consume('COMMA')
                        args.append(self.parse_expr())
                self.consume('RPAREN')
                return ASTNode("FUNC_CALL", children=args, value=tok.value)
            # qualificado: tabela.coluna
            if self._is('DOT'):
                self.consume('DOT')
                col = self.consume('IDENTIFIER').value
                return ASTNode("QUALIFIED", value=f"{tok.value}.{col}")
            return ASTNode("IDENTIFIER", value=tok.value)

        if tok.type == 'STAR':
            self.consume('STAR')
            return ASTNode("STAR", value="*")

        if tok.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expr()
            self.consume('RPAREN')
            return ASTNode("GROUP", children=[expr])

        # Permite keywords como nomes de coluna em alguns contextos
        if tok.type in ('DESC', 'ASC', 'KEY'):
            self.consume()
            return ASTNode("IDENTIFIER", value=tok.value)

        raise ParseError(
            f"Expressão inesperada: '{tok.value}' ({tok.type})",
            tok
        )

    # ── Condições ───────────────────────────────

    def parse_condition(self) -> ASTNode:
        self._log("condition → cond ((AND|OR) cond)*")
        left = self.parse_cond()
        while self._is('AND', 'OR'):
            op = self.consume().type
            right = self.parse_cond()
            left = ASTNode("LOGIC", children=[left, right], value=op)
        return left

    def parse_cond(self) -> ASTNode:
        self._log("cond → expr op expr | expr IS [NOT] NULL | NOT cond | ...")
        if self._is('NOT'):
            self.consume('NOT')
            return ASTNode("NOT", children=[self.parse_cond()])

        if self._is('LPAREN'):
            self.consume('LPAREN')
            node = self.parse_condition()
            self.consume('RPAREN')
            return node

        left = self.parse_expr()

        # IS [NOT] NULL
        if self._is('IS'):
            self.consume('IS')
            neg = False
            if self._is('NOT'):
                self.consume('NOT')
                neg = True
            self.consume('NULL')
            op = "IS NOT NULL" if neg else "IS NULL"
            return ASTNode("UNARY_COND", children=[left], value=op)

        # [NOT] IN
        if self._is('NOT') or self._is('IN'):
            neg = False
            if self._is('NOT'):
                self.consume('NOT')
                neg = True
            self.consume('IN')
            self.consume('LPAREN')
            vals = self.parse_value_list()
            self.consume('RPAREN')
            op = "NOT IN" if neg else "IN"
            return ASTNode("IN_COND", children=[left] + vals, value=op)

        # [NOT] LIKE
        if self._is('LIKE'):
            self.consume('LIKE')
            pattern = self.consume('STRING')
            return ASTNode("LIKE_COND", children=[left], value=pattern.value)

        # BETWEEN
        if self._is('BETWEEN'):
            self.consume('BETWEEN')
            lo = self.parse_expr()
            self.consume('AND')
            hi = self.parse_expr()
            return ASTNode("BETWEEN", children=[left, lo, hi])

        # Operadores relacionais
        op_types = ('EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE')
        if self._is(*op_types):
            op = self.consume().value
            right = self.parse_expr()
            return ASTNode("COMPARE", children=[left, right], value=op)

        raise ParseError(
            f"Condição inválida próximo a '{self._current().value}'",
            self._current()
        )

    # ── Cláusulas ───────────────────────────────

    def parse_where(self) -> ASTNode:
        self._log("where_clause → WHERE condition")
        self.consume('WHERE')
        return ASTNode("WHERE", children=[self.parse_condition()])

    def parse_group(self) -> ASTNode:
        self._log("group_clause → GROUP BY col_list")
        self.consume('GROUP')
        self.consume('BY')
        return ASTNode("GROUP_BY", children=[self.parse_col_list()])

    def parse_having(self) -> ASTNode:
        self._log("having_clause → HAVING condition")
        self.consume('HAVING')
        return ASTNode("HAVING", children=[self.parse_condition()])

    def parse_order(self) -> ASTNode:
        self._log("order_clause → ORDER BY order_list")
        self.consume('ORDER')
        self.consume('BY')
        node = ASTNode("ORDER_BY")
        node.children.append(self.parse_order_item())
        while self._is('COMMA'):
            self.consume('COMMA')
            node.children.append(self.parse_order_item())
        return node

    def parse_order_item(self) -> ASTNode:
        expr = self.parse_expr()
        direction = 'ASC'
        if self._is('ASC', 'DESC'):
            direction = self.consume().value.upper()
        return ASTNode("ORDER_ITEM", children=[expr], value=direction)

    def parse_limit(self) -> ASTNode:
        self._log("limit_clause → LIMIT NUMBER [OFFSET NUMBER]")
        self.consume('LIMIT')
        n = self.consume('NUMBER').value
        node = ASTNode("LIMIT", value=n)
        if self._is('OFFSET'):
            self.consume('OFFSET')
            node.children.append(ASTNode("OFFSET", value=self.consume('NUMBER').value))
        return node

    def parse_join(self) -> ASTNode:
        self._log("join_clause → [INNER|LEFT|RIGHT] JOIN table_ref ON condition")
        join_type = 'INNER'
        if self._is('INNER', 'LEFT', 'RIGHT'):
            join_type = self.consume().value.upper()
        self.consume('JOIN')
        table = self.parse_table_ref()
        self.consume('ON')
        cond = self.parse_condition()
        return ASTNode("JOIN", children=[table, cond], value=join_type)

    # ── Referência de tabela ─────────────────────

    def parse_table_ref(self) -> ASTNode:
        self._log("table_ref → IDENTIFIER [AS IDENTIFIER]")
        name = self.consume('IDENTIFIER').value
        node = ASTNode("TABLE_REF", value=name)
        if self._is('AS'):
            self.consume('AS')
            node.children.append(ASTNode("ALIAS", value=self.consume('IDENTIFIER').value))
        elif self._is('IDENTIFIER'):
            # alias sem AS
            node.children.append(ASTNode("ALIAS", value=self.consume('IDENTIFIER').value))
        return node

    # ── Listas de valores / atribuições ─────────

    def parse_value_list(self) -> List[ASTNode]:
        vals = [self.parse_expr()]
        while self._is('COMMA'):
            self.consume('COMMA')
            vals.append(self.parse_expr())
        return vals

    def parse_assignment_list(self) -> List[ASTNode]:
        assignments = [self.parse_assignment()]
        while self._is('COMMA'):
            self.consume('COMMA')
            assignments.append(self.parse_assignment())
        return assignments

    def parse_assignment(self) -> ASTNode:
        self._log("assignment → IDENTIFIER EQ expr")
        col = self.consume('IDENTIFIER').value
        self.consume('EQ')
        val = self.parse_expr()
        return ASTNode("ASSIGN", children=[val], value=col)

    # ── Definição de colunas (CREATE TABLE) ─────

    def parse_col_def_list(self) -> List[ASTNode]:
        defs = [self.parse_col_def()]
        while self._is('COMMA'):
            self.consume('COMMA')
            if self._is('PRIMARY', 'UNIQUE', 'FOREIGN'):
                defs.append(self.parse_table_constraint())
            else:
                defs.append(self.parse_col_def())
        return defs

    def parse_col_def(self) -> ASTNode:
        self._log("col_def → IDENTIFIER type_def [constraints]")
        name = self.consume('IDENTIFIER').value
        dtype = self.parse_type_def()
        constraints = self.parse_col_constraints()
        return ASTNode("COL_DEF", children=[dtype] + constraints, value=name)

    def parse_type_def(self) -> ASTNode:
        self._log("type_def → INT | VARCHAR(...) | TEXT | FLOAT | BOOLEAN | DATE")
        type_tokens = ('INT', 'VARCHAR', 'TEXT', 'FLOAT', 'BOOLEAN', 'DATE', 'IDENTIFIER')
        tok = self.consume(*type_tokens)
        node = ASTNode("TYPE", value=tok.value.upper())
        if tok.value.upper() == 'VARCHAR' and self._is('LPAREN'):
            self.consume('LPAREN')
            node.children.append(ASTNode("SIZE", value=self.consume('NUMBER').value))
            self.consume('RPAREN')
        return node

    def parse_col_constraints(self) -> List[ASTNode]:
        constraints = []
        while True:
            if self._is('PRIMARY'):
                self.consume('PRIMARY')
                self.consume('KEY')
                constraints.append(ASTNode("PRIMARY_KEY"))
            elif self._is('NOT'):
                self.consume('NOT')
                self.consume('NULL')
                constraints.append(ASTNode("NOT_NULL"))
            elif self._is('UNIQUE'):
                self.consume('UNIQUE')
                constraints.append(ASTNode("UNIQUE"))
            elif self._is('DEFAULT'):
                self.consume('DEFAULT')
                constraints.append(ASTNode("DEFAULT", children=[self.parse_factor()]))
            else:
                break
        return constraints

    def parse_table_constraint(self) -> ASTNode:
        if self._is('PRIMARY'):
            self.consume('PRIMARY')
            self.consume('KEY')
            self.consume('LPAREN')
            cols = [self.consume('IDENTIFIER').value]
            while self._is('COMMA'):
                self.consume('COMMA')
                cols.append(self.consume('IDENTIFIER').value)
            self.consume('RPAREN')
            return ASTNode("TABLE_PK", value=", ".join(cols))
        if self._is('UNIQUE'):
            self.consume('UNIQUE')
            self.consume('LPAREN')
            col = self.consume('IDENTIFIER').value
            self.consume('RPAREN')
            return ASTNode("TABLE_UNIQUE", value=col)
        if self._is('FOREIGN'):
            self.consume('FOREIGN')
            self.consume('KEY')
            self.consume('LPAREN')
            col = self.consume('IDENTIFIER').value
            self.consume('RPAREN')
            self.consume('REFERENCES')
            ref_table = self.consume('IDENTIFIER').value
            self.consume('LPAREN')
            ref_col = self.consume('IDENTIFIER').value
            self.consume('RPAREN')
            return ASTNode("FOREIGN_KEY", value=f"{col} → {ref_table}.{ref_col}")
        raise ParseError("Restrição de tabela inválida", self._current())


def parse(tokens) -> tuple:
    """Ponto de entrada público. Retorna (ASTNode, derivation_steps)."""
    p = Parser(tokens)
    ast = p.parse()
    return ast, p.derivation
