lexer grammar ast_lexer;

WhiteSpaces: [ \t\n\r]+ -> skip;
Comment:    '/*' .*? '*/' -> skip;

IF:         'if';
ELSE:       'else';
FOR:        'for';
IN:         'in';
ASSERT_TOKEN:     'assert';
ASSUME:     'assume';
FUNC:       'func';

// empty
ANY:        'Any';

// VarDef
IO_TENSOR:  '@!io_tensor';
PINNED:     '@!pinned';

// ReduceTo
ATOMIC:     '@!atomic';
PLUSEQ:     '+=';
STAREQ:     '*=';
MINEQ:      '@!min=';
MAXEQ:      '@!max=';
ANDEQ:      '&&=';
OREQ:       '||=';

// For
NO_DEPS:    '@!no_deps';
PARALLEL:   '@!parallel';
REDUCTION:  '@!reduction';
UNROLL:     '@!unroll';
VECTORIZE:  '@!vectorize';
PREFERLIBS: '@!prefer_libs';

TRUE:       'true';
FALSE:      'false';

// expr
EVAL:       '@!eval';
FLOOR:      '@!floor';
CEIL:       '@!ceil';
ROUNDTO0:   '@!towards0';
MAX:        '@!max';
MIN:        '@!min';
SQRT:       '@!sqrt';
EXP:        '@!exp';
SQUARE:     '@!square';
ABS:        '@!abs';
SIGMOID:    '@!sigmoid';
TANH:       '@!tanh';
INTRINSIC:  '@!intrinsic';
SIDE_EFFECT:    '@!side_effect';
CLOSURE:    '@!closure';


Integer:    ('+'|'-')? [0-9]+;
Float:      ('+'|'-')? Integer '.' [0-9]* (('E'|'e') Integer)?;
String:     '"' ~["\r\n]+? '"';
SimpleVar:  [a-zA-Z_][a-zA-Z0-9_]*;
EscapedVar: '`' ~[`\r\n]+? '`';
AtVar:      '@' ~[ !\t\r\n]+;

DOT:        '.';
ASSIGN:     '=';
PLUS:       '+';
MINUS:      '-';
STAR:       '*';
SLASH:      '/';
PERCENT:    '%';
PERCENTPERCENT:    '%%';
NOT:        '!';
AND:        '&';
OR:         '|';
EQ:         '==';
NE:         '!=';
LT:         '<';
GT:         '>';
LE:         '<=';
GE:         '>=';
LAND:       '&&';
LOR:        '||';
COLON:      ':';
QUESTION:   '?';
LPAREN:     '(';
RPAREN:     ')';
LBRACK:     '[';
RBRACK:     ']';
LBRACE:     '{';
RBRACE:     '}';
COMMA:      ',';
RARROW:     '->';
