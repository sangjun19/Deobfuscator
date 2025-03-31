/**
 * main.c
 *
 * The frontend for invoking the 'lexer', 'parser', and 'eval' phases.
 */

#include <stdio.h>
#include "lexer.h"
#include "token.h"
#include "parser.h"
#include "vec.h"
#include "eval.h"

VEC_DECLARE(token_t*, token);

void print_tree(node_t *node, size_t depth)
{
    switch (node->id) {
    case TOK_NUMBER:
        printf("(%zd): %ld\n", depth, node->value);
        break;
    default:
        printf("(%zd): %s\n", depth, __token_names[node->id]);
        break;
    }

    /* Need to change when adding unary operators */
    switch (node->arity) {
        case AST_BINARY:
            print_tree(node->right, depth + 1);
            /* Fallthrough */
        case AST_UNARY:
            print_tree(node->left, depth + 1);
            break;
        case AST_NULLARY:
        default:
            ;
            /* Nothing */
    }
}

int main(void)
{
    lex_t *lctx = lex_init("test.c", FILE_BACKED);
    vec_token_t *tokens = vec_token_init();

    /* Construct lexemes from a file */
    token_t *tok;
    do {
        tok = lex_token(lctx);
        vec_token_push(tokens, tok);
    } while (tok->type != TOK_EOF);

    printf("\nTOKEN STREAM:\n-----------\n");
    for (size_t i = 0; i < tokens->len; ++i) {
        token_t *t = tokens->data[i];
        printf("%s: ", __token_names[t->type]);
        printf("%s\n", t->is_literal ? t->literal : "");
    }

    lex_free(lctx);

    /* Channel lexemes through parser */
    rdp_t *rctx = rdp_init(tokens->data, tokens->len);

    /* Get a root node for an AST */
    printf("\nAST:\n-----------\n");
    node_t *root = rdp_generate_ast(rctx);

    print_tree(root, 0);
    rdp_free(rctx);

    printf("\nEVAL:\n----------\n");
    eval_t *ectx = eval_init(root);

    printf("%ld\n", eval_compute(ectx));

    eval_free(ectx);

    return 0;
}
