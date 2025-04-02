/**
 * l.c
 * Lambda Calculus Interpreter
 *
 * Noah Trupin
 * May 4, 2024
 *
 * Implementation of an untyped lambda calculus in a single .c file.
 * Contains a lexer, parser (produces abstract-syntax tree), and evaluator.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRUE (1)
#define FALSE (0)

#define VAR_DLEN (1)
#define ABSTR_DLEN (2)
#define APPL_DLEN (2)

static char *node_type[3] = {
  "var", "abstr", "appl"
};

typedef struct var var_t;
typedef struct abstr abstr_t;
typedef struct appl appl_t;
typedef struct node node_t;

void destroy_var(var_t **);
var_t *create_var(const char *);
void destroy_abstr(abstr_t **);
abstr_t *create_abstr(node_t *, node_t *);
void destroy_appl(appl_t **);
appl_t *create_appl(node_t *, node_t *);
void destroy_node(node_t **);
node_t *create_node(char, void **, int);

typedef struct var {
  int len;
  char *literal;
} var_t;

typedef struct abstr {
  node_t *var;
  node_t *body;
} abstr_t;

typedef struct appl {
  node_t *fn;
  node_t *arg;
} appl_t;

typedef struct node {
  char flag;
  union {
    var_t *var;
    abstr_t *abstr;
    appl_t *appl;
  };
} node_t;

enum {
  NODE_VAR, NODE_ABSTR, NODE_APPL
};

void destroy_var(var_t **var) {
  assert(var != NULL);

  if (!(*var)) return;

  if (!((*var)->literal)) {
    free((*var)->literal);
    (*var)->literal = NULL;
  }

  free(*var);
  *var = NULL;
}

var_t *create_var(const char *literal) {
  var_t *var = malloc(sizeof(var_t));
  if (!var) goto cleanup;

  var->len = strlen(literal);
  
  var->literal = malloc(sizeof(char) * (var->len + 1));
  if (!var->literal) goto cleanup;
  strncpy(var->literal, literal, var->len);

  return var;

cleanup:
  destroy_var(&var);
  return NULL;
}

void destroy_abstr(abstr_t **abstr) {
  assert(abstr != NULL);

  if (!(*abstr)) return;

  destroy_node(&((*abstr)->var));
  destroy_node(&((*abstr)->body));

  free(*abstr);
  *abstr = NULL;
}

abstr_t *create_abstr(node_t *var, node_t *body) {
  abstr_t *abstr = malloc(sizeof(abstr_t));
  if (!abstr) goto cleanup;

  abstr->var = var;
  abstr->body = body;

  return abstr;

cleanup:
  destroy_abstr(&abstr);
  return NULL;
}

void destroy_appl(appl_t **appl) {
  assert(appl != NULL);

  if (!(*appl)) return;

  destroy_node(&((*appl)->fn));
  destroy_node(&((*appl)->arg));

  free(*appl);
  *appl = NULL;
}

appl_t *create_appl(node_t *fn, node_t *arg) {
  assert(fn != NULL);
  assert(arg != NULL);

  appl_t *appl = malloc(sizeof(appl_t));
  if (!appl) goto cleanup;

  appl->fn = fn;
  appl->arg = arg;

  return appl;

cleanup:
  destroy_appl(&appl);
  return NULL;
}

void destroy_node(node_t **node) {
  if (!node) return;

  switch ((*node)->flag) {
    case NODE_VAR:
      destroy_var(&((*node)->var));
      break;
    case NODE_APPL:
      destroy_appl(&((*node)->appl));
      break;
    case NODE_ABSTR:
      destroy_abstr(&((*node)->abstr));
      break;
  }

  free(*node);
  *node = NULL;
}

node_t *create_node(char flag, void **data, int dlen) {
  node_t *node = malloc(sizeof(node_t));
  if (!node) goto cleanup;

  node->flag = flag;

  switch (node->flag) {
    case NODE_VAR:
      if (dlen != VAR_DLEN) 
        goto cleanup;
      node->var = create_var((char *)(*data));
      if (!(node->var))
        goto cleanup;
      break;
    case NODE_APPL:
      if (dlen != APPL_DLEN)
        goto cleanup;
      node->appl = create_appl((node_t *)(*data), (node_t *)(*(data + 1)));
      if (!(node->appl))
        goto cleanup;
      break;
    case NODE_ABSTR:
      if (dlen != ABSTR_DLEN) 
        goto cleanup;
      node->abstr = create_abstr((node_t *)(*data), (node_t *)(*(data + 1)));
      if (!(node->abstr))
        goto cleanup;
      break;
  }

  return node;

cleanup:
  destroy_node(&node);
  return NULL;
}

#define debug_node(node) _debug_node(node, 0)
void _debug_node(node_t *node, int depth) {
  if (!node) return;

  printf("%*c| %s", 4 * depth, ' ', node_type[node->flag]);

  switch (node->flag) {
    case NODE_VAR:
      printf(" '%*s'\n", node->var->len, node->var->literal);
      break;
    case NODE_ABSTR:
      printf("\n");
      _debug_node(node->abstr->var, depth + 1);
      _debug_node(node->abstr->body, depth + 1);
      break;
    case NODE_APPL:
      printf("\n");
      _debug_node(node->appl->fn, depth + 1);
      _debug_node(node->appl->arg, depth + 1);
      break;
  }

}

node_t *parse_expr(char **input) {
  node_t *node = NULL;

  switch (**input) {
    case '(': {
      *input += 1;
      node_t *fn = parse_expr(input);
      if (**input == ')') {
        *input += 1;
      } else {
        fprintf(stderr, "expected )\n");
        return NULL;
      }
      node_t *arg = parse_expr(input);

      void *appl_data[] = { (void *)fn, (void *)arg };
      node = create_node(NODE_APPL, appl_data, APPL_DLEN);
      break;
    }
    case '\\': {
      *input += 1;
      node_t *var = parse_expr(input);
      if (**input == '.') {
        *input += 1;
      } else {
        fprintf(stderr, "expected .\n");
        return NULL;
      }
      node_t *body = parse_expr(input);

      void *abstr_data[] = { (void *)var, (void *)body };
      node = create_node(NODE_ABSTR, abstr_data, ABSTR_DLEN);
      break;
    }
    case '\0': return NULL;
    default: {
      char lit[2] = { **input, '\0' };
      *input += 1;

      void *var_data[] = { (void *)(lit) };
      node = create_node(NODE_VAR, var_data, VAR_DLEN);
      break;
    }
  }

  return node;
}

void reduce(node_t **expr, var_t *var, node_t *node) {
  node_t *tmp;
  switch ((*expr)->flag) {
    case NODE_VAR:
      if (strncmp(var->literal, (*expr)->var->literal, var->len) == 0 &&
        var->len == (*expr)->var->len) {
        tmp = *expr;
        *expr = node;
        destroy_node(&tmp);
      }
      break;
    case NODE_ABSTR:
      if (strncmp(var->literal, (*expr)->abstr->var->var->literal, var->len) == 0 &&
        var->len == (*expr)->abstr->var->var->len) {
        return;
      }
      reduce(&((*expr)->abstr->body), var, node);
      break;
  }
}

void eval(node_t **input) {
  switch ((*input)->flag) {
    case NODE_VAR:
      return;
    case NODE_ABSTR:
      return;
    case NODE_APPL:
      if ((*input)->appl->fn->flag != NODE_ABSTR) {
        return;
      }
      eval(&((*input)->appl->arg));
      reduce(&((*input)->appl->fn->abstr->body),
        (*input)->appl->fn->abstr->var->var, (*input)->appl->arg);
      *input = (*input)->appl->fn->abstr->body;
      return;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "not enough arguments\n");
    return 1;
  }

  node_t *node = parse_expr(&(argv[1]));
  debug_node(node);
  eval(&node);
  debug_node(node);
  destroy_node(&node);

  return 0;
}
