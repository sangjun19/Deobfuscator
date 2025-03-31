#ifndef NN_MATH_H_
#define NN_MATH_H_

/* ---------
 * Includes:
 * --------- */
#include "nn_common_includes.h"
#include <math.h>

/* ----------------------
 * Function Declarations:
 * ---------------------- */
extern float
actf(float x, nn_act_func_enum actf);
extern float
dactf(float y, nn_act_func_enum actf);
extern float
sigmoidf(float x);
extern float
dsigmoidf(float y);
extern float
p_reluf(float x);
extern float
dp_reluf(float y);
extern float
tanhf(float x);
extern float
dtanhf(float y);
extern float
eluf(float x);
extern float
deluf(float y);
extern float
sigluf(float x);
extern float
dsigluf(float x);
extern float
swishf(float x);
extern float
dswishf(float x);

#endif /* NN_MATH_H_ */

#ifdef NN_MATH_IMPLEMENTATION

float
actf(float x, nn_act_func_enum actf)
{
    switch (actf) {
    case ACT_SIGMOID:
        return sigmoidf(x);
    case ACT_P_RELU:
        return p_reluf(x);
    case ACT_TANH:
        return tanhf(x);
    case ACT_ELU:
        return eluf(x);
    }

    NN_ASSERT(0 && "Unreachable");
    return 0.0f;
}

float
dactf(float y, nn_act_func_enum actf)
{
    switch (actf) {
    case ACT_SIGMOID:
        return dsigmoidf(y);
    case ACT_P_RELU:
        return dp_reluf(y);
    case ACT_TANH:
        return dtanhf(y);
    case ACT_ELU:
        return deluf(y);
    }

    NN_ASSERT(0 && "Unreachable");
    return 0.0f;
}

float
sigmoidf(float x)
{
    return (1.f / (1.f + expf(-x)));
}

float
dsigmoidf(float y)
{
    return (y * (1 - y));
}

float
p_reluf(float x)
{
    return (x >= 0 ? x : x * NN_RELU_PARAM);
}

float
dp_reluf(float y)
{
    return (y >= 0 ? 1 : NN_RELU_PARAM);
}

float
tanhf(float x)
{
    float ex = expf(x);
    float enx = expf(-x);
    return (ex - enx) / (ex + enx);
}

float
dtanhf(float y)
{
    return (1 - y * y);
}

float
eluf(float x)
{
    return (x > 0 ? x : NN_ELU_PARAM * (expf(x) - 1));
}

float
deluf(float y)
{
    return (
        (y > 0 || (y == 0 && NN_ELU_PARAM == 1)) ? 1 : NN_ELU_PARAM * expf(y));
}

float
sigluf(float x)
{
    return (x / (1.f + expf(-x)));
}

float
dsigluf(float x)
{
    return ((1 + expf(-x) + x * expf(-x)) / powf((1 + expf(-x)), 2));
}

float
swishf(float x)
{
    return sigluf(x);
}

float
dswishf(float x)
{
    return dsigluf(x);
}

#endif /* NN_MATH_IMPLEMENTATION */
