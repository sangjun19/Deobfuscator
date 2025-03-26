// Repository: SuperChaoM/iPhone15-3_17.6.1_21G101_Restore
// File: System/Library/PrivateFrameworks/SpringBoard.framework/SBInsertionSwitcherModifier.m

@implementation SBInsertionSwitcherModifier

- (uint64_t)initWithAppLayout:(void *)a3 
{
  uint64_t v6;
  uint64_t v7;
  uint64_t v9;
  _QWORD v10[2];

  v6 = MEMORY[0x1DB1D4980]();
  v10[0] = a1;
  v10[1] = off_1F3665370;
  v7 = MEMORY[0x1DB1D4770](v10, 0x182997FD5uLL);
  if (v7)
  {
    if (!v6)
    {
      v9 = objc_msgSend((id)MEMORY[0x1DB1D4690](objc_msgSend(off_1F365C208, "currentHandler")), "handleFailureInMethod:object:file:lineNumber:description:", a2, v7, &stru_1F36E9EB0, 31LL, &stru_1F36960F0, &stru_1F369D7D0);
      MEMORY[0x1DB1D4860](v9);
    }
    objc_storeStrong((id *)(v7 + 96), a3);
    *(_QWORD *)(v7 + 128) = 0LL;
  }
  MEMORY[0x1DB1D4810]();
  return v7;
}

- (void)handleInsertionEvent:(uint64_t)a1 
{
  void *v2;
  uint64_t v3;
  uint64_t v4;
  int v5;
  uint64_t v6;
  uint64_t v7;
  uint64_t v8;
  void *v9;
  uint64_t v10;
  uint64_t v11;
  uint64_t v12;
  void *v13;
  uint64_t v14;
  uint64_t v15;
  uint64_t v16;
  uint64_t v17;
  _QWORD v18[7];
  uint64_t v19;
  uint64_t *v20;
  uint64_t v21;
  uint64_t v22;
  _QWORD v23[2];
  uint64_t vars8;

  v2 = (void *)MEMORY[0x1DB1D4970]();
  v23[0] = a1;
  v23[1] = off_1F3665370;
  v3 = MEMORY[0x1DB1D4770](v23, 0x1845B5B6EuLL, v2);
  v4 = MEMORY[0x1DB1D4690](v3);
  v5 = objc_msgSend(*(id *)(a1 + 96), "isEqual:", MEMORY[0x1DB1D4690](objc_msgSend(v2, "appLayout")));
  v6 = MEMORY[0x1DB1D4860]();
  if (v5)
  {
    v6 = objc_msgSend(v2, "phase");
    *(_QWORD *)(a1 + 128) = v6;
    switch(v6)
    {
      case 4LL:
        v6 = objc_msgSend((id)a1, "setState:", 1LL);
        break;
      case 2LL:
        v9 = (void *)MEMORY[0x1DB1D4690](objc_msgSend((id)a1, "appLayouts"));
        v10 = objc_msgSend(v9, "count");
        if (v10)
        {
          v19 = 0LL;
          v20 = &v19;
          v21 = 0x2020000000LL;
          v22 = 0x7FFFFFFFFFFFFFFFLL;
          v18[0] = _NSConcreteStackBlock;
          v18[1] = 3221225472LL;
          v18[2] = __52__SBInsertionSwitcherModifier_handleInsertionEvent___block_invoke;
          v18[3] = &unk_1F3420A68;
          v18[6] = &v19;
          v18[4] = a1;
          v18[5] = MEMORY[0x1DB1D4960]();
          objc_msgSend((id)a1, "_performBlockBySimulatingPreInsertionState:", v18);
          v11 = objc_msgSend(v9, "count");
          v12 = v11 - 1;
          if (v11 - 1 >= (unint64_t)v20[3])
            v12 = v20[3];
          v20[3] = v12;
          if (v12 != 0x7FFFFFFFFFFFFFFFLL)
          {
            v13 = (void *)MEMORY[0x1DB1D4690](objc_msgSend((id)a1, "appLayouts"));
            v14 = MEMORY[0x1DB1D4690](objc_msgSend(v13, "objectAtIndex:", v20[3]));
            MEMORY[0x1DB1D4860]();
            v15 = objc_msgSend(off_1F365C638, "responseByAppendingResponse:toResponse:", objc_msgSend((id)MEMORY[0x1DB1D4610](off_1F365C640), "initWithAppLayout:alignment:animated:", v14, 0LL, 0LL), v4);
            MEMORY[0x1DB1D4690](v15);
            v16 = MEMORY[0x1DB1D4830]();
            v17 = MEMORY[0x1DB1D4870](v16);
            v11 = MEMORY[0x1DB1D4850](v17);
          }
          MEMORY[0x1DB1D48D0](v11);
          v10 = MEMORY[0x1DB1D39C0](&v19, 8LL);
        }
        v6 = MEMORY[0x1DB1D4840](v10);
        break;
      case 1LL:
        v6 = objc_msgSend((id)a1, "scrollViewContentOffset");
        *(_QWORD *)(a1 + 112) = v7;
        *(_QWORD *)(a1 + 120) = v8;
        break;
    }
  }
  MEMORY[0x1DB1D4810](v6);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  JUMPOUT(0x1DB1D4670LL);
}

uint64_t __52__SBInsertionSwitcherModifier_handleInsertionEvent___block_invoke(uint64_t a1)
{
  uint64_t result;

  result = objc_msgSend(*(id *)(a1 + 32), "indexToScrollToAfterInsertingAtIndex:", objc_msgSend(*(id *)(a1 + 40), "index"));
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL) + 24LL) = result;
  return result;
}

- (uint64_t)scrollViewContentOffset
{
  _QWORD v1[2];

  if (!*(_BYTE *)(result + 104))
  {
    v1[0] = result;
    v1[1] = off_1F3665370;
    return MEMORY[0x1DB1D4770](v1, 0x181B016ECuLL);
  }
  return result;
}

- (void)visibleAppLayouts
{
  uint64_t v2;
  void *v3;
  uint64_t v4;
  _QWORD v5[6];
  uint64_t v6;
  uint64_t *v7;
  uint64_t v8;
  uint64_t (*v9)(uint64_t, uint64_t);
  void (*v10)();
  uint64_t v11;
  _QWORD v12[2];
  uint64_t vars8;

  v12[0] = a1;
  v12[1] = off_1F3665370;
  v2 = MEMORY[0x1DB1D4770](v12, 0x181FD1955uLL);
  v3 = (void *)MEMORY[0x1DB1D4690](v2);
  if (a1[16] > 1uLL)
  {
    v6 = 0LL;
    v7 = &v6;
    v8 = 0x3032000000LL;
    v9 = __Block_byref_object_copy__108;
    v10 = __Block_byref_object_dispose__108;
    v11 = 0LL;
    v5[0] = _NSConcreteStackBlock;
    v5[1] = 3221225472LL;
    v5[2] = __48__SBInsertionSwitcherModifier_visibleAppLayouts__block_invoke;
    v5[3] = &unk_1F341F9A8;
    v5[4] = a1;
    v5[5] = &v6;
    objc_msgSend(a1, "_performBlockBySimulatingPreInsertionState:", v5);
    MEMORY[0x1DB1D4690](objc_msgSend(v3, "setByAddingObjectsFromSet:", v7[5]));
    v4 = MEMORY[0x1DB1D39C0](&v6, 8LL);
    MEMORY[0x1DB1D48D0](v4);
  }
  else
  {
    MEMORY[0x1DB1D4960]();
  }
  MEMORY[0x1DB1D4810]();
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  JUMPOUT(0x1DB1D4670LL);
}

uint64_t __48__SBInsertionSwitcherModifier_visibleAppLayouts__block_invoke(uint64_t a1)
{
  uint64_t v2;
  _QWORD v4[2];

  v4[0] = *(_QWORD *)(a1 + 32);
  v4[1] = off_1F3665370;
  v2 = MEMORY[0x1DB1D4770](v4, 0x181FD1955uLL);
  *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL) + 40LL) = MEMORY[0x1DB1D4690](v2);
  return MEMORY[0x1DB1D48E0]();
}

- (void)animationAttributesForLayoutElement:(void *)a1 
{
  uint64_t v2;
  void *v3;
  uint64_t v4;
  uint64_t v5;
  uint64_t v6;
  uint64_t v7;
  uint64_t v8;
  _QWORD v9[2];
  uint64_t vars8;

  v9[0] = a1;
  v9[1] = off_1F3665370;
  v2 = MEMORY[0x1DB1D4770](v9, 0x181C80855uLL);
  v3 = (void *)objc_msgSend((id)MEMORY[0x1DB1D4690](v2), "mutableCopy");
  MEMORY[0x1DB1D4830]();
  objc_msgSend(v3, "setUpdateMode:", 3LL);
  v4 = objc_msgSend((id)MEMORY[0x1DB1D4690](objc_msgSend(a1, "switcherSettings")), "animationSettings");
  v5 = objc_msgSend((id)MEMORY[0x1DB1D4690](v4), "opacitySettings");
  v6 = objc_msgSend(v3, "setOpacitySettings:", MEMORY[0x1DB1D4690](v5));
  v7 = MEMORY[0x1DB1D4850](v6);
  v8 = MEMORY[0x1DB1D4830](v7);
  MEMORY[0x1DB1D4810](v8);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  JUMPOUT(0x1DB1D4670LL);
}

- (uint64_t)clipsToUnobscuredMarginAtIndex:
{
  return 0LL;
}

- (double)opacityForLayoutRole:(uint64_t)a3 inAppLayout:(uint64_t)a4 atIndex:(uint64_t)a5 
{
  uint64_t v8;
  uint64_t v9;
  double v10;
  _QWORD v12[2];

  v8 = MEMORY[0x1DB1D4A10]();
  if (a1[16] != 2LL
    || (v9 = objc_msgSend((id)MEMORY[0x1DB1D4690](objc_msgSend(a1, "appLayouts")), "indexOfObject:", a1[12]),
        MEMORY[0x1DB1D4860](),
        v10 = 0.0,
        v9 != a5) )
  {
    v12[0] = a1;
    v12[1] = off_1F3665370;
    v10 = MEMORY[0x1DB1D4770](v12, 0x1820DC9D5uLL, a3, v8, a5);
  }
  MEMORY[0x1DB1D4810]();
  return v10;
}

- (uint64_t)_performBlockBySimulatingPreInsertionState:(uint64_t)a1 
{
  uint64_t v2;
  char v3;
  void *v4;
  void *v5;
  uint64_t v6;
  uint64_t v7;
  uint64_t v8;
  uint64_t v9;
  uint64_t v10;
  _QWORD v12[5];

  v2 = MEMORY[0x1DB1D4970]();
  v3 = *(_BYTE *)(a1 + 104);
  *(_BYTE *)(a1 + 104) = 1;
  v4 = (void *)MEMORY[0x1DB1D4610](off_1F365C470);
  v5 = (void *)objc_msgSend(v4, "initWithArray:", MEMORY[0x1DB1D4690](objc_msgSend((id)a1, "appLayouts")));
  MEMORY[0x1DB1D4850]();
  objc_msgSend(v5, "removeObject:", *(_QWORD *)(a1 + 96));
  v6 = objc_msgSend((id)MEMORY[0x1DB1D4610](off_1F365EF78), "initWithAppLayouts:", v5);
  v12[0] = _NSConcreteStackBlock;
  v12[1] = 3221225472LL;
  v12[2] = __74__SBInsertionSwitcherModifier__performBlockBySimulatingPreInsertionState___block_invoke;
  v12[3] = &unk_1F34200F8;
  v12[4] = v2;
  MEMORY[0x1DB1D4990]();
  v7 = objc_msgSend((id)a1, "performTransactionWithTemporaryChildModifier:usingBlock:", v6, v12);
  *(_BYTE *)(a1 + 104) = v3;
  v8 = MEMORY[0x1DB1D48D0](v7);
  v9 = MEMORY[0x1DB1D4840](v8);
  v10 = MEMORY[0x1DB1D4850](v9);
  return MEMORY[0x1DB1D4830](v10);
}

uint64_t __74__SBInsertionSwitcherModifier__performBlockBySimulatingPreInsertionState___block_invoke(uint64_t a1)
{
  return (*(uint64_t (**)(void))(*(_QWORD *)(a1 + 32) + 16LL))();
}

- (uint64_t)phase
{
  return *(_QWORD *)(a1 + 128);
}

- (void).cxx_destruct
{
  objc_storeStrong((id *)(a1 + 96), 0LL);
}

@end
