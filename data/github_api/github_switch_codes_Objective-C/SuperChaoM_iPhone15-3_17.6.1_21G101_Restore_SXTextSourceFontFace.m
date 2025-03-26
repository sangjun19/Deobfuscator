// Repository: SuperChaoM/iPhone15-3_17.6.1_21G101_Restore
// File: System/Library/PrivateFrameworks/Silex.framework/SXTextSourceFontFace.m

@implementation SXTextSourceFontFace

+ (void)fontFaceWithFontName:andAttributes:
{
  uint64_t v0;
  uint64_t v1;
  uint64_t v2;
  uint64_t v3;
  uint64_t v4;
  uint64_t vars8;

  v0 = MEMORY[0x20C5EF170]();
  v1 = MEMORY[0x20C5EF160]();
  v2 = MEMORY[0x20C5EEEC0](off_240038460);
  *(_QWORD *)(v2 + 8) = v0;
  MEMORY[0x20C5EF180]();
  v3 = MEMORY[0x20C5EF070]();
  *(_QWORD *)(v2 + 16) = v1;
  v4 = MEMORY[0x20C5EF0E0](v3);
  MEMORY[0x20C5EF050](v4);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  JUMPOUT(0x20C5EEEE0LL);
}

+ (void)fontFaceWithFontName:(void *)a1 
{
  uint64_t v2;
  uint64_t v3;
  uint64_t v4;
  uint64_t vars8;

  v2 = MEMORY[0x20C5EF170]();
  v3 = objc_msgSend(off_240038460, "fontFaceWithFontName:andAttributes:", v2, MEMORY[0x20C5EEF00](objc_msgSend(a1, "basicFontAttributesForFontName:", v2)));
  MEMORY[0x20C5EEF00](v3);
  v4 = MEMORY[0x20C5EF060]();
  MEMORY[0x20C5EF040](v4);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  JUMPOUT(0x20C5EEEE0LL);
}

+ (void)basicFontAttributesForFontName:
{
  void *v0;
  void *v1;
  uint64_t v2;
  uint64_t v3;
  uint64_t v4;
  uint64_t vars8;

  v0 = (void *)MEMORY[0x20C5EEF00](objc_msgSend(off_240036C60, "fontWithName:size:", 12.0));
  v1 = v0;
  if (v0)
  {
    v2 = objc_msgSend((id)MEMORY[0x20C5EEF00](objc_msgSend(v0, "fontDescriptor")), "symbolicTraits") & 1;
    MEMORY[0x20C5EF050]();
    if ((objc_msgSend((id)MEMORY[0x20C5EEF00](objc_msgSend(v1, "fontDescriptor")), "symbolicTraits") & 2) != 0)
      v3 = 700LL;
    else
      v3 = 400LL;
    MEMORY[0x20C5EF050]();
    v4 = objc_msgSend(off_240037280, "attributesWithFamilyName:style:weight:", MEMORY[0x20C5EEF00](objc_msgSend(v1, "familyName")), v2, v3);
    MEMORY[0x20C5EEF00](v4);
    v0 = (void *)MEMORY[0x20C5EF080]();
  }
  MEMORY[0x20C5EF040](v0);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  JUMPOUT(0x20C5EEEE0LL);
}

- (void)fontDescriptorAttributes
{
  uint64_t v2;
  const UIFontWeight *v3;
  uint64_t v4;
  const UIFontWeight *v5;
  uint64_t v6;
  double v7;
  uint64_t v8;
  uint64_t v9;
  uint64_t v10;
  uint64_t v11;
  uint64_t v12;
  uint64_t v13;
  uint64_t v14;
  _QWORD v15[3];
  _QWORD v16[3];
  _QWORD v17[2];
  _QWORD v18[2];
  uint64_t vars8;

  v17[0] = UIFontDescriptorNameAttribute;
  v18[0] = MEMORY[0x20C5EEF00](objc_msgSend(a1, "fontName"));
  v17[1] = UIFontDescriptorTraitsAttribute;
  v15[0] = UIFontWeightTrait;
  v2 = objc_msgSend((id)MEMORY[0x20C5EEF00](objc_msgSend(a1, "fontAttributes")), "weight");
  if (v2 <= 499)
  {
    if (v2 > 299)
    {
      if (v2 != 300)
      {
LABEL_15:
        v3 = &UIFontWeightRegular;
        goto LABEL_21;
      }
      v3 = &UIFontWeightLight;
    }
    else
    {
      if (v2 != 100)
      {
        if (v2 == 200)
        {
          v3 = &UIFontWeightUltraLight;
          goto LABEL_21;
        }
        goto LABEL_15;
      }
      v3 = &UIFontWeightThin;
    }
  }
  else if (v2 <= 699)
  {
    if (v2 != 500)
    {
      if (v2 == 600)
      {
        v3 = &UIFontWeightSemibold;
        goto LABEL_21;
      }
      goto LABEL_15;
    }
    v3 = &UIFontWeightMedium;
  }
  else
  {
    switch(v2)
    {
      case 700LL:
        v3 = &UIFontWeightBold;
        break;
      case 800LL:
        v3 = &UIFontWeightHeavy;
        break;
      case 900LL:
        v3 = &UIFontWeightBlack;
        break;
      default:
        goto LABEL_15;
    }
  }
LABEL_21:
  v16[0] = MEMORY[0x20C5EEF00](objc_msgSend(off_240036728, "numberWithDouble:", *v3));
  v15[1] = UIFontWidthTrait;
  v4 = objc_msgSend((id)MEMORY[0x20C5EEF00](objc_msgSend(a1, "fontAttributes")), "width");
  if (v4 <= 499)
  {
    if (v4 > 299)
    {
      if (v4 != 300)
      {
LABEL_35:
        v5 = &UIFontWeightRegular;
        goto LABEL_41;
      }
      v5 = &UIFontWeightLight;
    }
    else
    {
      if (v4 != 100)
      {
        if (v4 == 200)
        {
          v5 = &UIFontWeightUltraLight;
          goto LABEL_41;
        }
        goto LABEL_35;
      }
      v5 = &UIFontWeightThin;
    }
  }
  else if (v4 <= 699)
  {
    if (v4 != 500)
    {
      if (v4 == 600)
      {
        v5 = &UIFontWeightSemibold;
        goto LABEL_41;
      }
      goto LABEL_35;
    }
    v5 = &UIFontWeightMedium;
  }
  else
  {
    switch(v4)
    {
      case 700LL:
        v5 = &UIFontWeightBold;
        break;
      case 800LL:
        v5 = &UIFontWeightHeavy;
        break;
      case 900LL:
        v5 = &UIFontWeightBlack;
        break;
      default:
        goto LABEL_35;
    }
  }
LABEL_41:
  v16[1] = MEMORY[0x20C5EEF00](objc_msgSend(off_240036728, "numberWithDouble:", *v5));
  v15[2] = UIFontSlantTrait;
  v6 = objc_msgSend((id)MEMORY[0x20C5EEF00](objc_msgSend(a1, "fontAttributes")), "style");
  v7 = 0.06944444;
  if ((unint64_t)(v6 - 1) >= 2)
    v7 = 0.0;
  v16[2] = MEMORY[0x20C5EEF00](objc_msgSend(off_240036728, "numberWithDouble:", v7));
  v18[1] = MEMORY[0x20C5EEF00](objc_msgSend(off_240036670, "dictionaryWithObjects:forKeys:count:", v16, v15, 3LL));
  MEMORY[0x20C5EEF00](objc_msgSend(off_240036670, "dictionaryWithObjects:forKeys:count:", v18, v17, 2LL));
  v8 = MEMORY[0x20C5EF0C0]();
  v9 = MEMORY[0x20C5EF0B0](v8);
  v10 = MEMORY[0x20C5EF060](v9);
  v11 = MEMORY[0x20C5EF090](v10);
  v12 = MEMORY[0x20C5EF080](v11);
  v13 = MEMORY[0x20C5EF070](v12);
  v14 = MEMORY[0x20C5EF050](v13);
  MEMORY[0x20C5EF040](v14);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  JUMPOUT(0x20C5EEEE0LL);
}

- (uint64_t)fontName
{
  return *(_QWORD *)(a1 + 8);
}

- (uint64_t)fontAttributes
{
  return *(_QWORD *)(a1 + 16);
}

- (void).cxx_destruct
{
  uint64_t vars8;

  objc_storeStrong((id *)(a1 + 16), 0LL);
  if (((vars8 ^ (2 * vars8)) & 0x4000000000000000LL) != 0)
    __break(0xC471u);
  objc_storeStrong((id *)(a1 + 8), 0LL);
}

@end
