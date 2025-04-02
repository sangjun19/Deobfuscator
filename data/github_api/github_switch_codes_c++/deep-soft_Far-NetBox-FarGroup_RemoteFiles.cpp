
#include <vcl.h>
#pragma hdrstop

#include <Common.h>
#include <nbutils.h>
#include <Sysutils.hpp>
#include <StrUtils.hpp>

#include "RemoteFiles.h"
#include "Terminal.h"
#include "TextsCore.h"
#include "HelpCore.h"
/* TODO 1 : Path class instead of UnicodeString (handle relativity...) */

#if 0
// moved to base/Common.cpp

bool IsUnixStyleWindowsPath(UnicodeString APath)
{
  return (APath.Length() >= 3) && IsLetter(APath[1]) && (APath[2] == L':') && (APath[3] == L'/');
}

bool UnixIsAbsolutePath(UnicodeString APath)
{
  return
    ((APath.Length() >= 1) && (APath[1] == L'/')) ||
    // we need this for FTP only, but this is unfortunately used in a static context
    core::IsUnixStyleWindowsPath(APath);
}

UnicodeString UnixIncludeTrailingBackslash(UnicodeString APath)
{
  // it used to return "/" when input path was empty
  if (!APath.IsEmpty() && !APath.IsDelimiter(SLASH, APath.Length()))
  {
    return APath + SLASH;
  }
  else
  {
    return APath;
  }
}

// Keeps "/" for root path
UnicodeString UnixExcludeTrailingBackslash(UnicodeString APath, bool Simple)
{
  if (APath.IsEmpty() ||
    (APath == ROOTDIRECTORY) ||
    !APath.IsDelimiter(SLASH, APath.Length()) ||
    (!Simple && ((APath.Length() == 3) && core::IsUnixStyleWindowsPath(APath))))
  {
    return APath;
  }
  else
  {
    return APath.SubString(1, APath.Length() - 1);
  }
}

UnicodeString SimpleUnixExcludeTrailingBackslash(UnicodeString APath)
{
  return base::UnixExcludeTrailingBackslash(APath, true);
}

UnicodeString UnixCombinePaths(UnicodeString APath1, UnicodeString APath2)
{
  return UnixIncludeTrailingBackslash(APath1) + APath2;
}

Boolean UnixSamePath(UnicodeString APath1, UnicodeString APath2)
{
  return (base::UnixIncludeTrailingBackslash(APath1) == base::UnixIncludeTrailingBackslash(APath2));
}

bool UnixIsChildPath(UnicodeString AParent, UnicodeString AChild)
{
  UnicodeString Parent = base::UnixIncludeTrailingBackslash(AParent);
  UnicodeString Child = base::UnixIncludeTrailingBackslash(AChild);
  return (Child.SubString(1, Parent.Length()) == Parent);
}

UnicodeString UnixExtractFileDir(UnicodeString APath)
{
  intptr_t Pos = APath.LastDelimiter(L'/');
  // it used to return Path when no slash was found
  if (Pos > 1)
  {
    return APath.SubString(1, Pos - 1);
  }
  else
  {
    return (Pos == 1) ? UnicodeString(L"/") : UnicodeString();
  }
}

// must return trailing backslash
UnicodeString UnixExtractFilePath(UnicodeString APath)
{
  intptr_t Pos = APath.LastDelimiter(L'/');
  // it used to return Path when no slash was found
  if (Pos > 0)
  {
    return APath.SubString(1, Pos);
  }
  else
  {
    return UnicodeString();
  }
}

UnicodeString UnixExtractFileName(UnicodeString APath)
{
  intptr_t Pos = APath.LastDelimiter(L'/');
  UnicodeString Result;
  if (Pos > 0)
  {
    Result = APath.SubString(Pos + 1, APath.Length() - Pos);
  }
  else
  {
    Result = APath;
  }
  return Result;
}

UnicodeString UnixExtractFileExt(UnicodeString APath)
{
  UnicodeString FileName = UnixExtractFileName(APath);
  intptr_t Pos = FileName.LastDelimiter(L".");
  if (Pos > 0)
    return APath.SubString(Pos, APath.Length() - Pos + 1);
  else
    return UnicodeString();
}

UnicodeString ExtractFileName(UnicodeString APath, bool Unix)
{
  if (Unix)
  {
    return UnixExtractFileName(APath);
  }
  else
  {
    return base::ExtractFileName(APath, Unix);
  }
}

bool ExtractCommonPath(const TStrings *AFiles, UnicodeString &APath)
{
  DebugAssert(AFiles->GetCount() > 0);

  APath = ::ExtractFilePath(AFiles->GetString(0));
  bool Result = !APath.IsEmpty();
  if (Result)
  {
    for (intptr_t Index = 1; Index < AFiles->GetCount(); ++Index)
    {
      while (!APath.IsEmpty() &&
        (AFiles->GetString(Index).SubString(1, APath.Length()) != APath))
      {
        intptr_t PrevLen = APath.Length();
        APath = ::ExtractFilePath(::ExcludeTrailingBackslash(APath));
        if (APath.Length() == PrevLen)
        {
          APath.Clear();
          Result = false;
        }
      }
    }
  }

  return Result;
}

bool UnixExtractCommonPath(const TStrings *const AFiles, UnicodeString &APath)
{
  DebugAssert(AFiles->GetCount() > 0);

  APath = base::UnixExtractFilePath(AFiles->GetString(0));
  bool Result = !APath.IsEmpty();
  if (Result)
  {
    for (intptr_t Index = 1; Index < AFiles->GetCount(); ++Index)
    {
      while (!APath.IsEmpty() &&
        (AFiles->GetString(Index).SubString(1, APath.Length()) != APath))
      {
        intptr_t PrevLen = APath.Length();
        APath = base::UnixExtractFilePath(base::UnixExcludeTrailingBackslash(APath));
        if (APath.Length() == PrevLen)
        {
          APath.Clear();
          Result = false;
        }
      }
    }
  }

  return Result;
}

bool IsUnixRootPath(UnicodeString APath)
{
  return APath.IsEmpty() || (APath == ROOTDIRECTORY);
}

bool IsUnixHiddenFile(UnicodeString APath)
{
  return (APath != THISDIRECTORY) && (APath != PARENTDIRECTORY) &&
    !APath.IsEmpty() && (APath[1] == L'.');
}

UnicodeString AbsolutePath(UnicodeString Base, UnicodeString APath)
{
  // There's a duplicate implementation in TTerminal::ExpandFileName()
  UnicodeString Result;
  if (APath.IsEmpty())
  {
    Result = Base;
  }
  else if (APath[1] == L'/')
  {
    Result = base::UnixExcludeTrailingBackslash(APath);
  }
  else
  {
    Result = base::UnixIncludeTrailingBackslash(
        base::UnixIncludeTrailingBackslash(Base) + APath);
    intptr_t P;
    while ((P = Result.Pos(L"/../")) > 0)
    {
      // special case, "/../" => "/"
      if (P == 1)
      {
        Result = ROOTDIRECTORY;
      }
      else
      {
        intptr_t P2 = Result.SubString(1, P - 1).LastDelimiter(L"/");
        DebugAssert(P2 > 0);
        Result.Delete(P2, P - P2 + 3);
      }
    }
    while ((P = Result.Pos(L"/./")) > 0)
    {
      Result.Delete(P, 2);
    }
    Result = base::UnixExcludeTrailingBackslash(Result);
  }
  return Result;
}

UnicodeString FromUnixPath(UnicodeString APath)
{
  return ReplaceStr(APath, SLASH, BACKSLASH);
}

UnicodeString ToUnixPath(UnicodeString APath)
{
  return ReplaceStr(APath, BACKSLASH, SLASH);
}

static void CutFirstDirectory(UnicodeString &S, bool Unix)
{
  UnicodeString Sep = Unix ? SLASH : BACKSLASH;
  if (S == Sep)
  {
    S.Clear();
  }
  else
  {
    bool Root = false;
    intptr_t P = 0;
    if (S[1] == Sep[1])
    {
      Root = true;
      S.Delete(1, 1);
    }
    else
    {
      Root = false;
    }
    if (S[1] == L'.')
    {
      S.Delete(1, 4);
    }
    P = S.Pos(Sep[1]);
    if (P)
    {
      S.Delete(1, P);
      S = L"..." + Sep + S;
    }
    else
    {
      S.Clear();
    }
    if (Root)
    {
      S = Sep + S;
    }
  }
}

UnicodeString MinimizeName(UnicodeString AFileName, intptr_t MaxLen, bool Unix)
{
  UnicodeString Drive, Dir, Name;
  UnicodeString Sep = Unix ? SLASH : BACKSLASH;

  UnicodeString Result = AFileName;
  if (Unix)
  {
    intptr_t P = Result.LastDelimiter(SLASH);
    if (P)
    {
      Dir = Result.SubString(1, P);
      Name = Result.SubString(P + 1, Result.Length() - P);
    }
    else
    {
      Dir.Clear();
      Name = Result;
    }
  }
  else
  {
    Dir = ::ExtractFilePath(Result);
    Name = base::ExtractFileName(Result, false);

    if (Dir.Length() >= 2 && Dir[2] == L':')
    {
      Drive = Dir.SubString(1, 2);
      Dir.Delete(1, 2);
    }
  }

  while ((!Dir.IsEmpty() || !Drive.IsEmpty()) && (Result.Length() > MaxLen))
  {
    if (Dir == Sep + L"..." + Sep)
    {
      Dir = L"..." + Sep;
    }
    else if (Dir.IsEmpty())
    {
      Drive.Clear();
    }
    else
    {
      CutFirstDirectory(Dir, Unix);
    }
    Result = Drive + Dir + Name;
  }

  if (Result.Length() > MaxLen)
  {
    Result = Result.SubString(1, MaxLen);
  }
  return Result;
}

UnicodeString MakeFileList(const TStrings *AFileList)
{
  UnicodeString Result;
  for (intptr_t Index = 0; Index < AFileList->GetCount(); ++Index)
  {
    UnicodeString FileName = AFileList->GetString(Index);
    // currently this is used for local file only, so no delimiting is done
    AddToList(Result,  AddQuotes(FileName), L" ");
  }
  return Result;
}

// copy from BaseUtils.pas
TDateTime ReduceDateTimePrecision(const TDateTime &ADateTime,
  TModificationFmt Precision)
{
  TDateTime DateTime = ADateTime;
  if (Precision == mfNone)
  {
    DateTime = double(0.0);
  }
  else if (Precision != mfFull)
  {
    uint16_t Y, M, D, H, N, S, MS;

    ::DecodeDate(DateTime, Y, M, D);
    ::DecodeTime(DateTime, H, N, S, MS);
    switch (Precision)
    {
    case mfMDHM:
      S = 0;
      MS = 0;
      break;

    case mfMDY:
      H = 0;
      N = 0;
      S = 0;
      MS = 0;
      break;

    default:
      DebugFail();
    }

    DateTime = EncodeDateVerbose(Y, M, D) + EncodeTimeVerbose(H, N, S, MS);
  }
  return DateTime;
}

TModificationFmt LessDateTimePrecision(
  TModificationFmt Precision1, TModificationFmt Precision2)
{
  return (Precision1 < Precision2) ? Precision1 : Precision2;
}

UnicodeString UserModificationStr(const TDateTime &DateTime,
  TModificationFmt Precision)
{
  Word Year, Month, Day, Hour, Min, Sec, MSec;
  DateTime.DecodeDate(Year, Month, Day);
  DateTime.DecodeTime(Hour, Min, Sec, MSec);
  switch (Precision)
  {
  case mfNone:
    return L"";
  case mfMDY:
    return FORMAT("%3s %2d %2d", EngShortMonthNames[Month - 1], Day, Year);
  case mfMDHM:
    return FORMAT("%3s %2d %2d:%2.2d",
        EngShortMonthNames[Month - 1], Day, Hour, Min);
  case mfFull:
    return FORMAT("%3s %2d %2d:%2.2d:%2.2d %4d",
        EngShortMonthNames[Month - 1], Day, Hour, Min, Sec, Year);
  default:
    DebugAssert(false);
  }
  return UnicodeString();
}

UnicodeString ModificationStr(const TDateTime &DateTime,
  TModificationFmt Precision)
{
  uint16_t Year, Month, Day, Hour, Min, Sec, MSec;
  DateTime.DecodeDate(Year, Month, Day);
  DateTime.DecodeTime(Hour, Min, Sec, MSec);
  switch (Precision)
  {
  case mfNone:
    return L"";

  case mfMDY:
    return FORMAT("%3s %2d %2d", EngShortMonthNames[Month - 1], Day, Year);

  case mfMDHM:
    return FORMAT("%3s %2d %2d:%2.2d",
        EngShortMonthNames[Month - 1], Day, Hour, Min);

  default:
    DebugFail();
  // fall thru

  case mfFull:
    return FORMAT("%3s %2d %2d:%2.2d:%2.2d %4d",
        EngShortMonthNames[Month - 1], Day, Hour, Min, Sec, Year);
  }
}

int FakeFileImageIndex(UnicodeString /*AFileName*/, uint32_t /*Attrs*/,
  UnicodeString * /*TypeName*/)
{
  /*Attrs |= FILE_ATTRIBUTE_NORMAL;

  TSHFileInfoW SHFileInfo = {0};
  // On Win2k we get icon of "ZIP drive" for ".." (parent directory)
  if ((FileName == L"..") ||
      ((FileName.Length() == 2) && (FileName[2] == L':') && IsLetter(FileName[1])) ||
      IsReservedName(FileName))
  {
    FileName = L"dumb";
  }
  // this should be somewhere else, probably in TUnixDirView,
  // as the "partial" overlay is added there too
  if (::SameText(base::UnixExtractFileExt(FileName), PARTIAL_EXT))
  {
    static const size_t PartialExtLen = _countof(PARTIAL_EXT) - 1;
    FileName.SetLength(FileName.Length() - PartialExtLen);
  }

  int Icon;
  if (SHGetFileInfo(FileName.c_str(),
        Attrs, &SHFileInfo, sizeof(SHFileInfo),
        SHGFI_SYSICONINDEX | SHGFI_USEFILEATTRIBUTES | SHGFI_TYPENAME) != 0)
  {

    if (TypeName != nullptr)
    {
      *TypeName = SHFileInfo.szTypeName;
    }
    Icon = SHFileInfo.iIcon;
  }
  else
  {
    if (TypeName != nullptr)
    {
      *TypeName = L"";
    }
    Icon = -1;
  }

  return Icon;*/
  return -1;
}

bool SameUserName(UnicodeString UserName1, UnicodeString UserName2)
{
  // Bitvise reports file owner as "user@host", but we login with "user" only.
  UnicodeString AUserName1 = CopyToChar(UserName1, L'@', true);
  UnicodeString AUserName2 = CopyToChar(UserName2, L'@', true);
  return ::SameText(AUserName1, AUserName2);
}

UnicodeString FormatMultiFilesToOneConfirmation(UnicodeString ATarget, bool Unix)
{
  UnicodeString Dir;
  UnicodeString Name;
  UnicodeString Path;
  if (Unix)
  {
    Dir = UnixExtractFileDir(ATarget);
    Name = UnixExtractFileName(ATarget);
    Path = UnixIncludeTrailingBackslash(ATarget);
  }
  else
  {
    Dir = ::ExtractFilePath(ATarget);
    Name = ExtractFileName(ATarget, Unix);
    Path = ::IncludeTrailingBackslash(ATarget);
  }
  return FMTLOAD(MULTI_FILES_TO_ONE, Name, Dir, Path);
}

#endif // #if 0

TRemoteToken::TRemoteToken() :
  FID(0),
  FIDValid(false)
{
}

TRemoteToken::TRemoteToken(UnicodeString Name) :
  FName(Name),
  FID(0),
  FIDValid(false)
{
}

TRemoteToken::TRemoteToken(const TRemoteToken &rhs) :
  FName(rhs.FName),
  FID(rhs.FID),
  FIDValid(rhs.FIDValid)
{
}

void TRemoteToken::Clear()
{
  FID = 0;
  FIDValid = false;
}

bool TRemoteToken::operator==(const TRemoteToken &rhs) const
{
  return
    (FName == rhs.FName) &&
    (FIDValid == rhs.FIDValid) &&
    (!FIDValid || (FID == rhs.FID));
}

bool TRemoteToken::operator!=(const TRemoteToken &rhs) const
{
  return !(*this == rhs);
}

TRemoteToken &TRemoteToken::operator=(const TRemoteToken &rhs)
{
  if (this != &rhs)
  {
    FName = rhs.FName;
    FIDValid = rhs.FIDValid;
    FID = rhs.FID;
  }
  return *this;
}

intptr_t TRemoteToken::Compare(const TRemoteToken &rhs) const
{
  intptr_t Result;
  if (!FName.IsEmpty())
  {
    if (!rhs.FName.IsEmpty())
    {
      Result = ::AnsiCompareText(FName, rhs.FName);
    }
    else
    {
      Result = -1;
    }
  }
  else
  {
    if (!rhs.FName.IsEmpty())
    {
      Result = 1;
    }
    else
    {
      if (FIDValid)
      {
        if (rhs.FIDValid)
        {
          Result = (FID < rhs.FID) ? -1 : ((FID > rhs.FID) ? 1 : 0);
        }
        else
        {
          Result = -1;
        }
      }
      else
      {
        if (rhs.FIDValid)
        {
          Result = 1;
        }
        else
        {
          Result = 0;
        }
      }
    }
  }
  return Result;
}

void TRemoteToken::SetID(intptr_t Value)
{
  FID = Value;
  FIDValid = Value != 0;
}

bool TRemoteToken::GetNameValid() const
{
  return !FName.IsEmpty();
}

bool TRemoteToken::GetIsSet() const
{
  return !FName.IsEmpty() || FIDValid;
}

UnicodeString TRemoteToken::GetDisplayText() const
{
  if (!FName.IsEmpty())
  {
    return FName;
  }
  if (FIDValid)
  {
    return IntToStr(FID);
  }
  return UnicodeString();
}

UnicodeString TRemoteToken::GetLogText() const
{
  return FORMAT("\"%s\" [%d]", FName, ToInt(FID));
}


TRemoteTokenList *TRemoteTokenList::Duplicate() const
{
  std::unique_ptr<TRemoteTokenList> Result(new TRemoteTokenList());
  try__catch
  {
    TTokens::const_iterator it = FTokens.begin();
    while (it != FTokens.end())
    {
      Result->Add(*it);
      ++it;
    }
  }
#if 0
  catch (...)
  {
    delete Result;
    throw;
  }
#endif // #if 0
  return Result.release();
}

void TRemoteTokenList::Clear()
{
  FTokens.clear();
  FNameMap.clear();
  FIDMap.clear();
}

void TRemoteTokenList::Add(const TRemoteToken &Token)
{
  FTokens.push_back(Token);
  if (Token.GetIDValid())
  {
    // std::pair<TIDMap::iterator, bool> Position =
    FIDMap.insert(TIDMap::value_type(Token.GetID(), FTokens.size() - 1));
  }
  if (Token.GetNameValid())
  {
    // std::pair<TNameMap::iterator, bool> Position =
    FNameMap.insert(TNameMap::value_type(Token.GetName(), FTokens.size() - 1));
  }
}

void TRemoteTokenList::AddUnique(const TRemoteToken &Token)
{
  if (Token.GetIDValid())
  {
    TIDMap::const_iterator it = FIDMap.find(Token.GetID());
    if (it != FIDMap.end())
    {
      // is present already.
      // may have different name (should not),
      // but what can we do about it anyway?
    }
    else
    {
      Add(Token);
    }
  }
  else if (Token.GetNameValid())
  {
    TNameMap::const_iterator it = FNameMap.find(Token.GetName());
    if (it != FNameMap.end())
    {
      // is present already.
    }
    else
    {
      Add(Token);
    }
  }
  else
  {
    // can happen, e.g. with winsshd/SFTP
  }
}

bool TRemoteTokenList::Exists(UnicodeString Name) const
{
  // We should make use of SameUserName
  return (FNameMap.find(Name) != FNameMap.end());
}

const TRemoteToken *TRemoteTokenList::Find(uintptr_t ID) const
{
  TIDMap::const_iterator it = FIDMap.find(ID);
  const TRemoteToken *Result = nullptr;
  if (it != FIDMap.end())
  {
    Result = &FTokens[(*it).second];
  }
  return Result;
}

const TRemoteToken *TRemoteTokenList::Find(UnicodeString Name) const
{
  TNameMap::const_iterator it = FNameMap.find(Name);
  const TRemoteToken *Result = nullptr;
  if (it != FNameMap.end())
  {
    Result = &FTokens[(*it).second];
  }
  return Result;
}

void TRemoteTokenList::Log(TTerminal *Terminal, const wchar_t *Title)
{
  if (!FTokens.empty())
  {
    Terminal->LogEvent(FORMAT("Following %s found:", Title));
    for (intptr_t Index = 0; Index < static_cast<intptr_t>(FTokens.size()); ++Index)
    {
      Terminal->LogEvent(UnicodeString(L"  ") + FTokens[Index].GetLogText());
    }
  }
  else
  {
    Terminal->LogEvent(FORMAT("No %s found.", Title));
  }
}

intptr_t TRemoteTokenList::GetCount() const
{
  return static_cast<intptr_t>(FTokens.size());
}

const TRemoteToken *TRemoteTokenList::Token(intptr_t Index) const
{
  return &FTokens[Index];
}


TRemoteFile::TRemoteFile(TObjectClassId Kind, TRemoteFile *ALinkedByFile) :
  TPersistent(Kind),
  FDirectory(nullptr),
  FModificationFmt(mfFull),
  FLinkedFile(nullptr),
  FLinkedByFile(ALinkedByFile),
  FRights(nullptr),
  FTerminal(nullptr),
  FSize(0),
  FINodeBlocks(0),
  FIconIndex(-1),
  FIsHidden(-1),
  FType(0),
  FIsSymLink(false),
  FCyclicLink(false)
{
  Init();
  FLinkedByFile = ALinkedByFile;
}

TRemoteFile::TRemoteFile(TRemoteFile *ALinkedByFile) :
  TPersistent(OBJECT_CLASS_TRemoteFile)
{
  Init();
  FLinkedByFile = ALinkedByFile;
}

TRemoteFile::~TRemoteFile()
{
  SAFE_DESTROY(FRights);
  SAFE_DESTROY(FLinkedFile);
}

TRemoteFile *TRemoteFile::Duplicate(bool Standalone) const
{
  std::unique_ptr<TRemoteFile> Result(new TRemoteFile());
  try__catch
  {
    if (FLinkedFile)
    {
      Result->FLinkedFile = FLinkedFile->Duplicate(true);
      Result->FLinkedFile->FLinkedByFile = Result.get();
    }
    Result->SetRights(FRights);
#define COPY_FP(PROP) Result->F ## PROP = F ## PROP;
    COPY_FP(Terminal);
    COPY_FP(Owner);
    COPY_FP(ModificationFmt);
    COPY_FP(Size);
    COPY_FP(FileName);
    COPY_FP(DisplayName);
    COPY_FP(INodeBlocks);
    COPY_FP(Modification);
    COPY_FP(LastAccess);
    COPY_FP(Group);
    COPY_FP(IconIndex);
    COPY_FP(TypeName);
    COPY_FP(IsSymLink);
    COPY_FP(LinkTo);
    COPY_FP(Type);
    COPY_FP(CyclicLink);
    COPY_FP(HumanRights);
#undef COPY_FP
    if (Standalone && (!FFullFileName.IsEmpty() || (GetDirectory() != nullptr)))
    {
      Result->FFullFileName = GetFullFileName();
    }
  }
#if 0
  catch (...)
  {
    delete Result;
    throw;
  }
#endif // #if 0
  return Result.release();
}

void TRemoteFile::LoadTypeInfo() const
{
  /* TODO : If file is link: Should be attributes taken from linked file? */
#if 0
  uint32_t Attrs = INVALID_FILE_ATTRIBUTES;
  if (GetIsDirectory())
  {
    Attrs |= FILE_ATTRIBUTE_DIRECTORY;
  }
  if (GetIsHidden())
  {
    Attrs |= FILE_ATTRIBUTE_HIDDEN;
  }

  UnicodeString DumbFileName = (GetIsSymLink() && !GetLinkTo().IsEmpty() ? GetLinkTo() : GetFileName());

  FIconIndex = FakeFileImageIndex(DumbFileName, Attrs, &FTypeName);
#endif // #if 0
}

void TRemoteFile::Init()
{
  FDirectory = nullptr;
  FModificationFmt = mfFull;
  FLinkedFile = nullptr;
  FLinkedByFile = nullptr;
  FRights = new TRights();
  FTerminal = nullptr;
  FSize = 0;
  FINodeBlocks = 0;
  FIconIndex = -1;
  FIsHidden = -1;
  FType = 0;
  FIsSymLink = false;
  FCyclicLink = false;
}

int64_t TRemoteFile::GetSize() const
{
  return GetIsDirectory() ? 0 : FSize;
}

intptr_t TRemoteFile::GetIconIndex() const
{
  if (FIconIndex == -1)
  {
    const_cast<TRemoteFile *>(this)->LoadTypeInfo();
  }
  return FIconIndex;
}

UnicodeString TRemoteFile::GetTypeName() const
{
  // check availability of type info by icon index, because type name can be empty
  if (FIconIndex < 0)
  {
    LoadTypeInfo();
  }
  return FTypeName;
}

Boolean TRemoteFile::GetIsHidden() const
{
  bool Result;
  switch (FIsHidden)
  {
  case 0:
    Result = false;
    break;

  case 1:
    Result = true;
    break;

  default:
    Result = base::IsUnixHiddenFile(GetFileName());
    break;
  }

  return Result;
}

void TRemoteFile::SetIsHidden(bool Value)
{
  FIsHidden = Value ? 1 : 0;
}

Boolean TRemoteFile::GetIsDirectory() const
{
  return (::UpCase(GetType()) == FILETYPE_DIRECTORY);
}

Boolean TRemoteFile::GetIsParentDirectory() const
{
  return wcscmp(FFileName.c_str(), PARENTDIRECTORY) == 0;
}

Boolean TRemoteFile::GetIsThisDirectory() const
{
  return wcscmp(FFileName.c_str(), THISDIRECTORY) == 0;
}

Boolean TRemoteFile::GetIsInaccesibleDirectory() const
{
  Boolean Result;
  if (GetIsDirectory())
  {
    DebugAssert(GetTerminal());
    Result = !
      (base::SameUserName(GetTerminal()->TerminalGetUserName(), L"root")) ||
      (((GetRights()->GetRightUndef(TRights::rrOtherExec) != TRights::rsNo)) ||
        ((GetRights()->GetRight(TRights::rrGroupExec) != TRights::rsNo) &&
          GetTerminal()->GetMembership()->Exists(GetFileGroup().GetName())) ||
        ((GetRights()->GetRight(TRights::rrUserExec) != TRights::rsNo) &&
          (base::SameUserName(GetTerminal()->TerminalGetUserName(), GetFileOwner().GetName()))));
  }
  else
  {
    Result = False;
  }
  return Result;
}

wchar_t TRemoteFile::GetType() const
{
  if (GetIsSymLink() && FLinkedFile)
  {
    return FLinkedFile->GetType();
  }
  return FType;
}

void TRemoteFile::SetType(wchar_t AType)
{
  FType = AType;
  FIsSymLink = (UpCase(FType) == FILETYPE_SYMLINK);
}

TRemoteFile *TRemoteFile::GetLinkedFile() const
{
  // do not call FindLinkedFile as it would be called repeatedly for broken symlinks
  return FLinkedFile;
}

void TRemoteFile::SetLinkedFile(TRemoteFile *Value)
{
  if (FLinkedFile != Value)
  {
    if (FLinkedFile)
    {
      SAFE_DESTROY(FLinkedFile);
    }
    FLinkedFile = Value;
  }
}

bool TRemoteFile::GetBrokenLink() const
{
  DebugAssert(GetTerminal());
  // If file is symlink but we couldn't find linked file we assume broken link
  return (GetIsSymLink() && (FCyclicLink || !FLinkedFile) &&
      GetTerminal()->GetResolvingSymlinks());
  // "!FLinkTo.IsEmpty()" removed because it does not work with SFTP
}

bool TRemoteFile::GetIsTimeShiftingApplicable() const
{
  return GetIsTimeShiftingApplicable(GetModificationFmt());
}

bool TRemoteFile::GetIsTimeShiftingApplicable(TModificationFmt ModificationFmt)
{
  return (ModificationFmt == mfMDHM) || (ModificationFmt == mfFull);
}

void TRemoteFile::ShiftTimeInSeconds(int64_t Seconds)
{
  ShiftTimeInSeconds(FModification, GetModificationFmt(), Seconds);
  ShiftTimeInSeconds(FLastAccess, GetModificationFmt(), Seconds);
}

void TRemoteFile::ShiftTimeInSeconds(TDateTime &DateTime, TModificationFmt ModificationFmt, int64_t Seconds)
{
  if ((Seconds != 0) && GetIsTimeShiftingApplicable(ModificationFmt))
  {
    DebugAssert(int(DateTime) != 0);
    DateTime = IncSecond(DateTime, Seconds);
  }
}

void TRemoteFile::SetModification(const TDateTime &Value)
{
  if (FModification != Value)
  {
    FModificationFmt = mfFull;
    FModification = Value;
  }
}

UnicodeString TRemoteFile::GetUserModificationStr() const
{
  return base::UserModificationStr(GetModification(), FModificationFmt);
}

UnicodeString TRemoteFile::GetModificationStr() const
{
  return base::ModificationStr(GetModification(), FModificationFmt);
}

UnicodeString TRemoteFile::GetExtension() const
{
  return base::UnixExtractFileExt(FFileName);
}

void TRemoteFile::SetRights(TRights *Value)
{
  FRights->Assign(Value);
}

UnicodeString TRemoteFile::GetRightsStr() const
{
  // note that HumanRights is typically an empty string
  // (with an exception of Perm-fact-only MLSD FTP listing)
  return FRights->GetUnknown() ? GetHumanRights() : FRights->GetText();
}

void TRemoteFile::SetListingStr(UnicodeString Value)
{
  // Value stored in 'Value' can be used for error message
  UnicodeString Line = Value;
  FIconIndex = -1;
  try
  {
    UnicodeString Col;

    // Do we need to do this (is ever TAB is LS output)?
    Line = ReplaceChar(Line, L'\t', L' ');

    SetType(Line[1]);
    Line.Delete(1, 1);

    auto GetNCol = [&]()
    {
      if (Line.IsEmpty())
        throw Exception(L"");
      intptr_t P = Line.Pos(L' ');
      if (P)
      {
        Col = Line;
        Col.SetLength(P - 1);
        Line.Delete(1, P);
      }
      else
      {
        Col = Line;
        Line.Clear();
      }
    };
    auto GetCol = [&]()
    {
      GetNCol();
      Line = ::TrimLeft(Line);
    };

    // Rights string may contain special permission attributes (S,t, ...)
    TODO("maybe no longer necessary, once we can handle the special permissions");
    GetRights()->SetAllowUndef(True);
    // On some system there is no space between permissions and node blocks count columns
    // so we get only first 9 characters and trim all following spaces (if any)
    GetRights()->SetText(Line.SubString(1, 9));
    Line.Delete(1, 9);
    // Rights column maybe followed by '+', '@' or '.' signs, we ignore them
    // (On MacOS, there may be a space in between)
    if (!Line.IsEmpty() && ((Line[1] == L'+') || (Line[1] == L'@') || (Line[1] == L'.')))
    {
      Line.Delete(1, 1);
    }
    else if ((Line.Length() >= 2) && (Line[1] == L' ') &&
      ((Line[2] == L'+') || (Line[2] == L'@') || (Line[2] == L'.')))
    {
      Line.Delete(1, 2);
    }
    Line = Line.TrimLeft();

    GetCol();
    if (!::TryStrToInt64(Col, FINodeBlocks))
    {
      // if the column is not an integer, suppose it's owner
      // (Android BusyBox)
      FINodeBlocks = 0;
    }
    else
    {
      GetCol();
    }

    FOwner.SetName(Col);

    // #60 17.10.01: group name can contain space
    FGroup.SetName(L"");
    GetCol();
    int64_t ASize;
    do
    {
      FGroup.SetName(FGroup.GetName() + Col);
      GetCol();
      // SSH FS link like
      // d????????? ? ? ? ? ? name
      if ((FGroup.GetName() == L"?") && (Col == L"?"))
      {
        ASize = 0;
      }
      else
      {
        DebugAssert(!Col.IsEmpty());
        // for devices etc.. there is additional column ending by comma, we ignore it
        if (Col[Col.Length()] == L',')
          GetCol();
        ASize = ::StrToInt64Def(Col, -1);
        // if it's not a number (file size) we take it as part of group name
        // (at least on CygWin, there can be group with space in its name)
        if (ASize < 0)
          Col = L" " + Col;
      }
    }
    while (ASize < 0);

    // do not read modification time and filename if it is already set
    if (::IsZero(FModification.GetValue()) && GetFileName().IsEmpty())
    {
      FSize = ASize;

      Word Year = 0, Month = 0, Day = 0, Hour = 0, Min = 0, Sec = 0;
      Word CurrYear = 0, CurrMonth = 0, CurrDay = 0;
      ::DecodeDate(::Date(), CurrYear, CurrMonth, CurrDay);

      GetCol();
      // SSH FS link, see above
      if (Col == L"?")
      {
        GetCol();
        FModificationFmt = mfNone;
        FModification = 0;
        FLastAccess = 0;
      }
      else
      {
        bool DayMonthFormat = false;
        // format yyyy-mm-dd hh:mm:ss.ms ? // example: 2017-07-27 10:44:52.404136754 +0300 .
        int Y, M, D;
        int Filled =
          swscanf(Col.c_str(), L"%04d-%02d-%02d", &Y, &M, &D);
        if (Filled == 3)
        {
          Year = ToWord(Y);
          Month = ToWord(M);
          Day = ToWord(D);
          GetCol();
          int H, Mn, S, MS;
          Filled = swscanf(Col.c_str(), L"%02d:%02d:%02d.%d", &H, &Mn, &S, &MS);
          if (Filled == 4)
          {
            Hour = ToWord(H);
            Min = ToWord(Mn);
            Sec = ToWord(S);
            FModificationFmt = mfFull;
            // skip TZ (TODO)
            // do not trim leading space of filename
            GetNCol();
          }
          else
          {
            // format yyyy-mm-dd hh:mm:ss ? // example: 2017-07-27 10:44:52 +0300 TZ
            Filled = swscanf(Col.c_str(), L"%02d:%02d:%02d", &H, &Mn, &S);
            if (Filled == 3)
            {
              Hour = ToWord(H);
              Min = ToWord(Mn);
              Sec = ToWord(S);
              FModificationFmt = mfFull;
              // skip TZ (TODO)
              // do not trim leading space of filename
              GetNCol();
            }
          }
        } else {
        // format dd mmm or mmm dd ?
        Day = ::ToWord(::StrToIntDef(Col, 0));
        if (Day > 0)
        {
          DayMonthFormat = true;
          GetCol();
        }
        Month = 0;
        auto Col2Month = [&]()
        {
          for (Word IMonth = 0; IMonth < 12; IMonth++)
            if (!Col.CompareIC(EngShortMonthNames[IMonth]))
            {
              Month = IMonth;
              Month++;
              break;
            }
        };

        Col2Month();
        // if the column is not known month name, it may have been "yyyy-mm-dd"
        // for --full-time format
        if ((Month == 0) && (Col.Length() == 10) && (Col[5] == L'-') && (Col[8] == L'-'))
        {
          Year = ToWord(Col.SubString(1, 4).ToIntPtr());
          Month = ToWord(Col.SubString(6, 2).ToIntPtr());
          Day = ToWord(Col.SubString(9, 2).ToIntPtr());
          GetCol();
          Hour = ToWord(Col.SubString(1, 2).ToIntPtr());
          Min = ToWord(Col.SubString(4, 2).ToIntPtr());
          if (Col.Length() >= 8)
          {
            Sec = ToWord(::StrToInt64(Col.SubString(7, 2)));
          }
          else
          {
            Sec = 0;
          }
          FModificationFmt = mfFull;
          // skip TZ (TODO)
          // do not trim leading space of filename
          GetNCol();
        }
        else
        {
          bool FullTime = false;
          // or it may have been day name for another format of --full-time
          if (Month == 0)
          {
            GetCol();
            Col2Month();
            // neither standard, not --full-time format
            if (Month == 0)
            {
              Abort();
            }
            else
            {
              FullTime = true;
            }
          }

          if (Day == 0)
          {
            GetNCol();
            Day = ToWord(::StrToInt64(Col));
          }
          if ((Day < 1) || (Day > 31))
          {
            Abort();
          }

          // second full-time format
          // ddd mmm dd hh:nn:ss yyyy
          if (FullTime)
          {
            GetCol();
            if (Col.Length() != 8)
            {
              Abort();
            }
            Hour = ToWord(::StrToInt64(Col.SubString(1, 2)));
            Min = ToWord(::StrToInt64(Col.SubString(4, 2)));
            Sec = ToWord(::StrToInt64(Col.SubString(7, 2)));
            FModificationFmt = mfFull;
            // do not trim leading space of filename
            GetNCol();
            Year = ToWord(::StrToInt64(Col));
          }
          else
          {
            // for format dd mmm the below description seems not to be true,
            // the year is not aligned to 5 characters
            if (DayMonthFormat)
            {
              GetCol();
            }
            else
            {
              // Time/Year indicator is always 5 characters long (???), on most
              // systems year is aligned to right (_YYYY), but on some to left (YYYY_),
              // we must ensure that trailing space is also deleted, so real
              // separator space is not treated as part of file name
              Col = Line.SubString(1, 6).Trim();
              Line.Delete(1, 6);
            }
          }
          {
            // GetNCol(); // We don't want to trim input strings (name with space at beginning???)
            // Check if we got time (contains :) or year
            intptr_t P;
            if ((P = ToWord(Col.Pos(L':'))) > 0)
            {
              Hour = ToWord(::StrToInt64(Col.SubString(1, P - 1)));
              Min = ToWord(::StrToInt64(Col.SubString(P + 1, Col.Length() - P)));
              if ((Hour > 23) || (Min > 59))
                Abort();
              // When we don't got year, we assume current year
              // with exception that the date would be in future
              // in this case we assume last year.
              ::DecodeDate(::Date(), Year, CurrMonth, CurrDay);
              if ((Month > CurrMonth) ||
                  (Month == CurrMonth && Day > CurrDay))
              {
                Year--;
              }
              Sec = 0;
              FModificationFmt = mfMDHM;
            }
            else
            {
              Year = ToWord(::StrToInt64(Col));
              if (Year > 10000)
                Abort();
              // When we don't got time we assume midnight
              Hour = 0;
              Min = 0;
              Sec = 0;
              FModificationFmt = mfMDY;
            }
          }
        }}

        if (Year == 0)
          Year = CurrYear;
        if (Month == 0)
          Month = CurrMonth;
        if (Day == 0)
          Day = CurrDay;
        FModification = EncodeDateVerbose(Year, Month, Day) + EncodeTimeVerbose(Hour, Min, Sec, 0);
        // adjust only when time is known,
        // adjusting default "midnight" time makes no sense
        if (((FModificationFmt == mfMDHM) || (FModificationFmt == mfFull)) && GetTerminal())
        {
          FModification = ::AdjustDateTimeFromUnix(FModification,
              GetTerminal()->GetSessionData()->GetDSTMode());
        }

        if (::IsZero(FLastAccess.GetValue()))
        {
          FLastAccess = FModification;
        }
      }

      // separating space is already deleted, other spaces are treated as part of name

      {
        FLinkTo.Clear();
        if (GetIsSymLink())
        {
          intptr_t P = Line.Pos(SYMLINKSTR);
          if (P)
          {
            FLinkTo = Line.SubString(
                P + nb::StrLength(SYMLINKSTR), Line.Length() - P + nb::StrLength(SYMLINKSTR) + 1);
            Line.SetLength(P - 1);
          }
          else
          {
            Abort();
          }
        }
        FFileName = base::UnixExtractFileName(::Trim(Line));
      }
    }
  }
  catch (Exception &E)
  {
    throw ETerminal(&E, FMTLOAD(LIST_LINE_ERROR, Value), HELP_LIST_LINE_ERROR);
  }
}

void TRemoteFile::Complete()
{
  DebugAssert(GetTerminal() != nullptr);
  if (GetIsSymLink() && GetTerminal()->GetResolvingSymlinks())
  {
    FindLinkedFile();
  }
}

void TRemoteFile::FindLinkedFile()
{
  DebugAssert(GetTerminal() && GetIsSymLink());

  if (FLinkedFile)
  {
    SAFE_DESTROY(FLinkedFile);
  }
  FLinkedFile = nullptr;

  FCyclicLink = false;
  if (!GetLinkTo().IsEmpty())
  {
    // check for cyclic link
    TRemoteFile *LinkedBy = FLinkedByFile;
    while (LinkedBy)
    {
      if (LinkedBy->GetLinkTo() == GetLinkTo())
      {
        // this is currently redundant information, because it is used only to
        // detect broken symlink, which would be otherwise detected
        // by FLinkedFile == nullptr
        FCyclicLink = true;
        break;
      }
      LinkedBy = LinkedBy->FLinkedByFile;
    }
  }

  if (FCyclicLink)
  {
    TRemoteFile *LinkedBy = FLinkedByFile;
    while (LinkedBy)
    {
      LinkedBy->FCyclicLink = true;
      LinkedBy = LinkedBy->FLinkedByFile;
    }
  }
  else
  {
    DebugAssert(GetTerminal()->GetResolvingSymlinks());
    GetTerminal()->SetExceptionOnFail(true);
    try
    {
      try__finally
      {
        SCOPE_EXIT
        {
          GetTerminal()->SetExceptionOnFail(false);
        };
        GetTerminal()->ReadSymlink(this, FLinkedFile);
      }
      __finally
      {
#if 0
        GetTerminal()->SetExceptionOnFail(false);
#endif // #if 0
      };
    }
    catch (Exception &E)
    {
      if (isa<EFatal>(&E))
      {
        throw;
      }
      GetTerminal()->GetLog()->AddException(&E);
    }
  }
}

UnicodeString TRemoteFile::GetListingStr() const
{
  // note that ModificationStr is longer than 12 for mfFull
  UnicodeString LinkPart;
  // expanded from ?: to avoid memory leaks
  if (GetIsSymLink())
  {
    LinkPart = UnicodeString(SYMLINKSTR) + GetLinkTo();
  }
  return FORMAT("%s%s %3s %-8s %-8s %9s %-12s %s%s",
      GetType(), GetRights()->GetText(), ::Int64ToStr(FINodeBlocks), GetFileOwner().GetName(), GetFileGroup().GetName(),
      ::Int64ToStr(GetSize()),  // explicitly using size even for directories
      GetModificationStr(), GetFileName(),
      LinkPart);
}

UnicodeString TRemoteFile::GetFullFileName() const
{
  if (FFullFileName.IsEmpty())
  {
    DebugAssert(GetTerminal());
    DebugAssert(GetDirectory() != nullptr);
    UnicodeString Path;
    if (GetIsParentDirectory())
    {
      Path = GetDirectory()->GetParentPath();
    }
    else if (GetIsDirectory())
    {
      Path = base::UnixIncludeTrailingBackslash(GetDirectory()->GetFullDirectory() + GetFileName());
    }
    else
    {
      Path = GetDirectory()->GetFullDirectory() + GetFileName();
    }
    return GetTerminal()->TranslateLockedPath(Path, true);
  }
  return FFullFileName;
}

bool TRemoteFile::GetHaveFullFileName() const
{
  return !FFullFileName.IsEmpty() || (GetDirectory() != nullptr);
}

intptr_t TRemoteFile::GetAttr() const
{
  intptr_t Result = 0;
  if (GetRights()->GetReadOnly())
  {
    Result |= faReadOnly;
  }
  if (GetIsHidden())
  {
    Result |= faHidden;
  }
  return Result;
}

void TRemoteFile::SetTerminal(TTerminal *Value)
{
  FTerminal = Value;
  if (FLinkedFile)
  {
    FLinkedFile->SetTerminal(Value);
  }
}


TRemoteDirectoryFile::TRemoteDirectoryFile() :
  TRemoteFile(OBJECT_CLASS_TRemoteDirectoryFile)
{
  Init();
}

TRemoteDirectoryFile::TRemoteDirectoryFile(TObjectClassId Kind) :
  TRemoteFile(Kind)
{
  Init();
}

void TRemoteDirectoryFile::Init()
{
  SetModification(TDateTime(0.0));
  SetModificationFmt(mfNone);
  SetLastAccess(GetModification());
  SetType(L'D');
  SetSize(0);
}


TRemoteParentDirectory::TRemoteParentDirectory(TTerminal *ATerminal) :
  TRemoteDirectoryFile(OBJECT_CLASS_TRemoteParentDirectory)
{
  SetFileName(PARENTDIRECTORY);
  SetTerminal(ATerminal);
}

//=== TRemoteFileList ------------------------------------------------------
TRemoteFileList::TRemoteFileList() :
  TObjectList(OBJECT_CLASS_TRemoteFileList),
  FTimestamp(Now())
{
  SetOwnsObjects(true);
}

TRemoteFileList::TRemoteFileList(TObjectClassId Kind) :
  TObjectList(Kind),
  FTimestamp(Now())
{
  SetOwnsObjects(true);
}

void TRemoteFileList::AddFile(TRemoteFile *AFile)
{
  if (AFile)
  {
    Add(AFile);
    AFile->SetDirectory(this);
  }
}

void TRemoteFileList::AddFiles(const TRemoteFileList *AFileList)
{
  if (!AFileList)
    return;
  for (intptr_t Index = 0; Index < AFileList->GetCount(); ++Index)
  {
    AddFile(AFileList->GetFile(Index));
  }
}

TStrings *TRemoteFileList::CloneStrings(TStrings *List)
{
  std::unique_ptr<TStringList> Result(new TStringList());
  Result->SetOwnsObjects(true);
  for (intptr_t Index = 0; Index < List->GetCount(); Index++)
  {
    TRemoteFile *File = static_cast<TRemoteFile *>(List->GetObj(Index));
    Result->AddObject(List->GetString(Index), File->Duplicate(true));
  }
  return Result.release();
}

void TRemoteFileList::DuplicateTo(TRemoteFileList *Copy) const
{
  Copy->Reset();
  for (intptr_t Index = 0; Index < GetCount(); ++Index)
  {
    TRemoteFile *File = GetFile(Index);
    Copy->AddFile(File->Duplicate(false));
  }
  Copy->FDirectory = GetDirectory();
  Copy->FTimestamp = FTimestamp;
}

void TRemoteFileList::Reset()
{
  FTimestamp = Now();
  TObjectList::Clear();
}

void TRemoteFileList::SetDirectory(UnicodeString Value)
{
  FDirectory = base::UnixExcludeTrailingBackslash(Value);
}

UnicodeString TRemoteFileList::GetFullDirectory() const
{
  return base::UnixIncludeTrailingBackslash(GetDirectory());
}

TRemoteFile *TRemoteFileList::GetFile(Integer Index) const
{
  return GetAs<TRemoteFile>(Index);
}

Boolean TRemoteFileList::GetIsRoot() const
{
  return (GetDirectory() == ROOTDIRECTORY);
}

UnicodeString TRemoteFileList::GetParentPath() const
{
  return base::UnixExtractFilePath(GetDirectory());
}

int64_t TRemoteFileList::GetTotalSize() const
{
  int64_t Result = 0;
  for (intptr_t Index = 0; Index < GetCount(); ++Index)
  {
    // if (!GetFile(Index)->GetIsDirectory())
    {
      Result += GetFile(Index)->GetSize();
    }
  }
  return Result;
}

TRemoteFile *TRemoteFileList::FindFile(UnicodeString AFileName) const
{
  for (intptr_t Index = 0; Index < GetCount(); ++Index)
  {
    if (GetFile(Index)->GetFileName() == AFileName)
    {
      return GetFile(Index);
    }
  }
  return nullptr;
}

//=== TRemoteDirectory ------------------------------------------------------
TRemoteDirectory::TRemoteDirectory(TTerminal *ATerminal, TRemoteDirectory *Template) :
  TRemoteFileList(OBJECT_CLASS_TRemoteDirectory),
  FTerminal(ATerminal),
  FParentDirectory(nullptr),
  FThisDirectory(nullptr),
  FIncludeParentDirectory(false),
  FIncludeThisDirectory(false)
{
  if (Template == nullptr)
  {
    FIncludeThisDirectory = false;
    FIncludeParentDirectory = true;
  }
  else
  {
    FIncludeThisDirectory = Template->FIncludeThisDirectory;
    FIncludeParentDirectory = Template->FIncludeParentDirectory;
  }
}

TRemoteDirectory::~TRemoteDirectory()
{
  ReleaseRelativeDirectories();
}

void TRemoteDirectory::ReleaseRelativeDirectories()
{
  if ((GetThisDirectory() != nullptr) && !GetIncludeThisDirectory())
  {
    SAFE_DESTROY(FThisDirectory);
  }
  if ((GetParentDirectory() != nullptr) && !GetIncludeParentDirectory())
  {
    SAFE_DESTROY(FParentDirectory);
  }
}

void TRemoteDirectory::Reset()
{
  ReleaseRelativeDirectories();
  TRemoteFileList::Reset();
}

void TRemoteDirectory::SetDirectory(UnicodeString Value)
{
  TRemoteFileList::SetDirectory(Value);
}

void TRemoteDirectory::AddFile(TRemoteFile *AFile)
{
  if (AFile->GetIsThisDirectory())
  {
    FThisDirectory = AFile;
  }
  if (AFile->GetIsParentDirectory())
  {
    FParentDirectory = AFile;
  }

  if ((!AFile->GetIsThisDirectory() || GetIncludeThisDirectory()) &&
    (!AFile->GetIsParentDirectory() || GetIncludeParentDirectory()))
  {
    TRemoteFileList::AddFile(AFile);
  }
  AFile->SetTerminal(GetTerminal());
}

void TRemoteDirectory::DuplicateTo(TRemoteFileList *Copy) const
{
  TRemoteFileList::DuplicateTo(Copy);
  if (GetThisDirectory() && !GetIncludeThisDirectory())
  {
    Copy->AddFile(GetThisDirectory()->Duplicate(false));
  }
  if (GetParentDirectory() && !GetIncludeParentDirectory())
  {
    Copy->AddFile(GetParentDirectory()->Duplicate(false));
  }
}

bool TRemoteDirectory::GetLoaded() const
{
  return ((GetTerminal() != nullptr) && GetTerminal()->GetActive() && !GetDirectory().IsEmpty());
}

void TRemoteDirectory::SetIncludeParentDirectory(Boolean Value)
{
  if (GetIncludeParentDirectory() != Value)
  {
    FIncludeParentDirectory = Value;
    if (Value && GetParentDirectory())
    {
      DebugAssert(IndexOf(GetParentDirectory()) < 0);
      Add(GetParentDirectory());
    }
    else if (!Value && GetParentDirectory())
    {
      DebugAssert(IndexOf(GetParentDirectory()) >= 0);
      Extract(GetParentDirectory());
    }
  }
}

void TRemoteDirectory::SetIncludeThisDirectory(Boolean Value)
{
  if (GetIncludeThisDirectory() != Value)
  {
    FIncludeThisDirectory = Value;
    if (Value && GetThisDirectory())
    {
      DebugAssert(IndexOf(GetThisDirectory()) < 0);
      Add(GetThisDirectory());
    }
    else if (!Value && GetThisDirectory())
    {
      DebugAssert(IndexOf(GetThisDirectory()) >= 0);
      Extract(GetThisDirectory());
    }
  }
}

TRemoteDirectoryCache::TRemoteDirectoryCache() : TStringList()
{
  TStringList::SetSorted(true);
  SetDuplicates(dupError);
  TStringList::SetCaseSensitive(true);
}

TRemoteDirectoryCache::~TRemoteDirectoryCache()
{
  TRemoteDirectoryCache::Clear();
}

void TRemoteDirectoryCache::Clear()
{
  TGuard Guard(FSection);

  try__finally
  {
    SCOPE_EXIT
    {
      TStringList::Clear();
    };
    for (intptr_t Index = 0; Index < GetCount(); ++Index)
    {
      TRemoteFileList *List = GetAs<TRemoteFileList>(Index);
      SAFE_DESTROY(List);
      SetObj(Index, nullptr);
    }
  }
  __finally
  {
#if 0
    TStringList::Clear();
#endif // #if 0
  };
}

bool TRemoteDirectoryCache::GetIsEmptyPrivate() const
{
  TGuard Guard(FSection);

  return (const_cast<TRemoteDirectoryCache *>(this)->GetCount() == 0);
}

bool TRemoteDirectoryCache::HasFileList(UnicodeString Directory) const
{
  TGuard Guard(FSection);

  intptr_t Index = IndexOf(base::UnixExcludeTrailingBackslash(Directory));
  return (Index >= 0);
}

bool TRemoteDirectoryCache::HasNewerFileList(UnicodeString Directory,
  const TDateTime &Timestamp) const
{
  TGuard Guard(FSection);

  intptr_t Index = IndexOf(base::UnixExcludeTrailingBackslash(Directory));
  if (Index >= 0)
  {
    TRemoteFileList *FileList = GetAs<TRemoteFileList>(Index);
    if (FileList->GetTimestamp() <= Timestamp)
    {
      Index = -1;
    }
  }
  return (Index >= 0);
}

bool TRemoteDirectoryCache::GetFileList(UnicodeString Directory,
  TRemoteFileList *FileList) const
{
  TGuard Guard(FSection);

  intptr_t Index = IndexOf(base::UnixExcludeTrailingBackslash(Directory));
  bool Result = (Index >= 0);
  if (Result)
  {
    DebugAssert(GetObj(Index) != nullptr);
    GetAs<TRemoteFileList>(Index)->DuplicateTo(FileList);
  }
  return Result;
}

void TRemoteDirectoryCache::AddFileList(TRemoteFileList *FileList)
{
  DebugAssert(FileList);
  if (FileList)
  {
    TRemoteFileList *Copy = new TRemoteFileList();
    FileList->DuplicateTo(Copy);

    TGuard Guard(FSection);

    // file list cannot be cached already with only one thread, but it can be
    // when directory is loaded by secondary terminal
    DoClearFileList(FileList->GetDirectory(), false);
    AddObject(Copy->GetDirectory(), Copy);
  }
}

void TRemoteDirectoryCache::ClearFileList(UnicodeString Directory, bool SubDirs)
{
  TGuard Guard(FSection);
  DoClearFileList(Directory, SubDirs);
}

void TRemoteDirectoryCache::DoClearFileList(UnicodeString Directory, bool SubDirs)
{
  UnicodeString Directory2 = base::UnixExcludeTrailingBackslash(Directory);
  intptr_t Index = IndexOf(Directory2);
  if (Index >= 0)
  {
    Delete(Index);
  }
  if (SubDirs)
  {
    Directory2 = base::UnixIncludeTrailingBackslash(Directory2);
    Index = GetCount() - 1;
    while (Index >= 0)
    {
      if (GetString(Index).SubString(1, Directory2.Length()) == Directory2)
      {
        Delete(Index);
      }
      Index--;
    }
  }
}

void TRemoteDirectoryCache::Delete(intptr_t Index)
{
  TRemoteFileList *List = GetAs<TRemoteFileList>(Index);
  SAFE_DESTROY(List);
  TStringList::Delete(Index);
}

TRemoteDirectoryChangesCache::TRemoteDirectoryChangesCache(intptr_t MaxSize) :
  TStringList(),
  FMaxSize(MaxSize)
{
}

void TRemoteDirectoryChangesCache::Clear()
{
  TStringList::Clear();
}

bool TRemoteDirectoryChangesCache::GetIsEmptyPrivate() const
{
  return (const_cast<TRemoteDirectoryChangesCache *>(this)->GetCount() == 0);
}

void TRemoteDirectoryChangesCache::SetValue(UnicodeString Name,
  UnicodeString Value)
{
  intptr_t Index = IndexOfName(Name);
  if (Index >= 0)
  {
    Delete(Index);
  }
  TStringList::SetValue(Name, Value);
}

UnicodeString TRemoteDirectoryChangesCache::GetValue(UnicodeString Name)
{
  UnicodeString Value = TStringList::GetValue(Name);
  TStringList::SetValue(Name, Value);
  return Value;
}

void TRemoteDirectoryChangesCache::AddDirectoryChange(
  UnicodeString SourceDir, UnicodeString Change,
  UnicodeString TargetDir)
{
  DebugAssert(!TargetDir.IsEmpty());
  SetValue(TargetDir, L"//");
  if (TTerminal::ExpandFileName(Change, SourceDir) != TargetDir)
  {
    UnicodeString Key;
    if (DirectoryChangeKey(SourceDir, Change, Key))
    {
      SetValue(Key, TargetDir);
    }
  }
}

void TRemoteDirectoryChangesCache::ClearDirectoryChange(
  UnicodeString SourceDir)
{
  for (intptr_t Index = 0; Index < GetCount(); ++Index)
  {
    if (GetName(Index).SubString(1, SourceDir.Length()) == SourceDir)
    {
      Delete(Index);
      Index--;
    }
  }
}

void TRemoteDirectoryChangesCache::ClearDirectoryChangeTarget(
  UnicodeString TargetDir)
{
  UnicodeString Key;
  // hack to clear at least local sym-link change in case symlink is deleted
  DirectoryChangeKey(base::UnixExcludeTrailingBackslash(base::UnixExtractFilePath(TargetDir)),
    base::UnixExtractFileName(TargetDir), Key);

  for (intptr_t Index = 0; Index < GetCount(); ++Index)
  {
    UnicodeString Name = GetName(Index);
    if ((Name.SubString(1, TargetDir.Length()) == TargetDir) ||
      (GetValue(Name).SubString(1, TargetDir.Length()) == TargetDir) ||
      (!Key.IsEmpty() && (Name == Key)))
    {
      Delete(Index);
      Index--;
    }
  }
}

bool TRemoteDirectoryChangesCache::GetDirectoryChange(
  UnicodeString SourceDir, UnicodeString Change, UnicodeString &TargetDir) const
{
  UnicodeString Key = TTerminal::ExpandFileName(Change, SourceDir);
  bool Result = (IndexOfName(Key) >= 0);
  if (Result)
  {
    TargetDir = GetValue(Key);
    // TargetDir is not "//" here only when Change is full path to symbolic link
    if (TargetDir == L"//")
    {
      TargetDir = Key;
    }
  }
  else
  {
    Result = DirectoryChangeKey(SourceDir, Change, Key);
    if (Result)
    {
      UnicodeString Directory = GetValue(Key);
      Result = !Directory.IsEmpty();
      if (Result)
      {
        TargetDir = Directory;
      }
    }
  }
  return Result;
}

void TRemoteDirectoryChangesCache::Serialize(UnicodeString &Data) const
{
  Data = L"A";
  intptr_t ACount = GetCount();
  if (ACount > FMaxSize)
  {
    std::unique_ptr<TStrings> Limited(new TStringList());
    try__finally
    {
      intptr_t Index = ACount - FMaxSize;
      while (Index < ACount)
      {
        Limited->Add(GetString(Index));
        ++Index;
      }
      Data += Limited->GetText();
    }
    __finally
    {
#if 0
      delete Limited;
#endif // #if 0
    };
  }
  else
  {
    Data += GetText();
  }
}

void TRemoteDirectoryChangesCache::Deserialize(UnicodeString Data)
{
  if (Data.IsEmpty())
  {
    SetText(L"");
  }
  else
  {
    SetText(Data.c_str() + 1);
  }
}

bool TRemoteDirectoryChangesCache::DirectoryChangeKey(
  UnicodeString SourceDir, UnicodeString Change, UnicodeString &Key)
{
  bool Result = !Change.IsEmpty();
  if (Result)
  {
    bool Absolute = base::UnixIsAbsolutePath(Change);
    Result = !SourceDir.IsEmpty() || Absolute;
    if (Result)
    {
      // expanded from ?: to avoid memory leaks
      if (Absolute)
      {
        Key = Change;
      }
      else
      {
        Key = SourceDir + L"," + Change;
      }
    }
  }
  return Result;
}

const wchar_t TRights::BasicSymbols[] = L"rwxrwxrwx";
const wchar_t TRights::CombinedSymbols[] = L"--s--s--t";
const wchar_t TRights::ExtendedSymbols[] = L"--S--S--T";
const wchar_t TRights::ModeGroups[] = L"ugo";

TRights::TRights() :
  FSet(0),
  FUnset(0),
  FAllowUndef(false),
  FUnknown(true)
{
  SetNumber(0);
}

TRights::TRights(uint16_t ANumber) :
  FSet(0),
  FUnset(0),
  FAllowUndef(false),
  FUnknown(true)
{
  SetNumber(ANumber);
}

TRights::TRights(const TRights &Source)
{
  Assign(&Source);
}

void TRights::Assign(const TRights *Source)
{
  FAllowUndef = Source->GetAllowUndef();
  FSet = Source->FSet;
  FUnset = Source->FUnset;
  FText = Source->FText;
  FUnknown = Source->FUnknown;
}

TRights::TFlag TRights::RightToFlag(TRight Right)
{
  return static_cast<TFlag>(1 << (rrLast - Right));
}

bool TRights::operator==(const TRights &rhr) const
{
  if (GetAllowUndef() || rhr.GetAllowUndef())
  {
    for (int Right = rrFirst; Right <= rrLast; Right++)
    {
      if (GetRightUndef(static_cast<TRight>(Right)) !=
        rhr.GetRightUndef(static_cast<TRight>(Right)))
      {
        return false;
      }
    }
    return true;
  }
  return (GetNumber() == rhr.GetNumber());
}

bool TRights::operator==(uint16_t rhr) const
{
  return (GetNumber() == rhr);
}

bool TRights::operator!=(const TRights &rhr) const
{
  return !(*this == rhr);
}

TRights &TRights::operator=(uint16_t rhr)
{
  SetNumber(rhr);
  return *this;
}

TRights &TRights::operator=(const TRights &rhr)
{
  Assign(&rhr);
  return *this;
}

TRights TRights::operator~() const
{
  TRights Result(static_cast<uint16_t>(~GetNumber()));
  return Result;
}

TRights TRights::operator&(const TRights &rhr) const
{
  TRights Result(*this);
  Result &= rhr;
  return Result;
}

TRights TRights::operator&(uint16_t rhr) const
{
  TRights Result(*this);
  Result &= rhr;
  return Result;
}

TRights &TRights::operator&=(const TRights &rhr)
{
  if (GetAllowUndef() || rhr.GetAllowUndef())
  {
    for (int Right = rrFirst; Right <= rrLast; Right++)
    {
      if (GetRightUndef(static_cast<TRight>(Right)) !=
        rhr.GetRightUndef(static_cast<TRight>(Right)))
      {
        SetRightUndef(static_cast<TRight>(Right), rsUndef);
      }
    }
  }
  else
  {
    SetNumber(GetNumber() & rhr.GetNumber());
  }
  return *this;
}

TRights &TRights::operator&=(uint16_t rhr)
{
  SetNumber(GetNumber() & rhr);
  return *this;
}

TRights TRights::operator|(const TRights &rhr) const
{
  TRights Result(*this);
  Result |= rhr;
  return Result;
}

TRights TRights::operator|(uint16_t rhr) const
{
  TRights Result(*this);
  Result |= rhr;
  return Result;
}

TRights &TRights::operator|=(const TRights &rhr)
{
  SetNumber(GetNumber() | rhr.GetNumber());
  return *this;
}

TRights &TRights::operator|=(uint16_t rhr)
{
  SetNumber(GetNumber() | rhr);
  return *this;
}

void TRights::SetAllowUndef(bool Value)
{
  if (FAllowUndef != Value)
  {
    DebugAssert(!Value || ((FSet | FUnset) == rfAllSpecials));
    FAllowUndef = Value;
  }
}

void TRights::SetText(UnicodeString Value)
{
  if (Value != GetText())
  {
    if ((Value.Length() != TextLen) ||
      (!GetAllowUndef() && (Value.Pos(UndefSymbol) > 0)) ||
      (Value.Pos(L" ") > 0))
    {
      throw Exception(FMTLOAD(RIGHTS_ERROR, Value));
    }

    FSet = 0;
    FUnset = 0;
    intptr_t Flag = 00001;
    int ExtendedFlag = 01000; //-V536
    bool KeepText = false;
    for (intptr_t Index = TextLen; Index >= 1; Index--)
    {
      if (Value[Index] == UnsetSymbol)
      {
        FUnset |= static_cast<uint16_t>(Flag | ExtendedFlag);
      }
      else if (Value[Index] == UndefSymbol)
      {
        // do nothing
      }
      else if (Value[Index] == CombinedSymbols[Index - 1])
      {
        FSet |= static_cast<uint16_t>(Flag | ExtendedFlag);
      }
      else if (Value[Index] == ExtendedSymbols[Index - 1])
      {
        FSet |= static_cast<uint16_t>(ExtendedFlag);
        FUnset |= static_cast<uint16_t>(Flag);
      }
      else
      {
        if (Value[Index] != BasicSymbols[Index - 1])
        {
          KeepText = true;
        }
        FSet |= static_cast<uint16_t>(Flag);
        if (Index % 3 == 0)
        {
          FUnset |= static_cast<uint16_t>(ExtendedFlag);
        }
      }

      Flag <<= 1;
      if (Index % 3 == 1)
      {
        ExtendedFlag <<= 1;
      }
    }

    FText = KeepText ? Value : UnicodeString();
  }
  FUnknown = false;
}

UnicodeString TRights::GetText() const
{
  if (!FText.IsEmpty())
  {
    return FText;
  }
  else
  {
    UnicodeString Result(TextLen, 0);

    intptr_t Flag = 00001;
    int ExtendedFlag = 01000; //-V536
    bool ExtendedPos = true;
    wchar_t Symbol;
    intptr_t Index = TextLen;
    while (Index >= 1)
    {
      if (ExtendedPos &&
        ((FSet & (Flag | ExtendedFlag)) == (Flag | ExtendedFlag)))
      {
        Symbol = CombinedSymbols[Index - 1];
      }
      else if ((FSet & Flag) != 0)
      {
        Symbol = BasicSymbols[Index - 1];
      }
      else if (ExtendedPos && ((FSet & ExtendedFlag) != 0))
      {
        Symbol = ExtendedSymbols[Index - 1];
      }
      else if ((!ExtendedPos && ((FUnset & Flag) == Flag)) ||
        (ExtendedPos && ((FUnset & (Flag | ExtendedFlag)) == (Flag | ExtendedFlag))))
      {
        Symbol = UnsetSymbol;
      }
      else
      {
        Symbol = UndefSymbol;
      }

      Result[Index] = Symbol;

      Flag <<= 1;
      Index--;
      ExtendedPos = ((Index % 3) == 0);
      if (ExtendedPos)
      {
        ExtendedFlag <<= 1;
      }
    }
    return Result;
  }
}

void TRights::SetOctal(UnicodeString AValue)
{
  UnicodeString Value(AValue);
  if (Value.Length() == 3)
  {
    Value = L"0" + Value;
  }

  if (GetOctal() != Value)
  {
    bool Correct = (Value.Length() == 4);
    if (Correct)
    {
      for (intptr_t Index = 1; (Index <= Value.Length()) && Correct; ++Index)
      {
        Correct = (Value[Index] >= L'0') && (Value[Index] <= L'7');
      }
    }

    if (!Correct)
    {
      throw Exception(FMTLOAD(INVALID_OCTAL_PERMISSIONS, AValue));
    }

    SetNumber(static_cast<uint16_t>(
        ((Value[1] - L'0') << 9) +
        ((Value[2] - L'0') << 6) +
        ((Value[3] - L'0') << 3) +
        ((Value[4] - L'0') << 0)));
  }
  FUnknown = false;
}

uint32_t TRights::GetNumberDecadic() const
{
  uint32_t N = GetNumberSet(); // used to be "Number"
  uint32_t Result =
    ((N & 07000) / 01000 * 1000) +
    ((N & 00700) /  0100 *  100) +
    ((N & 00070) /   010 *   10) +
    ((N & 00007) /    01 *    1);

  return Result;
}

UnicodeString TRights::GetOctal() const
{
  UnicodeString Result;
  uint16_t N = GetNumberSet(); // used to be "Number"
  Result.SetLength(4);
  Result[1] = static_cast<wchar_t>(L'0' + ((N & 07000) >> 9));
  Result[2] = static_cast<wchar_t>(L'0' + ((N & 00700) >> 6));
  Result[3] = static_cast<wchar_t>(L'0' + ((N & 00070) >> 3));
  Result[4] = static_cast<wchar_t>(L'0' + ((N & 00007) >> 0));

  return Result;
}

void TRights::SetNumber(uint16_t Value)
{
  if ((FSet != Value) || ((FSet | FUnset) != rfAllSpecials))
  {
    FSet = Value;
    FUnset = static_cast<uint16_t>(rfAllSpecials & ~FSet);
    FText.Clear();
  }
  FUnknown = false;
}

uint16_t TRights::GetNumber() const
{
  DebugAssert(!GetIsUndef());
  return FSet;
}

void TRights::SetRight(TRight Right, bool Value)
{
  SetRightUndef(Right, (Value ? rsYes : rsNo));
}

bool TRights::GetRight(TRight Right) const
{
  TState State = GetRightUndef(Right);
  DebugAssert(State != rsUndef);
  return (State == rsYes);
}

void TRights::SetRightUndef(TRight Right, TState Value)
{
  if (Value != GetRightUndef(Right))
  {
    DebugAssert((Value != rsUndef) || GetAllowUndef());

    TFlag Flag = RightToFlag(Right);

    switch (Value)
    {
    case rsYes:
      FSet |= static_cast<uint16_t>(Flag);
      FUnset &= static_cast<uint16_t>(~Flag);
      break;

    case rsNo:
      FSet &= static_cast<uint16_t>(~Flag);
      FUnset |= static_cast<uint16_t>(Flag);
      break;

    case rsUndef:
    default:
      FSet &= static_cast<uint16_t>(~Flag);
      FUnset &= static_cast<uint16_t>(~Flag);
      break;
    }

    FText.Clear();
  }
  FUnknown = false;
}

TRights::TState TRights::GetRightUndef(TRight Right) const
{
  TFlag Flag = RightToFlag(Right);
  TState Result;

  if ((FSet & Flag) != 0)
  {
    Result = rsYes;
  }
  else if ((FUnset & Flag) != 0)
  {
    Result = rsNo;
  }
  else
  {
    Result = rsUndef;
  }
  return Result;
}

void TRights::SetReadOnly(bool Value)
{
  SetRight(rrUserWrite, !Value);
  SetRight(rrGroupWrite, !Value);
  SetRight(rrOtherWrite, !Value);
}

bool TRights::GetReadOnly() const
{
  return GetRight(rrUserWrite) && GetRight(rrGroupWrite) && GetRight(rrOtherWrite);
}

UnicodeString TRights::GetSimplestStr() const
{
  if (GetIsUndef())
  {
    return GetModeStr();
  }
  return GetOctal();
}

UnicodeString TRights::GetModeStr() const
{
  UnicodeString Result;
  UnicodeString SetModeStr, UnsetModeStr;
  TRight Right;
  intptr_t Index;

  for (intptr_t Group = 0; Group < 3; Group++)
  {
    SetModeStr.Clear();
    UnsetModeStr.Clear();
    for (intptr_t Mode = 0; Mode < 3; Mode++)
    {
      Index = (Group * 3) + Mode;
      Right = static_cast<TRight>(rrUserRead + Index);
      switch (GetRightUndef(Right))
      {
      case rsYes:
        SetModeStr += BasicSymbols[Index];
        break;

      case rsNo:
        UnsetModeStr += BasicSymbols[Index];
        break;

      case rsUndef:
        break;
      }
    }

    Right = static_cast<TRight>(rrUserIDExec + Group);
    Index = (Group * 3) + 2;
    switch (GetRightUndef(Right))
    {
    case rsYes:
      SetModeStr += CombinedSymbols[Index];
      break;

    case rsNo:
      UnsetModeStr += CombinedSymbols[Index];
      break;

    case rsUndef:
      break;
    }

    if (!SetModeStr.IsEmpty() || !UnsetModeStr.IsEmpty())
    {
      if (!Result.IsEmpty())
      {
        Result += L',';
      }
      Result += ModeGroups[Group];
      if (!SetModeStr.IsEmpty())
      {
        Result += L"+" + SetModeStr;
      }
      if (!UnsetModeStr.IsEmpty())
      {
        Result += L"-" + UnsetModeStr;
      }
    }
  }
  return Result;
}

void TRights::AddExecute()
{
  for (int Group = 0; Group < 3; Group++)
  {
    if ((GetRightUndef(static_cast<TRight>(rrUserRead + (Group * 3))) == rsYes) ||
      (GetRightUndef(static_cast<TRight>(rrUserWrite + (Group * 3))) == rsYes))
    {
      SetRight(static_cast<TRight>(rrUserExec + (Group * 3)), true);
    }
  }
  FUnknown = false;
}

void TRights::AllUndef()
{
  if ((FSet != 0) || (FUnset != 0))
  {
    FSet = 0;
    FUnset = 0;
    FText.Clear();
  }
  FUnknown = false;
}

bool TRights::GetIsUndef() const
{
  return ((FSet | FUnset) != rfAllSpecials);
}

TRights::operator uint16_t() const
{
  return GetNumber();
}

TRights::operator uint32_t() const
{
  return GetNumber();
}

TRemoteProperties::TRemoteProperties() :
  TObject(OBJECT_CLASS_TRemoteProperties)
{
  Default();
}

TRemoteProperties::TRemoteProperties(const TRemoteProperties &rhp) :
  TObject(OBJECT_CLASS_TRemoteProperties),
  Valid(rhp.Valid),
  Rights(rhp.Rights),
  Group(rhp.Group),
  Owner(rhp.Owner),
  Modification(rhp.Modification),
  LastAccess(rhp.Modification),
  Recursive(rhp.Recursive),
  AddXToDirectories(rhp.AddXToDirectories)
{
}

void TRemoteProperties::Default()
{
  Valid.Clear();
  AddXToDirectories = false;
  Rights.SetAllowUndef(false);
  Rights.SetNumber(0);
  Group.Clear();
  Owner.Clear();
  Modification = 0;
  LastAccess = 0;
  Recursive = false;
}

bool TRemoteProperties::operator==(const TRemoteProperties &rhp) const
{
  bool Result = (Valid == rhp.Valid && Recursive == rhp.Recursive);

  if (Result)
  {
    if ((Valid.Contains(vpRights) &&
        (Rights != rhp.Rights || AddXToDirectories != rhp.AddXToDirectories)) ||
      (Valid.Contains(vpOwner) && (Owner != rhp.Owner)) ||
      (Valid.Contains(vpGroup) && (Group != rhp.Group)) ||
      (Valid.Contains(vpModification) && (Modification != rhp.Modification)) ||
      (Valid.Contains(vpLastAccess) && (LastAccess != rhp.LastAccess)))
    {
      Result = false;
    }
  }
  return Result;
}

bool TRemoteProperties::operator!=(const TRemoteProperties &rhp) const
{
  return !(*this == rhp);
}

TRemoteProperties TRemoteProperties::CommonProperties(TStrings *AFileList)
{
  TODO("Modification and LastAccess");
  TRemoteProperties CommonProperties;
  for (intptr_t Index = 0; Index < AFileList->GetCount(); ++Index)
  {
    TRemoteFile *File = AFileList->GetAs<TRemoteFile>(Index);
    DebugAssert(File);
    if (!Index)
    {
      CommonProperties.Rights.Assign(File->GetRights());
      // previously we allowed undef implicitly for directories,
      // now we do it explicitly in properties dialog and only in combination
      // with "recursive" option
      CommonProperties.Rights.SetAllowUndef(File->GetRights()->GetIsUndef());
      CommonProperties.Valid << vpRights;
      if (File->GetFileOwner().GetIsSet())
      {
        CommonProperties.Owner = File->GetFileOwner();
        CommonProperties.Valid << vpOwner;
      }
      if (File->GetFileGroup().GetIsSet())
      {
        CommonProperties.Group = File->GetFileGroup();
        CommonProperties.Valid << vpGroup;
      }
    }
    else
    {
      CommonProperties.Rights.SetAllowUndef(True);
      CommonProperties.Rights &= *File->GetRights();
      if (CommonProperties.Owner != File->GetFileOwner())
      {
        CommonProperties.Owner.Clear();
        CommonProperties.Valid >> vpOwner;
      }
      if (CommonProperties.Group != File->GetFileGroup())
      {
        CommonProperties.Group.Clear();
        CommonProperties.Valid >> vpGroup;
      }
    }
  }
  return CommonProperties;
}

TRemoteProperties TRemoteProperties::ChangedProperties(
  const TRemoteProperties &OriginalProperties, TRemoteProperties &NewProperties)
{
  TODO("Modification and LastAccess");
  if (!NewProperties.Recursive)
  {
    if (NewProperties.Rights == OriginalProperties.Rights &&
      !NewProperties.AddXToDirectories)
    {
      NewProperties.Valid >> vpRights;
    }

    if (NewProperties.Group == OriginalProperties.Group)
    {
      NewProperties.Valid >> vpGroup;
    }

    if (NewProperties.Owner == OriginalProperties.Owner)
    {
      NewProperties.Valid >> vpOwner;
    }
  }
  return NewProperties;
}

TRemoteProperties &TRemoteProperties::operator=(const TRemoteProperties &other)
{
  Valid = other.Valid;
  Rights = other.Rights;
  Group = other.Group;
  Owner = other.Owner;
  Modification = other.Modification;
  LastAccess = other.LastAccess;
  Recursive = other.Recursive;
  AddXToDirectories = other.AddXToDirectories;
  return *this;
}

void TRemoteProperties::Load(THierarchicalStorage *Storage)
{
  uint8_t Buf[sizeof(Valid)];
  if (static_cast<size_t>(Storage->ReadBinaryData("Valid", &Buf, sizeof(Buf))) == sizeof(Buf))
  {
    memmove(&Valid, Buf, sizeof(Valid));
  }

  if (Valid.Contains(vpRights))
  {
    Rights.SetText(Storage->ReadString("Rights", Rights.GetText()));
  }

  // TODO
}

void TRemoteProperties::Save(THierarchicalStorage *Storage) const
{
  Storage->WriteBinaryData(UnicodeString(L"Valid"),
    static_cast<const void *>(&Valid), sizeof(Valid));

  if (Valid.Contains(vpRights))
  {
    Storage->WriteString("Rights", Rights.GetText());
  }

  // TODO
}

