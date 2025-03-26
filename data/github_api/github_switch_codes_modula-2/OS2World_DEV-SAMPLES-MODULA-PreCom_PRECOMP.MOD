// Repository: OS2World/DEV-SAMPLES-MODULA-PreCom
// File: PRECOMP.MOD

MODULE PreComp;

(****************************************************************************


   Precompiler for Mod2 Programs written for the IBM Database Manager
   Written by: Ed Ross
   Date      : 9/24/92


   Copyright Ed Ross, PMI 1992, All rights reserved


   Some ramdom thoughts you will want to remove

   This precompile will process Mod2 programs with embedded SQL statements
   and translate them to the appropiate calls to the IBM Database Manager.

   The DEFINITION modules must have the extention "SQD"
   The IMPLEMENTATION modules must have the extention "SQM"

   The precomile MUST compile both the def and the mod files at the same
   time.  (NOTE - the definition module is optional.  It is only included
   so record types and variables can be shared.  The implementation module
   must be present).  The definition module may only have the
       EXEC SQL BEGIN DECLARE
   and EXEC SQL END DECLARE sections.

   The input parameters are the same as the IBM precompile.  However, not
   all options make sense in a MOD2 envirionment.


   Only simple types are alowed in the DECLARE section.  Any complex type
   is ignored.  (you can create complex types or objects by adding 
   there definition to the table - then expanding them at compile time)

    I use a Genlis module that uses the Object extentions for JPI.  This could
    easily be replaced with just record reads from a file or the GenList
    from PMI.


 NEW FEATURE
   A mod file can "import" other def files to get additional host variables
   by adding a (* SQLIMPORT filename *)  The file name is assumed not to have
   any etentions

    N O T E  - this should be changed to use the redirection file from
    the JPI stuff


***************************************************************)


FROM Lists IMPORT GenList,StrElmt;
FROM ParseVar IMPORT GetVars,RemoveComment,SQLStatement,PrepareSQL,GetVarName;
FROM FIO IMPORT File,WrStr,WrLn,Create,WrCard,Close,StandardOutput,Exists;
FROM FormIO IMPORT WrF1,WrF2,WrF3,WrF4,WrF5;
FROM IO IMPORT WrStrRedirect;
FROM Str IMPORT Copy,Pos,Append,Delete,Length,Subst,Caps,Compare;
FROM Lib IMPORT Fill,ParamCount,ParamStr;
FROM SQLCA IMPORT sqlca;
FROM SQLStuff IMPORT SQLErrorHandler;
IMPORT SQLPREP;
FROM SQL IMPORT SQLGINTP;
FROM SQLPREP IMPORT SQLA_START, SQLA_DECLARE, SQLA_INCLUDE, SQLA_ALLOC_INPUT,
      SQLA_ALLOC_OUTPUT, SQLA_SETS, SQLA_USDA_INPUT, SQLA_USDA_OUTPUT, SQLA_CALL,
      SQLA_DEALLOC, SQLA_STOP,	 SQLA_SQLERROR,	 SQLA_SQLWARNING, SQLA_NOT_FOUND;	


CONST
  MaxSqlStmLen = 2000;
  MaxLineLength = 320;


TYPE
  ProcessType = (DefFile,ModFile,Done);
VAR
  Process : ProcessType;
  TheString : StrElmt;
  TheStr : ARRAY[0..MaxLineLength] OF CHAR;
  TestStr : ARRAY[0..MaxLineLength] OF CHAR;
  TheList : GenList;
  J : CARDINAL;
  Statement : ARRAY[0..MaxSqlStmLen] OF CHAR;
  SectionNbr : CARDINAL;
  SQLComment : CARDINAL;
  Optimize   : BOOLEAN;  (* variable initialization *)

  InComment  : CARDINAL;
  OutFile    : File;
  InDeclare  : BOOLEAN;
  MsgFile    : File;

  sqldaID    : CARDINAL;    (* needed for runtime services *)
  InProgName : ARRAY[0..12] OF CHAR;
(* data structures used by precompiler services *)


  Options    : SQLPREP.sqla_options;
  TaskArray  : SQLPREP.sqla_tasks;
  TokenArray : SQLPREP.sqla_tokens;
  ProgramRec : SQLPREP.sqla_program_id;
  CA         : sqlca;



PROCEDURE  Intercept(Str : ARRAY OF CHAR);
(* this routine will take the output of WrStr from IO and place it in
   the global TheStr.  This is done so I can use the format IO mixed
   with the regular file IO

*)
BEGIN
  Copy(TheStr,Str);
END Intercept;


PROCEDURE CalcOptionCnt(Opt : SQLPREP.sqla_options): LONGCARD;
VAR
  S : CARDINAL;
BEGIN
  S := SIZE(Opt) - SIZE(Opt.Header); (* size of options field*)
  RETURN LONGCARD(S DIV SIZE(Opt.Option[1]))-1;
END CalcOptionCnt;

PROCEDURE ParamType(InParam : ARRAY OF CHAR; VAR Typ : CHAR; VAR  Value : ARRAY OF CHAR);

BEGIN
  Caps(InParam);
  IF (Pos(InParam,'.') < MAX(CARDINAL))  (* no extention existed *)
  THEN InParam[Pos(InParam,'.')] := 0C;  (* set to no extention *)
  END;
  Copy(Value,'');
  Subst(InParam,'-','/');  (* allow either the '/' or '-' switch *)
  IF Pos(InParam,'/') = MAX(CARDINAL)  (* no switch was supplied *)
    THEN
      Typ := 'D';   (* assume it was the database name *)
      Copy(Value,InParam);
      RETURN;
    END;
  Typ := InParam[1];  (* take the char after the switch *)
  IF Pos(InParam,'=') = MAX(CARDINAL)
    THEN RETURN;  (* no value supplied *)
    END;
  Copy(Value,InParam[Pos(InParam,'=')+1]);  (* copy the value *)


END ParamType;


PROCEDURE Usage();  (* if no parameters or less than two pramaeters specified*)
BEGIN
  WrStr(MsgFile,'  * * * * Modula 2 SQL Precompiler for IBM Database Manager ');
  WrLn(MsgFile);
  WrStr(MsgFile,'Usage is same as the IBM "SQLPREP" precompiler for "C"');
END Usage;

PROCEDURE SetOption(VAR  Opt : SQLPREP.sqla_options; Type, Option : LONGCARD);
VAR Idx : CARDINAL;
BEGIN
 INC(Opt.Header.used);  (* use next available option *)
 Idx := CARDINAL(Opt.Header.used);
 Opt.Option[Idx].type :=Type;
 Opt.Option[Idx].value := Option;
END SetOption;


PROCEDURE Initialize();
TYPE
  Param = ARRAY[0..80] OF CHAR;
VAR J : CARDINAL;
  L1,L2,L3,L4,L5 : CARDINAL;

  OutProgName : ARRAY[0..12] OF CHAR;
  BindFileName : ARRAY[0..12] OF CHAR;
  InParams     : ARRAY[1..12] OF Param;
  DataBaseName : ARRAY[0..12] OF CHAR;
  Password     : ARRAY[0..12] OF CHAR;
  MsgFileName  : ARRAY[0..12] OF CHAR;
  TmpProgName  : ARRAY[0..12] OF CHAR;

  PT           : CHAR;
  Value        : ARRAY[0..80] OF CHAR;



(* options used by database manager *)
  Blocking,
  BindFile,
  Format,
  Cursor,
  Plan : LONGCARD;


BEGIN
(* assign defaults to the input parameters *)

  Optimize := FALSE;
  Blocking := SQLPREP.SQLA_NO_BLOCK;
  Format   := SQLPREP.SQLA_POA_DEF;
  BindFile := SQLPREP.SQLA_NO_BIND_FILE;
  Plan     := SQLPREP.SQLA_CREATE_PLAN;

  WrStrRedirect := Intercept;  (* intercept the results of formated IO *)
  MsgFile := StandardOutput;
  sqldaID := 0;
  Fill(ADR(Options),SIZE(Options),0);
  Fill(ADR(CA),SIZE(sqlca),0);
  Fill(ADR(TaskArray),SIZE(TaskArray),0);
  Fill(ADR(TokenArray),SIZE(TokenArray),0);


  Options.Header.allocated := CalcOptionCnt(Options);
  TaskArray.Header.allocated  := SQLPREP.MaxTasks;
  TokenArray.Header.allocated := SQLPREP.MaxTokens;


  (*


  *)


  FOR J := 1 TO ParamCount() DO  (* get input parameters *)
    ParamStr(InParams[J],J);
  END;

  IF J < 2
    THEN
      Usage();
      HALT;
    END;
  Fill(ADR(ProgramRec),SIZE(ProgramRec),0);
  Copy(InProgName,  InParams[1]);
  InProgName[8] :=0C;   (* program name SIZE MAX 8 *)
  Copy(DataBaseName,InParams[2]);
  IF Length(DataBaseName) < 2
    THEN Copy(DataBaseName,InProgName);
    END;



  IF (Pos(InProgName,'.') < MAX(CARDINAL))  (* no extention existed *)
    THEN InProgName[Pos(InProgName,'.')] := 0C;
    END;

  Copy(TmpProgName,InProgName);
  Copy(OutProgName,InProgName);
  Copy(MsgFileName,InProgName);
  Append(MsgFileName,'.log');
  MsgFile := Create(MsgFileName);
  WrStr(MsgFile,'Precompiler FOR - ');
  WrStr(MsgFile,InProgName);
  WrLn(MsgFile);
  Append(TmpProgName,'.SQD');
  IF Exists(TmpProgName)
    THEN
      Append(OutProgName,'.DEF');
      Process := ModFile;    (* set up for next time *)
    ELSE
      Copy(TmpProgName,InProgName);
      Append(TmpProgName,'.sqm');
      IF NOT Exists(TmpProgName)
        THEN
          WrStr(MsgFile,'Not source specified ');
          WrStr(MsgFile,InProgName);
          Close(MsgFile);
          HALT;
        END;
      Append(OutProgName,'.MOD');
      Process := Done;       (* no def file - process the mod file *)
  END;


  Copy(BindFileName,InProgName);

  Append(BindFileName,'.bnd');

  J := 2;
  LOOP
   IF J > ParamCount()
     THEN EXIT;
     END;
   ParamType(InParams[J],PT,Value);
   CASE PT OF
     'D' : Copy(DataBaseName,Value);   (* override database name *)

    |'B' : BindFile := SQLPREP.SQLA_CREATE_BIND_FILE;
           IF Length(Value) > 0
             THEN
               Copy(BindFileName,Value);
               Append(BindFileName,'.bnd');
             END;
    |'C' : (* nothing  what to do with include SQLCA/DA *)
    |'F' : IF Pos(Value,'USA') < MAX(CARDINAL)
             THEN Format := SQLPREP.SQLA_POA_USA;
            ELSIF Pos(Value,'EUR') < MAX(CARDINAL)
             THEN Format := SQLPREP.SQLA_POA_EUR;
            ELSIF Pos(Value,'ISO') < MAX(CARDINAL)
             THEN Format := SQLPREP.SQLA_POA_ISO;
            ELSIF Pos(Value,'JIS') < MAX(CARDINAL)
             THEN Format := SQLPREP.SQLA_POA_JIS;
            ELSIF Pos(Value,'LOC') < MAX(CARDINAL)
              THEN Format := SQLPREP.SQLA_POA_LOC;
            ELSE Format := SQLPREP.SQLA_POA_DEF;   (* default *)
            END;
   |'I'  : IF Pos(Value,'RR') < MAX (CARDINAL)
             THEN Cursor := SQLPREP.SQLA_REPEATABLE_READ;
            ELSIF Pos(Value,'UR') < MAX(CARDINAL)
             THEN Cursor := SQLPREP.SQLA_UNCOMMITTED_READ;
            ELSE Cursor := SQLPREP.SQLA_CURSOR_STABILITY;
           END;
           SetOption(Options,SQLPREP.SQLA_ISOLATION,Cursor);
   |'K'  : IF Pos(Value,'ALL') < MAX(CARDINAL)
             THEN
               Blocking := SQLPREP.SQLA_BLOCK_ALL;
             ELSIF Pos(Value,'UNAMBIG') < MAX(CARDINAL)
               THEN Blocking := SQLPREP.SQLA_BLOCK_UNAMBIG;
             ELSE Blocking := SQLPREP.SQLA_NO_BLOCK;
           END;
           SetOption(Options,SQLPREP.SQLA_BLOCK,Blocking);
   |'P'  : Plan := SQLPREP.SQLA_CREATE_PLAN;
   |'M'  : MsgFile := Create(Value);
   |'O'  : Optimize := TRUE;
   END;
   INC(J);
 END;  (* end of loop *)

(* these three options must always be present in the initilaization *)

  SetOption(Options,SQLPREP.SQLA_FORMAT,Format);
  SetOption(Options,SQLPREP.SQLA_ACCESS_PLAN,Plan);
  SetOption(Options,SQLPREP.SQLA_BIND_FILE,BindFile);

  L1 := Length(InProgName);
  L2 := Length(DataBaseName);
  L3 := 0;
  L4 := Length(BindFileName);

  WrStr(MsgFile,'Starting MOD2 precompiler for IBM Database Manager');
  SQLPREP.SQLGINIT(L1,TmpProgName,
                   L2,DataBaseName,
                   L3,Password,
                   L4,BindFileName,
                   Options,
                   ProgramRec,
                   NIL,
                   CA);

  IF CA.sqlcode < 0
    THEN
      SQLErrorHandler(MsgFile,'Initializing',CA);
    END;
  TheList.FileToList(TmpProgName);
  OutFile := Create(OutProgName);

END Initialize;



PROCEDURE InitializeMod();
VAR
  OutProgName  : ARRAY[0..12] OF CHAR;
BEGIN
  Process := Done;
  Copy(OutProgName,InProgName);
  Append(OutProgName,'.mod');
  Append(InProgName,'.sqm');   (* set up for mod file *)
  TheList.DisposeList();

  TheList.FileToList(InProgName);
  OutFile := Create(OutProgName);

END InitializeMod;


VAR
  InputSQLDA,OutputSQLDA : CARDINAL;

PROCEDURE GenCode();
  (* use the task array to determine the type of code to generate *)
CONST

VAR TaskNbr : CARDINAL;
    J,K     : CARDINAL;
    typ     : CARDINAL;
    len     : CARDINAL;
    VarName : ARRAY[0..30] OF CHAR;
    CallType : CARDINAL;
    val     : CARDINAL;
    IndVar  : ARRAY[0..30] OF CHAR;
    PID     : POINTER TO ARRAY[1..40] OF CHAR;


PROCEDURE GenSetV(K, Index: CARDINAL);
BEGIN
    WrF4('    RC :=SQLPREP.SQLASETV(%u,%u,%u,SIZE(%s),',sqldaID,Index,typ,VarName);
    WrStr(OutFile,TheStr);
    IF (Compare(IndVar,'NIL') = 0)
      THEN
         WrF2('ADR(%s),%s,NIL);',VarName,IndVar)
      ELSE
         WrF2('ADR(%s),ADR(%s),NIL);',VarName,IndVar);
      END;
    WrStr(OutFile,TheStr);
    WrLn(OutFile);
END GenSetV;


PROCEDURE GenSetVArray(InputVars : BOOLEAN);
VAR V : CARDINAL;
    IndTyp,IndLen : CARDINAL;
    Idx : CARDINAL;

(* If an input or output SQLDA was allocated,
   Associate each of the variables used to that SQLDA
   using the SQLASETV procedure

*)


BEGIN
  Idx := 0;
  INC(sqldaID);   (* make unique ID for this sql statement *)
  IF InputVars
    THEN InputSQLDA := sqldaID;
    ELSE OutputSQLDA := sqldaID;
  END;
  WrF3('  RC := SQLPREP.SQLAALOC(%u,%u,%u,NIL);',sqldaID,val,sqldaID);
  WrStr(OutFile,TheStr); (* results from formatio in thestr*)
  WrLn(OutFile);


(* one of the optimizations that can be done*)
(* is to avoid redundent initialization of varaibles *)
(* this optimization is only used if the optimize option was selected *)
  IF Optimize
    THEN
       WrStr(OutFile,'  IF (SQLca.sqlcode =0)(* new sqlda?*) ');
       WrLn(OutFile);
       WrStr(OutFile,'  THEN  (* init vars *)');
       WrLn(OutFile);
    END;




  FOR V := 1 TO CARDINAL(TokenArray.Header.used) DO  (* for each token *)
  GetVarName(VarName,typ,len,TokenArray.Tokens[V].tokenID);
  CASE InputVars OF
    TRUE :
          CASE CARDINAL(TokenArray.Tokens[V].use ) OF
              SQLPREP.SQLA_INPUT_HVAR : Copy(IndVar,'NIL');
                                            GenSetV(V,Idx);
                                            INC(Idx);
             |SQLPREP.SQLA_INPUT_WITH_IND:  GetVarName(IndVar,IndTyp,IndLen,TokenArray.Tokens[V+1].tokenID);
                                            INC(typ);  (* make type to include nul indicator*)
                                            GenSetV(V,Idx);
                                            INC(Idx);
             |SQLPREP.SQLA_INVALID_ID     : WrF1(' Invalid hostvariable ID - %u',
                                                  TokenArray.Tokens[V].tokenID);
                                            WrStr(MsgFile,TheStr);
                                            WrLn(MsgFile);
             |SQLPREP.SQLA_INVALID_USE    : WrF1(' Invalid use for token - %s ',VarName);
                                            WrStr(MsgFile,TheStr);
                                            WrLn(MsgFile);
             |SQLPREP.SQLA_USER_SQLDA     :
             |SQLPREP.SQLA_INDICATOR      :
          END; (*case use of *)

   |FALSE :
             CASE CARDINAL(TokenArray.Tokens[V].use ) OF
              SQLPREP.SQLA_OUTPUT_HVAR    : Copy(IndVar,'NIL');
                                            GenSetV(V,Idx);
                                            INC(Idx);
             |SQLPREP.SQLA_OUTPUT_WITH_IND: GetVarName(IndVar,IndTyp,IndLen,TokenArray.Tokens[V+1].tokenID);
                                            INC(typ);  (* make type to include nul indicator*)
                                            GenSetV(V,Idx);
                                            INC(Idx);
             |SQLPREP.SQLA_INVALID_ID     : WrF1(' Invalid hostvariable ID - %u',
                                                  TokenArray.Tokens[V].tokenID);
                                            WrStr(MsgFile,TheStr);
                                            WrLn(MsgFile);
             |SQLPREP.SQLA_INVALID_USE    : WrF1(' Invalid use for token - %s ',VarName);
                                            WrStr(MsgFile,TheStr);
                                            WrLn(MsgFile);
             |SQLPREP.SQLA_USER_SQLDA     :
             |SQLPREP.SQLA_INDICATOR      :
          END;



          END;


    END; (* end for *)
    IF Optimize
        THEN
          WrStr(OutFile,' END;  (* end if new sqlda *) ');
        END;
    WrLn(OutFile);
END GenSetVArray;

BEGIN
  InputSQLDA := 0;
  OutputSQLDA := 0;
  FOR TaskNbr := 1 TO CARDINAL(TaskArray.Header.used) DO
    WITH TaskArray.tasks[TaskNbr] DO
      val := CARDINAL(value);
      CASE  CARDINAL(func) OF

          SQLA_START         : WrStr(OutFile,'  RC := SQLPREP.SQLASTRT(ADR(xxProgName),NIL,SQLca); (* get semaphore*)');
                               WrLn(OutFile);
         |SQLA_DECLARE       : IF (val = SQLPREP.SQLA_BEGIN)
                                  THEN InDeclare := TRUE
                                  ELSE InDeclare := FALSE;
                               END;
         |SQLA_INCLUDE       : (* this option will include the needed vars, *)
                               (* const, and structures into the source code *)
                               WrStr(OutFile,'(***** included by precompiler *)');
                               WrLn(OutFile);
                               WrStr(OutFile,'FROM SQLDA IMPORT sqlda,sqldaPtr;');
                               WrLn(OutFile);
                               WrStr(OutFile,'FROM SQLCA IMPORT sqlca;');
                               WrLn(OutFile);
                               WrStr(OutFile,'FROM SQLStuff IMPORT AllocSQLDA, DeallocSQLDA,SQLErrorHandler;');
                               WrLn(OutFile);

                               WrStr(OutFile,'CONST');
                               WrLn(OutFile);
                               WrStr(OutFile,"xxProgName = ");
                               PID := ADR(ProgramRec);
                               K := 0;
                               FOR J := 1 TO 39 DO
                                 WrF1('CHR(%u) +',ORD(PID^[J]));
                                 WrStr(OutFile,TheStr);
                                 INC(K);
                                 IF (K MOD 8) = 0
                                   THEN
                                     WrLn(OutFile);
                                     WrStr(OutFile,'        ');
                                   END;
                               END;
                               WrF1('CHR(%u);',ORD(PID^[40]));
                               WrStr(OutFile,TheStr);
                               WrLn(OutFile);
                               WrStr(OutFile,'VAR ');
                               WrLn(OutFile);
                               WrStr(OutFile,'  SQLca : sqlca; (* communications area*)');
                               WrLn(OutFile);
                               WrStr(OutFile,'  SQLda : sqldaPtr; (* data area (if needed ) *)');
                               WrLn(OutFile);
                               WrStr(OutFile,'  RC    : CARDINAL;');
                               WrLn(OutFile);




         |SQLA_ALLOC_INPUT   : GenSetVArray(TRUE);

         |SQLA_ALLOC_OUTPUT  : GenSetVArray(FALSE);

         |SQLA_SETS          :GetVarName(VarName,typ,len,value);
                              WrF2('RC := SQLPREP.SQLASETS(Length(%s),%s,NIL);',VarName,VarName);
                              WrStr(OutFile,TheStr);
                              WrLn(OutFile);

         |SQLA_USDA_INPUT    :INC(sqldaID);   (* make unique ID for this sql statement *)
                              InputSQLDA := sqldaID;
                              GetVarName(VarName,typ,len,value);
                              WrF2('RC := SQLPREP.SQLAUSDA(%u,%s,NIL);',sqldaID,VarName);
                              WrStr(OutFile,TheStr);
                              WrLn(OutFile);
         |SQLA_USDA_OUTPUT   :INC(sqldaID);   (* make unique ID for this sql statement *)
                              OutputSQLDA := sqldaID;
                              GetVarName(VarName,typ,len,value);
                              WrF2('RC := SQLPREP.SQLAUSDA(%u,%s,NIL);',sqldaID,VarName);
                              WrStr(OutFile,TheStr);
                              WrLn(OutFile);
         |SQLA_CALL          :
                              WrF4('  RC := SQLPREP.SQLACALL(%u,%u,%u,%u,NIL); ',
                                    val,SectionNbr,InputSQLDA,OutputSQLDA);
                              WrStr(OutFile,TheStr);
                              WrLn(OutFile);
         |SQLA_DEALLOC       :
         |SQLA_STOP          : WrStr(OutFile,'  RC := SQLPREP.SQLASTOP(NIL);(* release semephore*)');
                               WrLn(OutFile);

         |SQLA_SQLERROR      : WrStr(OutFile,' IF SQLca.sqlcode < 0');
                               WrLn(OutFile);
                               WrStr(OutFile,'   THEN SQLErrorHandler(0,"DATA BASE ERROR",SQLca);');
                               WrLn(OutFile);
                               WrStr(OutFile,'   END;');
                               WrLn(OutFile);

         |SQLA_SQLWARNING    : WrStr(OutFile, 'IF ((SQLca.sqlcode > 0) AND'   );
                               WrLn(OutFile);
                               WrStr(OutFile, '    (SQLca.sqlcode <> 100)  OR');
                               WrLn(OutFile);
                               WrStr(OutFile, '   ((SQLca.sqlcode = 0  AND'   );
                               WrLn(OutFile);
                               WrStr(OutFile, "    (SQLca.sqlwarn[0] = 'W')"  );
                               WrLn(OutFile);
                               WrStr(OutFile,'   THEN');
                               WrLn(OutFile);
                               WrStr(OutFile,'     SQLErrorHandler(0,"DATA BASE ERROR ",SQLca);');
                               WrLn(OutFile);
                               WrStr(OutFile,'   END;');
                               WrLn(OutFile);
         |SQLA_NOT_FOUND     : WrStr(OutFile,'  UNTIL SQLca.sqlcode =100);'); (* not normally used *)
                               WrLn(OutFile);  (* the not found indicates end of loop condition *)
      END; (* end case of *)
    END;
 END;

END GenCode;

PROCEDURE CompileSQL(Stm : ARRAY OF CHAR);
VAR
  L,V : CARDINAL;
  LineNbr : CARDINAL;

  StmtType : CARDINAL;
  buf1,buf2,buf3 : SQLPREP.buf128;

  RC : CARDINAL;
  VarName : ARRAY[0..30] OF CHAR;
  typ    : CARDINAL;
  len     : CARDINAL;

BEGIN
  L := Length(Stm);
  LineNbr := J;
  RC := SQLPREP.SQLGCMPL(L,Stm,LineNbr,TaskArray,TokenArray,SectionNbr,StmtType,
                         buf1,buf2,buf3,NIL,CA);
  IF CA.sqlcode < 0
    THEN
      WrF1('* * * SQL ERROR \n Statement %s',Stm);
      WrStr(MsgFile,TheStr);
      WrLn(MsgFile);
      SQLErrorHandler(MsgFile,'Statement',CA);
      SQLErrorHandler(OutFile,'Statement',CA);  (* write the error inline *)

      WrLn(MsgFile);
      WrStr(MsgFile,'Variables used ');
      WrLn(MsgFile);
(********** Diagnotic help in the event of compile error *)
      FOR V := 1 TO CARDINAL(TokenArray.Header.used) DO  (* for each token *)
        GetVarName(VarName,typ,len,TokenArray.Tokens[V].tokenID);
        WrF3('  %15s   %u   %u',VarName,typ,len);
        WrStr(MsgFile,TheStr);
        WrLn(MsgFile);
      END;
		WrLn(MsgFile);
		WrLn(MsgFile);
    END;


END CompileSQL;

PROCEDURE SQLImport(TheStr : ARRAY OF CHAR) : BOOLEAN;
BEGIN
  Caps(TheStr);
  RETURN Pos(TheStr,'SQLIMPORT') < MAX(CARDINAL);


END SQLImport;


PROCEDURE Import(IncludeStr : ARRAY OF CHAR);
(* include a def file to get additional host variables *)
(* if in the modfile an (* SQLIMPORT modulename *) no extention *)
(*  s encountered, *)
(*   Then the def file (only) is included to pick up the additional *)
(*   host variables that could be defined in other modules *)
TYPE
 CharSet = SET OF CHAR;

CONST
  Alpha = CharSet{'A'..'Z'};

VAR
  IncludeList : GenList;
  J,K : CARDINAL;
  FileName : ARRAY[0..40] OF CHAR;
  TestStr : ARRAY[0..500] OF CHAR;
  DeclareSec : BOOLEAN;
  InComment : CARDINAL;

BEGIN
    DeclareSec := InDeclare;
    InComment := 0;
    Subst(IncludeStr,'(*','');  (* get rid of comment part *)
    Subst(IncludeStr,'*)','');
    Subst(IncludeStr,'SQLIMPORT','');
    Caps(IncludeStr);
    Fill(ADR(FileName),SIZE(FileName),0);
    K := 0;
    FOR J := 0 TO Length(IncludeStr) DO
      IF IncludeStr[J] IN Alpha
         THEN
          FileName[K] := IncludeStr[J];
          INC(K);
         END;
    END;
    FileName[8] := 0C;   (* the import name may be longer than 8 char *)
    Append(FileName,'.SQD');
    IncludeList.FileToList(FileName);
    FOR J := 1 TO IncludeList.ListLength() DO
      IncludeList.GetItem(TheString,J);
      Copy(TestStr,TheString.TheStr);
      RemoveComment(TestStr,InComment);
      GetVars(TestStr,InComment); (* get host variables *)
    END;
  IncludeList.DisposeList();
END Import;

BEGIN

 Process := DefFile;

 REPEAT
  IF Process = DefFile
     THEN Initialize();   (* get set up the envirionment and get the def file *)
     ELSE InitializeMod();
  END;

  FOR J := 1 TO TheList.ListLength() DO
    TheList.GetItem(TheString,J);
    Copy(TestStr,TheString.TheStr);
    IF SQLImport(TestStr)
      THEN
        Import(TestStr);
      END;
    RemoveComment(TestStr,InComment);
    IF SQLStatement(TestStr)
      THEN
        WrStr(OutFile,'(****');
        WrLn(OutFile);

        Fill(ADR(Statement),SIZE(Statement),0);
        Copy(Statement,TheString.TheStr);
        WrStr(OutFile,TheString.TheStr);  (* copy original statement to file*)
        WrLn(OutFile);
        IF (Pos(Statement,';') = MAX(CARDINAL))  (* not end of statement ; wasn't there*)
          THEN
            REPEAT
              INC(J);
              TheList.GetItem(TheString,J);
              Append(Statement,TheString.TheStr);
              WrStr(OutFile,TheString.TheStr);  (* copy original statement to file*)
              WrLn(OutFile);
            UNTIL (Pos(TheString.TheStr,';') < MAX(CARDINAL))
         END;  (* end if  end of statement *)

        WrStr(OutFile,'***)');
        WrLn(OutFile);

          (* parse the sql command, compile and gen the code *)
        PrepareSQL(Statement,TokenArray);
        CompileSQL(Statement);
        GenCode();


      ELSE  (* this is not an sql statement - write normal *)
        WrStr(OutFile,TheString.TheStr);
        WrLn(OutFile);
        IF InDeclare
          THEN

             GetVars(TestStr,InComment); (* get host variables *)
          END;
      END;
  END;
  Close(OutFile);
 UNTIL Process = Done;
  J := SQLPREP.SQLA_SAVE;
  SQLPREP.SQLGFINI(J,NIL,CA);
  WrStr(MsgFile,'END of precompiler ');
END PreComp.





