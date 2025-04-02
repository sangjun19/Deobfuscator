static char sccsid[]="@(#)13	1.1  src/agrep/main.c, aixlike.src, aixlike3  9/27/95  15:43:25";
/* Copyright (c) 1991 Sun Wu and Udi Manber.  All Rights Reserved.      */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <io.h>                      /* 21/09/94                     @DT*/
#include <fcntl.h>                   /* 21/09/94                     @DT*/
#include <sys\stat.h>                /* 21/09/94                     @DT*/
#include <errno.h>                   /* 21/09/94                     @DT*/
#include <os2.h>                                   /* 22/09/94 @DT */
#include "agrep.h"
#include "chkfile.h"

/************************************************************************/
/*                                                                      */
/* NOTES:                                                               */
/*  Ported to OS/2: 21/09/94 D.Twyerould                                */
/*   - Added options to help text                                       */
/*   - Update low level file IO routines to call CSET/2 _xxx routines via*/
/*     Macros.                                                          */
/*   - Added O_RDONLY to 2nd parm of open() instead of 0                */
/*                                                                      */
/*  Updated for searching file subdirectories and wildcards & changed flags*/
/*   - changed -s to -q (-s is search for subdirs, -q is silent mode)   */
/*   - changed -c to -i and vice versa (only because -c means case to me)*/
/*   - changed -c to mean Make case sensistive (ie, default is noncasesens)*/
/*   - added   -b to mean best match, no prompting                      */
/*   - eliminated GOTOs in best match prompting                         */
/************************************************************************/

#define  PVERSION       "2"
#define  PSUBVERSION    "04"
#define  PATHSIZE       512

typedef struct _STFILELIST
  {
  PSZ   pszFileName;
  ULONG ulFileSize;
  ULONG ulFileDate;
  struct _STFILELIST *pNext;
  } STFILELIST, *PSTFILELIST;

unsigned Mask[MAXSYM];
unsigned Init1,NO_ERR_MASK,Init[MaxError];
unsigned Bit[WORD+1];
CHAR   buffer[BlockSize+Maxline+1];
unsigned Next[MaxNext],Next1[MaxNext];
unsigned wildmask,endposition,D_endpos;
int    REGEX,RE_ERR,FNAME,WHOLELINE,SIMPLEPATTERN;
int    COUNT,HEAD,TAIL,LINENUM,INVERSE,I,S,DD,AND,SGREP,JUMP;
int    Num_Pat,PSIZE,num_of_matched,SILENT,NOPROMPT,BESTMATCH,NOUPPER, SUBDIR;
int    NOMATCH,TRUNCATE,FIRST_IN_RE,FIRSTOUTPUT;
int    WORDBOUND,DELIMITER,D_length;
int    EATFIRST,OUTTAIL;
int    FILEOUT;
int    DNA = 0;
int    APPROX = 0;
int    PAT_FILE = 0;
int    CONSTANT = 0;
int    total_line = 0;               /* used in mgrep                   */
CHAR   **Textfiles;                  /* array of filenames to be        */
                                     /* searched                        */
CHAR   old_D_pat[MaxDelimit] = "\n"; /* to hold original D_pattern      */
CHAR   CurrentFileName[MAXNAME];
CHAR   Progname[MAXNAME];
CHAR   D_pattern[MaxDelimit] = "\n; ";/* string which delimits records  */
                                     /* -- defaults to newline          */
int    NOFILENAME = 0,               /* Boolean flag, set for -h option */
FILENAMEONLY = 0,                    /* Boolean flag, set for -l option */
Numfiles = 0;                        /* indicates how many files in     */
                                     /* Textfiles                       */
extern int init();
int    table[WORD][WORD];

/********************************************************************/
/* FUNCTION:    initial_value                                       */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
initial_value()
  {
  int    i;

  JUMP = REGEX = FNAME = BESTMATCH = NOPROMPT = SUBDIR = 0;/* 21/09/94  */
                                     /*                     @DTADDSUBDIR*/
  COUNT = LINENUM = WHOLELINE = SGREP = 0;
  EATFIRST = INVERSE = AND = TRUNCATE = OUTTAIL = 0;
  NOUPPER = 1;                       /* 21/09/94               @DT-WAS=0*/
  FIRST_IN_RE = NOMATCH = FIRSTOUTPUT = ON;
  I = DD = S = 1;
  HEAD = TAIL = ON;
  D_length = 2;
  SILENT = Num_Pat = PSIZE = SIMPLEPATTERN = num_of_matched = 0;
  WORDBOUND = DELIMITER = RE_ERR = 0;
  Bit[WORD] = 1;
  for (i = WORD-1; i > 0; i--)
    Bit[i] = Bit[i+1]<<1;
  for (i = 0; i < MAXSYM; i++)
    Mask[i] = 0;
}

/********************************************************************/
/* FUNCTION:    compute                                             */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
compute_next(M,Next,Next1)
  int    M;
  unsigned *Next,*Next1;
{
  int    i,j = 0,n,k,temp;
  int    mid,pp;
  int    MM,base;
  unsigned V[WORD];

  base = WORD-M;
  temp = Bit[base];
  Bit[base] = 0;
  for (i = 0; i < WORD; i++)
    V[i] = 0;
  for (i = 1; i < M; i++)
    {
    j = 0;
    while (table[i][j] > 0 && j < 10)
      {
      V[i] = V[i]|Bit[base+table[i][j++]];
      }
    }
  Bit[base] = temp;
  if (M <= SHORTREG)
    {
    k = exponen(M);
    pp = 2*k;
    for (i = k; i < pp; i++)
      {
      n = i;
      Next[i] = (k>>1);
      for (j = M; j >= 1; j--)
        {
        if (n&Bit[WORD])
          Next[i] = Next[i]|V[j];
        n = (n>>1);
        }
      }
    return ;
    }
  if (M > MAXREG)
    fprintf(stderr, "%s: regular expression too long\n", Progname);
  MM = M;
  if (M&1)
    M = M+1;
  k = exponen(M/2);
  pp = 2*k;
  mid = MM/2;
  for (i = k; i < pp; i++)
    {
    n = i;
    Next[i] = (Bit[base]>>1);
    for (j = MM; j > mid; j--)
      {
      if (n&Bit[WORD])
        Next[i] = Next[i]|V[j-mid];
      n = (n>>1);
      }
    n = i-k;
    Next1[i-k] = 0;
    for (j = 0; j < mid; j++)
      {
      if (n&Bit[WORD])
        Next1[i-k] = Next1[i-k]|V[MM-j];
      n = (n>>1);
      }
    }
  return ;
}

/********************************************************************/
/* FUNCTION:    exponen                                             */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
exponen(m)
  int    m;
{

  int i,ex;
  ex = 1;
  for (i = 0; i < m; i++)
    ex = ex *2;
  return (ex);
}

/********************************************************************/
/* FUNCTION:    re1                                                 */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
re1(Text,M,D)
  int    Text,M,D;
{
  register unsigned i,c,r0,r1,r2,r3,CMask,Newline,Init0,r_NO_ERR;
  register unsigned end;
  register unsigned hh,LL = 0,k;     /* Lower part                      */
  int    FIRST_TIME = ON,num_read,j = 0,base;
  unsigned A[MaxRerror+1],B[MaxRerror+1];
  unsigned Next[MaxNext],Next1[MaxNext];
  CHAR   buffer[BlockSize+Maxline+1];
  int    FIRST_LOOP = 1;

  r_NO_ERR = NO_ERR_MASK;
  if (M > 30)
    {
    fprintf(stderr, "%s: regular expression too long\n", Progname);
    exit(2);
    }
  base = WORD-M;
  hh = M/2;
  for (i = WORD, j = 0; j < hh; i--, j++)
    LL = LL|Bit[i];
  if (FIRST_IN_RE)
    compute_next(M, Next, Next1);

          /**************************************************************/
          /* SUN: try: change to memory allocation                      */
          /**************************************************************/

  FIRST_IN_RE = 0;
  Newline = '\n';
  Init[0] = Bit[base];
  if (HEAD)
    Init[0] = Init[0]|Bit[base+1];
  for (i = 1; i <= D; i++)
    Init[i] = Init[i-1]|Next[Init[i-1]>>hh]|Next1[Init[i-1]&LL];
  Init1 = Init[0]|1;
  Init0 = Init[0];
  r2 = r3 = Init[0];
  for (k = 0; k <= D; k++)
    {
    A[k] = B[k] = Init[k];
    }
  if (D == 0)
    {
    while ((num_read = read(Text, buffer+Maxline, BlockSize)) > 0)
      {
      i = Maxline;
      end = num_read+Maxline;
      if ((num_read < BlockSize) && buffer[end-1] != '\n')
        buffer[end] = '\n';
      if (FIRST_LOOP)
        {                            /* if first time in the loop add a */
                                     /* newline                         */
        buffer[i-1] = '\n';          /* in front the text.              */
        i--;
        FIRST_LOOP = 0;
        }
      while (i < end)
        {
        c = buffer[i++];
        CMask = Mask[c];
        if (c != Newline)
          {
          if (CMask != 0)
            {
            r1 = Init1&r3;
            r2 = ((Next[r3>>hh]|Next1[r3&LL])&CMask)|r1;
            }
          else
            {
            r2 = r3&Init1;
            }
          }
        else
          {
          j++;
          r1 = Init1&r3;             /* match against endofline         */
          r2 = ((Next[r3>>hh]|Next1[r3&LL])&CMask)|r1;
          if (TAIL)
            r2 = (Next[r2>>hh]|Next1[r2&LL])|r2;/* epsilon move         */
          if ((r2&1)^INVERSE)
            {
            if (FILENAMEONLY)
              {
              num_of_matched++;
              printf("%s\n", CurrentFileName);
              return ;
              }
            r_output(buffer, i-1, end, j);
            }
          r3 = Init0;
          r2 = (Next[r3>>hh]|Next1[r3&LL])&CMask|Init0;/* match begin of*/
                                     /* line                            */
          }
        c = buffer[i++];
        CMask = Mask[c];
        if (c != Newline)
          {
          if (CMask != 0)
            {
            r1 = Init1&r2;
            r3 = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|r1;
            }
          else
            r3 = r2&Init1;
          }                          /* if(NOT Newline)                 */
        else

          {
          j++;
          r1 = Init1&r2;             /* match against endofline         */
          r3 = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|r1;
          if (TAIL)
            r3 = (Next[r3>>hh]|Next1[r3&LL])|r3;/* epsilon move         */
          if ((r3&1)^INVERSE)
            {
            if (FILENAMEONLY)
              {
              num_of_matched++;
              printf("%s\n", CurrentFileName);
              return ;
              }
            r_output(buffer, i-1, end, j);
            }
          r2 = Init0;
          r3 = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|Init0;

          /**************************************************************/
          /* match begin of line                                        */
          /**************************************************************/

          }
        }                            /* while i < end ...               */
      strncpy(buffer, buffer+num_read, Maxline);
      }                              /* end while read()...             */
    return ;
    }                                /* end if (D == 0)                 */
  while ((num_read = read(Text, buffer+Maxline, BlockSize)) > 0)
    {
    i = Maxline;
    end = Maxline+num_read;
    if ((num_read < BlockSize) && buffer[end-1] != '\n')
      buffer[end] = '\n';
    if (FIRST_TIME)
      {                              /* if first time in the loop add a */
                                     /* newline                         */
      buffer[i-1] = '\n';            /* in front the text.              */
      i--;
      FIRST_TIME = 0;
      }
    while (i < end)
      {
      c = buffer[i];
      CMask = Mask[c];
      if (c != Newline)
        {
        if (CMask != 0)
          {
          r2 = B[0];
          r1 = Init1&r2;
          A[0] = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|r1;
          r3 = B[1];
          r1 = Init1&r3;
          r0 = r2|A[0];              /* A[0] | B[0]                     */
          A[1] = ((Next[r3>>hh]|Next1[r3&LL])&CMask)|((r2|Next[r0>>hh]|
             Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 1)
            goto Nextchar;
          r2 = B[2];
          r1 = Init1&r2;
          r0 = r3|A[1];
          A[2] = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|((r3|Next[r0>>hh]|
             Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 2)
            goto Nextchar;
          r3 = B[3];
          r1 = Init1&r3;
          r0 = r2|A[2];
          A[3] = ((Next[r3>>hh]|Next1[r3&LL])&CMask)|((r2|Next[r0>>hh]|
             Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 3)
            goto Nextchar;
          r2 = B[4];
          r1 = Init1&r2;
          r0 = r3|A[3];
          A[4] = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|((r3|Next[r0>>hh]|
             Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 4)
            goto Nextchar;
          }                          /* if(CMask)                       */
        else
          {
          r2 = B[0];
          A[0] = r2&Init1;
          r3 = B[1];
          r1 = Init1&r3;
          r0 = r2|A[0];
          A[1] = ((r2|Next[r0>>hh]|Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 1)
            goto Nextchar;
          r2 = B[2];
          r1 = Init1&r2;
          r0 = r3|A[1];
          A[2] = ((r3|Next[r0>>hh]|Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 2)
            goto Nextchar;
          r3 = B[3];
          r1 = Init1&r3;
          r0 = r2|A[2];
          A[3] = ((r2|Next[r0>>hh]|Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 3)
            goto Nextchar;
          r2 = B[4];
          r1 = Init1&r2;
          r0 = r3|A[3];
          A[4] = ((r3|Next[r0>>hh]|Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 4)
            goto Nextchar;
          }
        }
      else
        {
        j++;
        r1 = Init1&B[D];             /* match against endofline         */
        A[D] = ((Next[B[D]>>hh]|Next1[B[D]&LL])&CMask)|r1;
        if (TAIL)
          A[D] = (Next[A[D]>>hh]|Next1[A[D]&LL])|A[D];/* epsilon move   */
        if ((A[D]&1)^INVERSE)
          {
          if (FILENAMEONLY)
            {
            num_of_matched++;
            printf("%s\n", CurrentFileName);
            return ;
            }
          r_output(buffer, i, end, j);
          }
        for (k = 0; k <= D; k++)
          B[k] = Init[0];
        r1 = Init1&B[0];
        A[0] = ((Next[B[0]>>hh]|Next1[B[0]&LL])&CMask)|r1;
        for (k = 1; k <= D; k++)
          {
          r3 = B[k];
          r1 = Init1&r3;
          r2 = A[k-1]|B[k-1];
          A[k] = ((Next[r3>>hh]|Next1[r3&LL])&CMask)|((B[k-1]|Next[r2>>
             hh]|Next1[r2&LL])&r_NO_ERR)|r1;
          }
        }
Nextchar:
      i = i+1;
      c = buffer[i];
      CMask = Mask[c];
      if (c != Newline)
        {
        if (CMask != 0)
          {
          r2 = A[0];
          r1 = Init1&r2;
          B[0] = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|r1;
          r3 = A[1];
          r1 = Init1&r3;
          r0 = B[0]|r2;
          B[1] = ((Next[r3>>hh]|Next1[r3&LL])&CMask)|((r2|Next[r0>>hh]|
             Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 1)
            goto Nextchar1;
          r2 = A[2];
          r1 = Init1&r2;
          r0 = B[1]|r3;
          B[2] = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|((r3|Next[r0>>hh]|
             Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 2)
            goto Nextchar1;
          r3 = A[3];
          r1 = Init1&r3;
          r0 = B[2]|r2;
          B[3] = ((Next[r3>>hh]|Next1[r3&LL])&CMask)|((r2|Next[r0>>hh]|
             Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 3)
            goto Nextchar1;
          r2 = A[4];
          r1 = Init1&r2;
          r0 = B[3]|r3;
          B[4] = ((Next[r2>>hh]|Next1[r2&LL])&CMask)|((r3|Next[r0>>hh]|
             Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 4)
            goto Nextchar1;
          }                          /* if(CMask)                       */
        else
          {
          r2 = A[0];
          B[0] = r2&Init1;
          r3 = A[1];
          r1 = Init1&r3;
          r0 = B[0]|r2;
          B[1] = ((r2|Next[r0>>hh]|Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 1)
            goto Nextchar1;
          r2 = A[2];
          r1 = Init1&r2;
          r0 = B[1]|r3;
          B[2] = ((r3|Next[r0>>hh]|Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 2)
            goto Nextchar1;
          r3 = A[3];
          r1 = Init1&r3;
          r0 = B[2]|r2;
          B[3] = ((r2|Next[r0>>hh]|Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 3)
            goto Nextchar1;
          r2 = A[4];
          r1 = Init1&r2;
          r0 = B[3]|r3;
          B[4] = ((r3|Next[r0>>hh]|Next1[r0&LL])&r_NO_ERR)|r1;
          if (D == 4)
            goto Nextchar1;
          }
        }                            /* if(NOT Newline)                 */
      else
        {
        j++;
        r1 = Init1&A[D];             /* match against endofline         */
        B[D] = ((Next[A[D]>>hh]|Next1[A[D]&LL])&CMask)|r1;
        if (TAIL)
          B[D] = (Next[B[D]>>hh]|Next1[B[D]&LL])|B[D];/* epsilon move   */
        if ((B[D]&1)^INVERSE)
          {
          if (FILENAMEONLY)
            {
            num_of_matched++;
            printf("%s\n", CurrentFileName);
            return ;
            }
          r_output(buffer, i, end, j);
          }
        for (k = 0; k <= D; k++)
          A[k] = Init0;
        r1 = Init1&A[0];
        B[0] = ((Next[A[0]>>hh]|Next1[A[0]&LL])&CMask)|r1;
        for (k = 1; k <= D; k++)
          {
          r3 = A[k];
          r1 = Init1&r3;
          r2 = A[k-1]|B[k-1];
          B[k] = ((Next[r3>>hh]|Next1[r3&LL])&CMask)|((A[k-1]|Next[r2>>
             hh]|Next1[r2&LL])&r_NO_ERR)|r1;
          }
        }
Nextchar1:
      i = i+1;
      }                              /* while                           */
    strncpy(buffer, buffer+num_read, Maxline);
    }                                /* while                           */
  return ;
}                                    /* re1                             */

/********************************************************************/
/* FUNCTION:    re                                                  */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
re(Text,M,D)
  int    Text,M,D;
{
  register unsigned i,c,r1,r2,r3,CMask,k,Newline,Init0,Init1,end;
  register unsigned r_even,r_odd,r_NO_ERR;
  unsigned RMask[MAXSYM];
  unsigned A[MaxRerror+1],B[MaxRerror+1];
  int    num_read,j = 0,bp,lasti,base,ResidueSize;
  int    FIRST_TIME;                 /* Flag                            */

  base = WORD-M;
  k = 2*exponen(M);
  if (FIRST_IN_RE)
    {
    compute_next(M, Next, Next1);
    FIRST_IN_RE = 0;
    }
  for (i = 0; i < MAXSYM; i++)
    RMask[i] = Mask[i];
  r_NO_ERR = NO_ERR_MASK;
  Newline = '\n';
  lasti = Maxline;
  Init0 = Init[0] = Bit[base];
  if (HEAD)
    Init0 = Init[0] = Init0|Bit[base+1];
  for (i = 1; i <= D; i++)
    Init[i] = Init[i-1]|Next[Init[i-1]];/* can be out?                  */
  Init1 = Init0|1;
  r2 = r3 = Init0;
  for (k = 0; k <= D; k++)
    {
    A[k] = B[k] = Init[0];
    }                                /* can be out?                     */
  FIRST_TIME = ON;
  if (D == 0)
    {
    while ((num_read = read(Text, buffer+Maxline, BlockSize)) > 0)
      {
      i = Maxline;
      end = Maxline+num_read;
      if ((num_read < BlockSize) && buffer[end-1] != '\n')
        buffer[end] = '\n';
      if (FIRST_TIME)
        {
        buffer[i-1] = '\n';
        i--;
        FIRST_TIME = 0;
        }
      while (i < end)
        {
        c = buffer[i++];
        CMask = RMask[c];
        if (c != Newline)
          {
          r1 = Init1&r3;
          r2 = (Next[r3]&CMask)|r1;
          }
        else
          {
          r1 = Init1&r3;             /* match against '\n'              */
          r2 = Next[r3]&CMask|r1;
          j++;
          if (TAIL)
            r2 = Next[r2]|r2;        /* epsilon move                    */
          if ((r2&1)^INVERSE)
            {
            if (FILENAMEONLY)
              {
              num_of_matched++;
              printf("%s\n", CurrentFileName);
              return ;
              }
            r_output(buffer, i-1, end, j);
            }
          lasti = i-1;
          r3 = Init0;
          r2 = (Next[r3]&CMask)|Init0;
          }
        c = buffer[i++];
        CMask = RMask[c];
        if (c != Newline)
          {
          r1 = Init1&r2;
          r3 = (Next[r2]&CMask)|r1;
          }
        else
          {
          j++;
          r1 = Init1&r2;             /* match against endofline         */
          r3 = Next[r2]&CMask|r1;
          if (TAIL)
            r3 = Next[r3]|r3;
          if ((r3&1)^INVERSE)
            {
            if (FILENAMEONLY)
              {
              num_of_matched++;
              printf("%s\n", CurrentFileName);
              return ;
              }
            r_output(buffer, i-1, end, j);
            }
          lasti = i-1;
          r2 = Init0;
          r3 = (Next[r2]&CMask)|Init0;/* match the newline              */
          }
        }                            /* while                           */
      ResidueSize = Maxline+num_read-lasti;
      if (ResidueSize > Maxline)
        {
        ResidueSize = Maxline;
        }
      strncpy(buffer+Maxline-ResidueSize, buffer+lasti, ResidueSize);
      lasti = Maxline-ResidueSize;
      }                              /* while                           */
    return ;
    }                                /* end if(D==0)                    */
  while ((num_read = read(Text, buffer+Maxline, BlockSize)) > 0)
    {
    i = Maxline;
    end = Maxline+num_read;
    if ((num_read < BlockSize) && buffer[end-1] != '\n')
      buffer[end] = '\n';
    if (FIRST_TIME)
      {
      buffer[i-1] = '\n';
      i--;
      FIRST_TIME = 0;
      }
    while (i < end)
      {
      c = buffer[i++];
      CMask = RMask[c];
      if (c != Newline)
        {
        r_even = B[0];
        r1 = Init1&r_even;
        A[0] = (Next[r_even]&CMask)|r1;
        r_odd = B[1];
        r1 = Init1&r_odd;
        r2 = (r_even|Next[r_even|A[0]])&r_NO_ERR;
        A[1] = (Next[r_odd]&CMask)|r2|r1;
        if (D == 1)
          goto Nextchar;
        r_even = B[2];
        r1 = Init1&r_even;
        r2 = (r_odd|Next[r_odd|A[1]])&r_NO_ERR;
        A[2] = (Next[r_even]&CMask)|r2|r1;
        if (D == 2)
          goto Nextchar;
        r_odd = B[3];
        r1 = Init1&r_odd;
        r2 = (r_even|Next[r_even|A[2]])&r_NO_ERR;
        A[3] = (Next[r_odd]&CMask)|r2|r1;
        if (D == 3)
          goto Nextchar;
        r_even = B[4];
        r1 = Init1&r_even;
        r2 = (r_odd|Next[r_odd|A[3]])&r_NO_ERR;
        A[4] = (Next[r_even]&CMask)|r2|r1;
        goto   Nextchar;
        }                            /* if NOT Newline                  */
      else

        {
        j++;
        r1 = Init1&B[D];             /* match endofline                 */
        A[D] = (Next[B[D]]&CMask)|r1;
        if (TAIL)
          A[D] = Next[A[D]]|A[D];
        if ((A[D]&1)^INVERSE)
          {
          if (FILENAMEONLY)
            {
            num_of_matched++;
            printf("%s\n", CurrentFileName);
            return ;
            }
          r_output(buffer, i-1, end, j);
          }
        for (k = 0; k <= D; k++)
          {
          A[k] = B[k] = Init[k];
          }
        r1 = Init1&B[0];
        A[0] = (Next[B[0]]&CMask)|r1;
        for (k = 1; k <= D; k++)
          {
          r1 = Init1&B[k];
          r2 = (B[k-1]|Next[A[k-1]|B[k-1]])&r_NO_ERR;
          A[k] = (Next[B[k]]&CMask)|r1|r2;
          }
        }
Nextchar:
      c = buffer[i];
      CMask = RMask[c];
      if (c != Newline)
        {
        r1 = Init1&A[0];
        B[0] = (Next[A[0]]&CMask)|r1;
        r1 = Init1&A[1];
        B[1] = (Next[A[1]]&CMask)|((A[0]|Next[A[0]|B[0]])&r_NO_ERR)|r1;
        if (D == 1)
          goto Nextchar1;
        r1 = Init1&A[2];
        B[2] = (Next[A[2]]&CMask)|((A[1]|Next[A[1]|B[1]])&r_NO_ERR)|r1;
        if (D == 2)
          goto Nextchar1;
        r1 = Init1&A[3];
        B[3] = (Next[A[3]]&CMask)|((A[2]|Next[A[2]|B[2]])&r_NO_ERR)|r1;
        if (D == 3)
          goto Nextchar1;
        r1 = Init1&A[4];
        B[4] = (Next[A[4]]&CMask)|((A[3]|Next[A[3]|B[3]])&r_NO_ERR)|r1;
        goto   Nextchar1;
        }                            /* if(NOT Newline)                 */
      else

        {
        j++;
        r1 = Init1&A[D];             /* match endofline                 */
        B[D] = (Next[A[D]]&CMask)|r1;
        if (TAIL)
          B[D] = Next[B[D]]|B[D];
        if ((B[D]&1)^INVERSE)
          {
          if (FILENAMEONLY)
            {
            num_of_matched++;
            printf("%s\n", CurrentFileName);
            return ;
            }
          r_output(buffer, i, end, j);
          }
        for (k = 0; k <= D; k++)
          {
          A[k] = B[k] = Init[k];
          }
        r1 = Init1&A[0];
        B[0] = (Next[A[0]]&CMask)|r1;
        for (k = 1; k <= D; k++)
          {
          r1 = Init1&A[k];
          r2 = (A[k-1]|Next[A[k-1]|B[k-1]])&r_NO_ERR;
          B[k] = (Next[A[k]]&CMask)|r1|r2;
          }
        }
Nextchar1:
      i++;
      }                              /* while i < end                   */
    strncpy(buffer, buffer+num_read, Maxline);
    }                                /* while read()                    */
  return ;
}                                    /* re                              */

/********************************************************************/
/* FUNCTION:    r                                                   */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
r_output(buffer,i,end,j)
  int    i,end,j;
  CHAR   *buffer;
{
  int    bp;

  if (i >= end)
    return ;
  num_of_matched++;
  if (COUNT)
    return ;
  if (FNAME)
    printf("%s: ", CurrentFileName);
  bp = i-1;
  while ((buffer[bp] != '\n') && (bp > 0))
    bp--;
  if (LINENUM)
    printf("%d: ", j);
  if (buffer[bp] != '\n')
    bp = Maxline-1;
  bp++;
  while (bp <= i)
    putchar(buffer[bp++]);
}

/********************************************************************/
/* FUNCTION:    GetFileNameOnly                                     */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  pszOut                                              */
/*              pszIn                                               */
/*                                                                  */
/* Returns:     None                                                */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
VOID GetFileNameOnly(PSZ pszOut, PSZ pszIn)
  {
  CHAR *pchPtr;

  pchPtr = strrchr(pszIn, '\\');
  if (pchPtr)
    pchPtr++;
  else
    {
    pchPtr = strrchr(pszIn, ':');
    if (pchPtr)
      pchPtr++;
    else
      pchPtr = pszIn;
    }
  strcpy(pszOut, pchPtr);
  }
/********************************************************************/
/* FUNCTION:    GetFilePathOnly                                     */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  pszOut                                              */
/*              pszIn                                               */
/*                                                                  */
/* Returns:     None                                                */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
VOID GetFilePathOnly(PSZ pszOut, PSZ pszIn)
  {
  CHAR   *pchPtr;
  CHAR   szTemp[PATHSIZE+1];

  strcpy(szTemp, pszIn);
  pchPtr = strrchr(szTemp, (INT)'\\');
  if (pchPtr)
    *(++pchPtr) = '\0';
  else
    {
    pchPtr = strrchr(szTemp, (INT)':');
    if (pchPtr)
      *(++pchPtr) = '\0';
    else
      szTemp[0] = 0;
    }
  strcpy(pszOut, szTemp);
  }

/********************************************************************/
/* FUNCTION:    GetFileList                                         */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  pszFileSpec                                         */
/*                                                                  */
/* Returns:     PSTFILELIST                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
PSTFILELIST GetFileList(PSTFILELIST pstLastFile, PSZ pszFileSpec)
  {
  /* call ourself recursively to build up a linked list of files */
  HDIR   hdirFindHandle = 0xFFFFFFFF;    /* Let OS/2 get us the handle      */
  ULONG  ulFileFindCount = 1;
  APIRET rc;
  CHAR   szPath[PATHSIZE + 1];
  CHAR   szNewPath[PATHSIZE + 1];
  CHAR   szFileMask[13];
  static FILEFINDBUF3 stFindBuffer;

  /* Recursively process all files in (sub)directories as required */
  hdirFindHandle = 0xFFFFFFFF;
  /* save the path part of the file name */
  GetFilePathOnly(szPath, pszFileSpec);

  /* iFileFindCount = 1; */
  /* printf("Searching for filespec %s\n", pszFileSpec); */
  rc = DosFindFirst(pszFileSpec, &hdirFindHandle, 0, &stFindBuffer,
                    sizeof(stFindBuffer), (PULONG)&ulFileFindCount, FIL_STANDARD);
  while (!rc) /* Process each file that matches the mask */
    {
    /* update the files found count */
    /* printf("File found: %s\n", stFindBuffer.achName); */

    /* add to the linked list */
    pstLastFile->pNext = (PSTFILELIST)malloc(sizeof(STFILELIST));
    pstLastFile = pstLastFile->pNext;
    pstLastFile->pszFileName = malloc(strlen(szPath)+strlen(stFindBuffer.achName)+2);
    strcpy(pstLastFile->pszFileName, szPath);
    strcat(pstLastFile->pszFileName, stFindBuffer.achName);

    rc = DosFindNext(hdirFindHandle, &stFindBuffer, sizeof(stFindBuffer),
                       (PULONG)&ulFileFindCount);
    }
  pstLastFile->pNext = NULL;

  /* process subdirectories as well if required */
  if (SUBDIR)
    {
    GetFileNameOnly(szFileMask, pszFileSpec);

    strcpy(szNewPath, szPath);
    if ((szNewPath[strlen(szNewPath)-1] == ':') || (!szNewPath[0]) ||
        (szNewPath[strlen(szNewPath)-1] == '\\'))
      strcat(szNewPath, "*.*");
    else 
      strcat(szNewPath, "\\*.*");

    /* printf("searching for subdirectories with %s\n", szNewPath); */
    hdirFindHandle = 0xFFFFFFFF;    
    ulFileFindCount = 1L;
    rc = DosFindFirst(szNewPath,       /* PSZ                           */
         &hdirFindHandle,              /* PHDIR dirhandle                 */
         FILE_DIRECTORY,               /* ULONG Attrib                    */
         &stFindBuffer,                /* PFILEFINDBUF                    */
         (ULONG)sizeof(stFindBuffer), /* ULONG                           */
         (PULONG)&ulFileFindCount,      /* PULONG entries to find          */
         FIL_STANDARD);                /* (ULONG)0x0001 level of file data*/
                                       /* required                        */
    if (rc)
      return pstLastFile;

    while (!rc)
      {
      if (strcmp(stFindBuffer.achName, ".") &&
                 strcmp(stFindBuffer.achName, ".."))
        {
        if (stFindBuffer.attrFile == FILE_DIRECTORY)
          {
          if (szPath[0])
            {
            if (szPath[strlen(szPath)-1] == '\\')
              sprintf(szNewPath, "%s%s\\%s", szPath, stFindBuffer.achName, szFileMask);
            else
              sprintf(szNewPath, "%s\\%s\\%s", szPath, stFindBuffer.achName, szFileMask);
            }
          else
            sprintf(szNewPath, "%s\\%s", stFindBuffer.achName, szFileMask);
          /* strcpy(szNewPath, szPath); */
          /* strcat(szNewPath, stFindBuffer.achName); */
          /* strcat(szNewPath, szFileMask); */
          /* printf("Searching subdirectory %s\n", szNewPath); */
          pstLastFile = GetFileList(pstLastFile, szNewPath);
          }
        }
      /* get the next directory */
      rc = DosFindNext(hdirFindHandle,   /* HDIR                            */
         &stFindBuffer,                /* PFILEFINDBUF                    */
         sizeof(stFindBuffer),         /* ULONG                           */
         (PULONG)&ulFileFindCount);       /* PULONG                          */
      }

    }
  return pstLastFile;
  }

/********************************************************************/
/* FUNCTION:    FreeFileList                                        */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  pstRootFileList                                     */
/*                                                                  */
/* Returns:     None                                                */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
VOID FreeFileList(PSTFILELIST pstRootFileList)
  {
  PSTFILELIST pstFile, pstNewFile;

  pstFile = pstRootFileList;
  while (pstFile)
    {
    free(pstFile->pszFileName);
    pstNewFile = pstFile->pNext;
    free(pstFile);
    pstFile = pstNewFile;
    }
  }
/********************************************************************/
/* FUNCTION:    DisplayFileList                                     */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  pstRootFileList                                     */
/*                                                                  */
/* Returns:     None                                                */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
VOID DisplayFileList(PSTFILELIST pstRootFileList)
  {
  PSTFILELIST pstFile;

  pstFile = pstRootFileList;
  while (pstFile)
    {
    printf("%s\n", pstFile->pszFileName);
    pstFile = pstFile->pNext;
    }
  }
/********************************************************************/
/* FUNCTION:    main                                                */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
main(argc,argv)
  int    argc;
  char *argv[];
{
  int    N,M,D = 0,fp,fd,i,j;
  char   c;
  int    filetype;
  unsigned char Pattern[MAXPAT],OldPattern[MAXPAT],temp[MAXPAT];
  STFILELIST stRootList, *pstFiles;

  initial_value();
  strcpy(Progname, argv[0]);
  if (argc < 2)
    usage();
  Pattern[0] = '\0';
  while (--argc > 0 && (*++argv)[0] == '-')
    {
    c = *(argv[0]+1);
    switch (c)
      {
      case 'i' :
        COUNT = ON;                  /* output the # of matched         */
        break;
      case 's' :
        SUBDIR = ON;
        break;
      case 'b' :
        BESTMATCH = ON;              /* 21/09/94                     @DT*/
        NOPROMPT = ON;
        break;
      case 'q' :
        SILENT = ON;                 /* silent mode                     */
        break;
      case 'p' :
        I = 0;                       /* insertion cost is 0             */
        break;
      case 'x' :
        WHOLELINE = ON;              /* match the whole line            */
        if (WORDBOUND)
          {
          fprintf(stderr, "%s: illegal option combination\n", Progname);
          exit(2);
          }
        break;
      case 'L' :
        break;
      case 'd' :
        DELIMITER = ON;              /* user defines delimiter          */
        if (argc <= 1)
          usage();
        if (argv[0][2] == '\0')
          {                          /* space after -d option           */
          argv++;
          if ((D_length = strlen(argv[0])) > MaxDelimit)
            {
            fprintf(stderr, "%s: delimiter pattern too long\n", Progname
               );
            exit(2);
            }
          D_pattern[0] = '<';
          strcpy(D_pattern+1, argv[0]);
          argc--;
          }
        else
          {
          if ((D_length = strlen(argv[0]+2)) > MaxDelimit)
            {
            fprintf(stderr, "%s: delimiter pattern too long\n", Progname
               );
            exit(2);
            }
          D_pattern[0] = '<';
          strcpy(D_pattern+1, argv[0]+2);
          }                          /* else                            */
        strcat(D_pattern, ">; ");
        D_length++;                  /* to count ';' as one             */
        break;
      case 'e' :
        argc--;
        if (argc == 0)
          {
          fprintf(stderr,
             "%s: the pattern should immediately follow the -e option\n"
             , Progname);
          usage();
          }
        if ((++argv)[0][0] == '-')
          {
          Pattern[0] = '\\';
          strcat(Pattern, (argv)[0]);
          }
        else
          strcat(Pattern, argv[0]);
        break;

      case 'f' :
        PAT_FILE = ON;
        argv++;
        argc--;
        if ((fp = open(argv[0], O_RDONLY)) < 0)
          {
          fprintf(stderr, "%s: Can't open pattern file %s\n", Progname,
             argv[0]);
          exit(2);
          }
        break;
      case 'h' :
        NOFILENAME = ON;
        break;
      case 'c' :
        NOUPPER = 0;
        break;
      case 'k' :
        argc--;
        if (argc == 0)
          {
          fprintf(stderr,
             "%s: the pattern should immediately follow the -k option\n"
             , Progname);
          usage();
          }
        CONSTANT = ON;
        argv++;
        strcat(Pattern, argv[0]);
        if (argc > 1 && argv[1][0] == '-')
          {
          fprintf(stderr,
             "%s: -k should be the last option in the command\n",
             Progname);
          exit(2);
          }
        break;
      case 'l' :
        FILENAMEONLY = ON;
        break;
      case 'n' :
        LINENUM = ON;                /* output prefixed by line no      */
        break;
      case 'v' :
        INVERSE = ON;                /* output no-matched lines         */
        break;
      case 't' :
        OUTTAIL = ON;                /* output from tail of delimiter   */
        break;
      case 'B' :
        BESTMATCH = ON;
        break;
      case 'w' :
        WORDBOUND = ON;              /* match to words                  */
        if (WHOLELINE)
          {
          fprintf(stderr, "%s: illegal option combination\n", Progname);
          exit(2);
          }
        break;
      case 'y' :
        NOPROMPT = ON;
        break;
      case 'I' :
        I = atoi(argv[0]+2);         /* Insertion Cost                  */
        JUMP = ON;
        break;
      case 'S' :
        S = atoi(argv[0]+2);         /* Substitution Cost               */
        JUMP = ON;
        break;
      case 'D' :
        DD = atoi(argv[0]+2);        /* Deletion Cost                   */
        JUMP = ON;
        break;
      case 'G' :
        FILEOUT = ON;
        COUNT = ON;
        break;
      default  :
        if (isdigit(c))
          {
          APPROX = ON;
          D = atoi(argv[0]+1);
          if (D > MaxError)
            {
            fprintf(stderr, "%s: the maximum number of errors is %d \n",
               Progname, MaxError);
            exit(2);
            }
          }
        else
          {
          fprintf(stderr, "%s: illegal option  -%c\n", Progname, c);
          usage();
          }
      }                              /* switch(c)                       */
    }                                /* while (--argc > 0 &&            */
                                     /* (*++argv)[0] == '-')            */
  if (FILENAMEONLY && NOFILENAME)
    {
    fprintf(stderr, "%s: -h and -l options are mutually exclusive\n",
       Progname);
    }
  if (COUNT && (FILENAMEONLY || NOFILENAME))
    {
    FILENAMEONLY = OFF;
    if (!FILEOUT)
      NOFILENAME = OFF;
    }
  if (!(PAT_FILE) && Pattern[0] == '\0')
    {                                /* Pattern not set with -e option  */
    if (argc == 0)
      usage();
    strcpy(Pattern, *argv);
    argc--;
    argv++;
    }
  Numfiles = 0;
  fd = 3;                            /* make sure it's not 0            */
  if (argc == 0)                     /* check Pattern against stdin     */
    fd = 0;
  else
    {
    if (!(Textfiles = (CHAR **)malloc(argc *sizeof(CHAR *))))
      {
      fprintf(stderr,
         "%s: malloc failure (you probably don't have enough memory)\n",
         Progname);
      exit(2);
      }
    while (argc--)
      {                              /* one or more filenames on command*/
                                     /* line -- put the valid filenames */
                                     /* in a array of strings           */

/*
      if ((filetype = check_file(*argv)) != ISASCIIFILE) {
        if(filetype == NOSUCHFILE) fprintf(stderr,"%s: %s: no such file or directory\n",Progname,*argv);
        argv++;
                                                                        */
      /*
      if ((filetype = check_file(*argv)) == NOSUCHFILE)
        {
        if (filetype == NOSUCHFILE)
          fprintf(stderr, "%s: %s: no such file or directory\n",
             Progname, *argv);
        argv++;
        }
      else
      */
        {                            /* file is ascii                   */
        if (!(Textfiles[Numfiles] = (CHAR *)malloc((strlen(*argv)+5))))
          /* NOTE: + 5 is to allow space for appended \*.* if required */
          {
          fprintf(stderr,
           "%s: malloc failure (you probably don't have enough memory)\n"
             , Progname);
          exit(2);
          }
        strcpy(Textfiles[Numfiles++], *argv++);
        }                            /* else                            */
      }                              /* while (argc--)                  */
    }                                /* else                            */
  checksg(Pattern, D);               /* check if the pattern is simple  */
  strcpy(OldPattern, Pattern);
  if (SGREP == 0)
    {
    preprocess(D_pattern, Pattern);
    strcpy(old_D_pat, D_pattern);
    M = maskgen(Pattern, D);
    }
  else
    M = strlen(OldPattern);

  if (PAT_FILE)
    prepf(fp);
  if (Numfiles > 1)                                /* 25/01/95 @DT */
    FNAME = ON;
  if (Numfiles > 0) /* ie: multiple files likely */
    {
    if (strchr(Textfiles[0], '*') || strchr(Textfiles[0], '?'))
      FNAME = ON;
    }
  if (NOFILENAME)
    FNAME = 0;
  num_of_matched = 0;
  compat();                          /* check compatibility between     */
                                     /* options                         */
  if (fd == 0)
    {
    if (FILENAMEONLY)
      {
      fprintf(stderr, "%s: -l option is not compatible with standard input\n",
         Progname);
      exit(2);
      }
    if (PAT_FILE)
      mgrep(fd);
    else
      {
      if (SGREP)
        sgrep(OldPattern, strlen(OldPattern), fd, D);
      else
        bitap(old_D_pat, Pattern, fd, M, D);
      }

    if (COUNT)
      {
      if (INVERSE && PAT_FILE)
        printf("%d\n", total_line-num_of_matched);
      else
        printf("%d\n", num_of_matched);
      }
    }
  else
    {
    for (i = 0; i < Numfiles; i++, num_of_matched = 0)
      {
      GetFileList(&stRootList, Textfiles[i]);
      if (!stRootList.pNext)
        {
        fprintf(stderr, "%s: %s: no such file or directory\n",
                Progname, Textfiles[i]);
        continue;
        }
      pstFiles = stRootList.pNext;
      while(pstFiles)
        {
        num_of_matched = 0;
        strcpy(CurrentFileName, pstFiles->pszFileName);
        if ((fd = open(CurrentFileName, O_RDONLY)) <= 0)
          {
          fprintf(stderr, "%s: can't open file %s (rc=%i)\n", Progname,
             CurrentFileName, errno);
          }
        else
          {
          if (PAT_FILE)
            mgrep(fd);
          else
            {
            if (SGREP)
              sgrep(OldPattern, strlen(OldPattern), fd, D);
            else
              bitap(old_D_pat, Pattern, fd, M, D);
            }

          if (num_of_matched)
            NOMATCH = OFF;
          if (COUNT && !FILEOUT)
            {
            if (INVERSE && PAT_FILE)
              {
              if (FNAME)
                printf("%s: %d\n", CurrentFileName, total_line-
                   num_of_matched);
              else
                printf("%d\n", total_line-num_of_matched);
              }
            else
              {
              if (FNAME)
                printf("%s: %d\n", CurrentFileName, num_of_matched);
              else
                printf("%d\n", num_of_matched);
              }
            }                          /* if COUNT                        */
      
          if (FILEOUT && num_of_matched)
            {
            file_out(CurrentFileName);
            }
          }                            /* else                            */
        close(fd);
        pstFiles = pstFiles->pNext;
        }
      FreeFileList(stRootList.pNext);
      }                              /* for i < Numfiles                */
    if (NOMATCH && BESTMATCH)
      {
      if (WORDBOUND || WHOLELINE || LINENUM || INVERSE)
        {
        SGREP = 0;
        preprocess(D_pattern, Pattern);
        strcpy(old_D_pat, D_pattern);
        M = maskgen(Pattern, D);
        }
      COUNT = ON;
      D = 1;
      while (D < M && D <= MaxError && num_of_matched == 0)
        {
        for (i = 0; i < Numfiles; i++)
          {
          GetFileList(&stRootList, Textfiles[i]);
          if (!stRootList.pNext)
            {
            fprintf(stderr, "%s: %s: no such file or directory\n",
                    Progname, Textfiles[i]);
            continue;
            }
          pstFiles = stRootList.pNext;
          while(pstFiles)
            {
            strcpy(CurrentFileName, pstFiles->pszFileName);
            if ((fd = open(CurrentFileName, O_RDONLY)) > 0)
              {
              if (PAT_FILE)
                mgrep(fd);
              else
                {
                if (SGREP)
                  sgrep(OldPattern, strlen(OldPattern), fd, D);
                else
                  bitap(old_D_pat, Pattern, fd, M, D);
                }
              }
            close(fd);
            pstFiles = pstFiles->pNext;
            }
          FreeFileList(stRootList.pNext);
          }                          /* for i < Numfiles                */
        D++;
        }                            /* while                           */

      if (num_of_matched > 0)
        {
        D--;
        COUNT = 0;
        if (!NOPROMPT)               /* 21/09/94                     @DT*/
          {
          if (D == 1)
            fprintf(stderr, "best match has 1 error, ");
          else
            fprintf(stderr, "best match has %d errors, ", D);

          fflush(stderr);
          if (num_of_matched == 1)
            fprintf(stderr, "there is 1 match, output it? (y/n)");
          else
            fprintf(stderr, "there are %d matches, output them? (y/n)",
               num_of_matched);

          fflush(stderr);
          scanf("%c", &c);
          }
        else
          c = 'y';
        if ((c == 'y')|(c == 'Y'))   /* 21/09/94                     @DT*/
          {
          for (i = 0; i < Numfiles; i++)
            {
            GetFileList(&stRootList, Textfiles[i]);
            if (!stRootList.pNext)
              {
              fprintf(stderr, "%s: %s: no such file or directory\n",
                      Progname, Textfiles[i]);
              continue;
              }
            pstFiles = stRootList.pNext;
            while(pstFiles)
              {
              strcpy(CurrentFileName, pstFiles->pszFileName);
              if ((fd = open(CurrentFileName, O_RDONLY)) > 0)
                {
                if (PAT_FILE)
                  mgrep(fd);
                else
                  {
                  if (SGREP)
                    sgrep(OldPattern, strlen(OldPattern), fd, D);
                  else
                    bitap(old_D_pat, Pattern, fd, M, D);
                  }
                }
              close(fd);
              pstFiles = pstFiles->pNext;
              }
            FreeFileList(stRootList.pNext);
            }                        /* for i < Numfiles                */
          }

        NOMATCH = 0;
        }
      }
    }
CONT:
  if (EATFIRST)
    {
    printf("\n");
    EATFIRST = OFF;
    }
  if (num_of_matched)
    NOMATCH = OFF;
  if (NOMATCH)
    exit(1);
  exit(0);
}                                    /* end of main()                   */

/********************************************************************/
/* FUNCTION:    file                                                */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
file_out(fname)
  char   *fname;
{
  int    num_read;
  int    fd;
  int    i,len;
  CHAR   buf[4097];

  if (FNAME)
    {
    len = strlen(fname);
    putchar('\n');
    for (i = 0; i < len; i++)
      putchar(':');
    putchar('\n');
    printf("%s\n", CurrentFileName);
    len = strlen(fname);
    for (i = 0; i < len; i++)
      putchar(':');
    putchar('\n');
    fflush(stdout);
    }
  fd = open(fname, O_RDONLY);
  while ((num_read = read(fd, buf, 4096)) > 0)
    write(1, buf, num_read);
}

/********************************************************************/
/* FUNCTION:    usage                                               */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
usage()
{
  fprintf(stderr, "%s V%s.%s - Advanced file searching utility\n",
     Progname, PVERSION, PSUBVERSION);
  fprintf(stderr, "usage: %s [-#bcdehiklnpqstvwxBDGIS] [-f patternfile] \
pattern [files]\n", Progname);
  printf("\n");
  fprintf(stderr, "summary of options:\n");
  fprintf(stderr, "-#: find best matches with at most # errors\n");
  fprintf(stderr,
   "-b: best match mode. find closest matches to pattern, no prompting\n"
     );
  fprintf(stderr, "-c: case-sensitive search, e.g., 'a' <> 'A'\n");
  fprintf(stderr, "-i: output the number of matched records\n");
  fprintf(stderr, "-d: define record delimiter\n");
  fprintf(stderr, "-e: pattern is...\n");
  fprintf(stderr, "-f: pattern file is...\n");
  fprintf(stderr, "-h: do not output file names\n");
  fprintf(stderr, "-k: constant pattern is...\n");
  fprintf(stderr, "-l: output the names of files that contain a match\n"
     );
  fprintf(stderr, "-n: output record prefixed by record number\n");
  fprintf(stderr, "-q: quite (silent) mode\n");
  fprintf(stderr, "-s: search for files in subdirectories too\n");
  fprintf(stderr, "-t: output from tail of delimiter\n");
  fprintf(stderr, "-v: output those records containing no matches\n");
  fprintf(stderr, "-w: pattern has to match as a word, e.g., 'win' will \
not match 'wind'\n");
  fprintf(stderr, "-y: no prompting\n");
  fprintf(stderr, "-x: match whole line only\n");
  fprintf(stderr, "-B: best match mode. find the closest matches to the \
pattern, prompt to show\n");
  fprintf(stderr, "-G: output the files that contain a match\n");
  fprintf(stderr, "-p: insertion cost is 0\n");
  fprintf(stderr, "-I: insertion cost is n\n");
  fprintf(stderr, "-S: substitution cost is n\n");
  fprintf(stderr, "-D: deletion cost is n\n");
  printf("\n");
  exit(2);
}

/********************************************************************/
/* FUNCTION:    checksg                                             */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
checksg(Pattern,D)
  CHAR   *Pattern;
  int D;
{
  char   c;
  int    i,m;

  m = strlen(Pattern);
  if (!(PAT_FILE) && m <= D)
    {
    fprintf(stderr,
       "%s: size of pattern must be greater than number of errors\n",
       Progname);
    exit(2);
    }
  SIMPLEPATTERN = ON;
  for (i = 0; i < m; i++)
    {
    switch (Pattern[i])
      {
      case ';' :
        SIMPLEPATTERN = OFF;
        break;
      case ',' :
        SIMPLEPATTERN = OFF;
        break;
      case '.' :
        SIMPLEPATTERN = OFF;
        break;
      case '*' :
        SIMPLEPATTERN = OFF;
        break;
      case '-' :
        SIMPLEPATTERN = OFF;
        break;
      case '[' :
        SIMPLEPATTERN = OFF;
        break;
      case ']' :
        SIMPLEPATTERN = OFF;
        break;
      case '(' :
        SIMPLEPATTERN = OFF;
        break;
      case ')' :
        SIMPLEPATTERN = OFF;
        break;
      case '<' :
        SIMPLEPATTERN = OFF;
        break;
      case '>' :
        SIMPLEPATTERN = OFF;
        break;
      case '^' :
        if (D > 0)
          SIMPLEPATTERN = OFF;
        break;
      case '$' :
        if (D > 0)
          SIMPLEPATTERN = OFF;
        break;
      case '|' :
        SIMPLEPATTERN = OFF;
        break;
      case '#' :
        SIMPLEPATTERN = OFF;
        break;
      case '\\' :
        SIMPLEPATTERN = OFF;
        break;
      default  :
        break;
      }
    }
  if (CONSTANT)
    SIMPLEPATTERN = ON;
  if (SIMPLEPATTERN == OFF)
    return ;
  if (NOUPPER && D)
    return ;
  if (JUMP == ON)
    return ;
  if (I == 0)
    return ;
  if (LINENUM)
    return ;
  if (DELIMITER)
    return ;
  if (INVERSE)
    return ;
  if (WORDBOUND && D > 0)
    return ;
  if (WHOLELINE && D > 0)
    return ;
  if (SILENT)
    return ;                         /* REMINDER: to be removed         */
  SGREP = ON;
  if (m >= 16)
    DNA = ON;
  for (i = 0; i < m; i++)
    {
    c = Pattern[i];
    if (c == 'a' || c == 'c' || c == 't' || c == 'g')
      ;
    else
      DNA = OFF;
    }
  return ;
}

/********************************************************************/
/* FUNCTION:    output                                              */
/* Description: Function description                                */
/*                                                                  */
/* Author:      D.Twyerould                                         */
/* Owner:       D.Twyerould                                         */
/* Globals:                                                         */
/*                                                                  */
/* Parameters:  None                                                */
/*                                                                  */
/* Returns:                                                         */
/********************* Amendment Summary ****************************/
/* Programmer  Date     Tag  Reason                                 */
/* ----------- -------- ---- -------------------------------------- */
/*                                                                  */
/********************************************************************/
output(buffer,i1,i2,j)
  register CHAR *buffer;
  int i1,i2,j;
{
  register CHAR *bp,*outend;

  if (i1 > i2)
    return ;
  num_of_matched++;
  if (COUNT)
    return ;
  if (SILENT)
    return ;
  if (OUTTAIL)
    {
    i1 = i1+D_length;
    i2 = i2+D_length;
    }
  if (DELIMITER)
    j = j+1;
  if (FIRSTOUTPUT)
    {
    if (buffer[i1] == '\n')
      {
      i1++;
      EATFIRST = ON;
      }
    FIRSTOUTPUT = 0;
    }
  if (TRUNCATE)
    {
    fprintf(stderr,
      "WARNING!!!  some lines have been truncated in output record #%d\n"
       , num_of_matched-1);
    }
  while (buffer[i1] == '\n' && i1 <= i2)
    {
    printf("\n");
    i1++;
    }
  if (FNAME == ON)
    printf("%s: ", CurrentFileName);
  if (LINENUM)
    printf("%d: ", j-1);
  bp = buffer+i1;
  outend = buffer+i2;
  while (bp <= outend)
    putchar(*bp++);
}


/* end of main.c                                                        */

