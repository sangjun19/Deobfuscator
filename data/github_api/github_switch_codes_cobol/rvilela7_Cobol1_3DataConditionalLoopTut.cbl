// Repository: rvilela7/Cobol1
// File: Learning/3DataConditionalLoopTut.cbl

       *> Complex DATA types with CLASS
       *> TYPE Definition
       *> 88 Data structure definition
       *> Conditional IFs,
       *> Loop

       
       IDENTIFICATION DIVISION. *> Configuration of custom data
       PROGRAM-ID. 3DataConditionalLoopTut.
       AUTHOR. Rui Vilela.
       DATE-WRITTEN.  16/04/2024.
       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SPECIAL-NAMES.
           CLASS PassingScore IS "A" THRU "C" , "D".
       DATA DIVISION.
        FILE SECTION. *> Can omit
        WORKING-STORAGE SECTION.
        01 AGE PIC 99 VALUE 0. *> Note that
        01 GRADE PIC 99 VALUE 0.
        01 SCORE PIC X(1) VALUE "B".
        01 CANVOTEFLAG PIC 9 VALUE 0.
           88 CANVOTE VALUE 1.  *> 88 - USE a condition TO a variable (bool) . This is a "special feature" from COB
           88 CANTVOTE VALUE 0.
        01 TESTNUMBER PIC X.
           88 ISPRIME VALUE "1", "3", "5", "7".
           88 ISODD VALUE "1", "3", "5", "7", "9".
           88 ISEVEN VALUE "2", "4", "6", "8".
           88 LESSTHAN5 VALUE "1" THRU "4". *> RANGE
           88 ANumber VALUE "0" THRU "9".

       PROCEDURE DIVISION.
       DISPLAY "Enter Age: " WITH NO ADVANCING
       ACCEPT AGE
       IF AGE > 18 THEN *> Normal IF. I CAN ALSO USE "GREATER THAN"
           DISPLAY "You can vote"
       ELSE *> ELSE IF Should be avoided. 
           DISPLAY "You can't vote"
       END-IF

       *> GREATER THAN
       *> LESS THAN
       *> EQUAL TO
       *> NOT EQUAL TO

       IF AGE LESS THAN 5 THEN DISPLAY "Stay HOME" END-IF

       IF AGE > 5 AND AGE < 18 THEN
           COMPUTE GRADE = AGE - 5
           DISPLAY "Go to GRADE " Grade
       END-IF

       IF AGE GREATER THAN OR EQUAL TO 18 *> You can ommit THEN!!
           DISPLAY "Go to College"
       END-IF *> Can BE ommited as well

       IF SCORE IS PASSINGSCORE THEN
           DISPLAY " You Passed"
       ELSE
           DISPLAY "You failed"
       END-IF

       *> NUMERIC ALPHABETIC ALPHABETIC-UPPER
       IF SCORE IS NOT NUMERIC THEN
           DISPLAY "Not a Number"
       END-IF

       *> SET

       IF AGE > 18 *> THEN OMIT
           SET CANVOTE TO TRUE *> I can't use MOVE because it's a conditional
       ELSE
           SET CANTVOTE TO TRUE
       END-IF
       DISPLAY "Vote: " CANVOTEFLAG

       DISPLAY "Enter Single Number or X to Exis: " *> Loop example
       ACCEPT TESTNUMBER
       PERFORM UNTIL NOT ANumber
           EVALUATE TRUE *> Similar to a switch
               WHEN ISPRIME DISPLAY "PRIME"
               WHEN ISODD DISPLAY "ODD"
               WHEN ISPRIME DISPLAY "Even"
               WHEN LESSTHAN5 DISPLAY "Less than 5"
               WHEN OTHER DISPLAY "Default Action"
           END-EVALUATE
           ACCEPT TESTNUMBER *> return with alpha or empty input breaks loop
       END-PERFORM

       STOP RUN.
