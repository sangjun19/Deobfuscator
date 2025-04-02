// Repository: jhill1971/grade-calculator
// File: gradeCalculator.cob

       IDENTIFICATION DIVISION.
       PROGRAM-ID. gradeCalculator.
       AUTHOR. James Hill
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 numericGrade PIC 999.

       PROCEDURE DIVISION.
        DISPLAY "Enter numeric grade: " WITH NO ADVANCING.
        ACCEPT numericGrade.
      * EVALUATE implements a case/switch structure   
        EVALUATE TRUE
            WHEN numericGrade >= 90 AND <= 100
                DISPLAY "Student Grade = A"
            WHEN numericGrade >= 80 AND <= 89
                DISPLAY "Student Grade = B"
            WHEN numericGrade >= 70 AND <= 79
                DISPLAY "Student Grade = C"
            WHEN numericGrade >= 60 AND <= 69
                DISPLAY "Student Grade = D"
            WHEN numericGrade <= 59
                DISPLAY "Student Grade = F"
            WHEN OTHER
                DISPLAY "Unexpected Condition"
        END-EVALUATE.
        STOP RUN.

