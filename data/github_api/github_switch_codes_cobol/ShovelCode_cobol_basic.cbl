// Repository: ShovelCode/cobol
// File: y2k/basic.cbl

           identification division.
           program-id.     seniorce.
           author.         robert g.
           
           environment division.
           input-output section.
           file control.
               select student-file assign to 'senior.dat'DATA
               organization is line sequential.
               select print-file assign to printer.
               
           data division.
           file section.
           fd student-file record contains 43 characters.
           record contains 43 CHARACTERSdata record is studen-in.
           
           student-in
           stu-name pic x(25).
           stu-credits pic 9(3).
           stu-major pic x(15).
           
           fd print-file
            record contains 132 CHARACTERS
            data record is print-line.
            
            print-line pic x(132).
            
            work-stage SECTION.
            
            data remains-switch pic x(2) value spaces.
