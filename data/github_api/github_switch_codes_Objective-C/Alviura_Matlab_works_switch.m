// Repository: Alviura/Matlab_works
// File: switch.m

score=input("Enter the grade")
grade='';
switch true
    case score>=70
        grade='A';
    case score>=60
        grade='B';
    case score>=50
        grade='C';
    case score>=40
        grade='D';
    case score<40
        grade='F'
end
fprintf("The grade for a score %d is %s")
