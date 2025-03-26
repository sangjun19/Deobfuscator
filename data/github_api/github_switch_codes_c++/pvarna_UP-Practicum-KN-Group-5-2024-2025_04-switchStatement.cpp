// Repository: pvarna/UP-Practicum-KN-Group-5-2024-2025
// File: Week02-ConditionalOperators/snippets/04-switchStatement.cpp

#include <iostream>

int main ()
{
    unsigned int dayOfTheWeek;

    std::cout << "Enter a day of the week (1 - 7): ";
    std::cin >> dayOfTheWeek;

    // Can be used with integer types and char
    switch (dayOfTheWeek)
    {
    case 1:
        std::cout << "Monday";
        break;
    case 2:
        std::cout << "Tuesday";
        break;
    case 3:
        std::cout << "Wednesday";
        break;
    case 4:
        std::cout << "Thursday";
        break;
    case 5:
        std::cout << "Friday";
        break;
    case 6:
        std::cout << "Saturday";
        break;
    case 7:
        std::cout << "Sunday";
        break;
    default:
        std::cout << "Invalid day of the week" << std::endl;
        return 1; // exit code for errors
    }

    std::cout << std::endl;

    return 0;
}