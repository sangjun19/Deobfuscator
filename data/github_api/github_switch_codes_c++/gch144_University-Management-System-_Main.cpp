
#include <iomanip>
#include <iostream>
#include <list>
#include <functional>
#include "University.h"
using namespace std;

template <typename T>
T find(std::list<T>& listOfElements, const int& id)
{
	for (auto element : listOfElements)
	{
		if (element->get_id() == id)
		{
			return element;
		}
	}
	return {};
}

enum Options
{
	create_university = 1,
	view_universities = 2,
	remove_university = 3,
	add_department = 4,
	remove_department = 5,
	Exit = 5
};

void addMenuItem(const int id, const string name)
{
	cout << "\t\t\t---------------------------------\n";
	cout << "\t\t\t|\t" << id << ". " << name << "\t|\n";
	cout << "\t\t\t---------------------------------\n";
}
//Main Menu
void print_menu() {
	cout << "\n";
	cout << "\t\t\t ==================================\n";
	cout << "\t\t\t|   University Management System |\n";
	cout << "\t\t\t ==================================\n\n";
	addMenuItem(1, "Add University");
	addMenuItem(2, "View Universities");
	addMenuItem(3, "Remove University");
	addMenuItem(4, "Add Department(s)");
	addMenuItem(5, "Remove Department(s)");
	addMenuItem(6, "EXIT");

	cout << "\t\t\tEnter choice: ";

}


void addDepartment(University* university)
{
	int noOfDepartments;
	cout << "How many departments you want to create: ";
enterAgain:
	cin >> noOfDepartments;
	noOfDepartments += university->departments.size();
addMore:
	if (noOfDepartments <= 0)
	{
		cout << "Invalid Input, Enter valid number for departments";
		goto enterAgain;
	}
	else
	{

		for (int i = university->departments.size(); i < noOfDepartments; i++)
		{
			string departmentName = "Department ";
			auto* dept = new Department(university->departments.size() + 1, departmentName.append(to_string(university->departments.size() + 1)) , university->get_id());
			auto* const date = new Date(01, 01, 2021);
			auto* teacher = new Teacher(dept->teachers.size() + 1, "John", "Doe", "Computer Science", date);
			dept->teachers.push_back(teacher);
			auto* student = new Student(dept->students.size() + 1, "John", "Doe", date, 2021, i);
			dept->students.push_back(student);
			university->departments.push_back(dept);

		}
	}
	string choice;
	cout << "Do you want to add more departments (Yes/No): ";
	cin >> choice;
	if (choice == "yes" || choice == "Yes" || choice == "YES")
	{
		int n;
		cout << "How many department(s) you want to add more: ";
		cin >> n;
		noOfDepartments += n;
		goto addMore;
	}
}

int main()
{
	
	list<University*> universities;
	int choice;
	do
	{
		print_menu();

		cin >> choice;

		switch (choice)
		{
		case create_university:
		{
			cin.ignore();
			string universityName = "University ";
			auto* uni = new University(universities.size() + 1, universityName.append(to_string(universities.size() + 1)));
			universities.push_back(uni);
			cout << "\nThere are total " << universities.size() << " Universities in our system" << endl;
			break;
		}
		case view_universities:
		{
			University::printUniversityDetails(universities);
			break;
		}
		case remove_university:
		{
			/*int uniId;
			cout << "Enter University Id to remove : ";
			cin >> uniId;*/

			universities.pop_back();
			break;
		}
		case add_department:
		{
			/*int uniId;
			cout << "Enter University Id to search : ";
			cin >> uniId;*/
			auto* university = find(universities, 1);
			if (university != nullptr)
			{
				addDepartment(university);
				cout << "Departments add with following details" << endl;
				University::printDepartmentDetails(university);
			}
			else
			{
				cout << "We are unable to find university with this id" << endl;
			}

			break;
		}
		case remove_department:
		{
			auto* university = find(universities, 1);
			if (university != nullptr)
			{
				//One depart ment will be removed form the end of the list
				university->departments.pop_back();
			}
		}
		default:
		{
			cout << "Invalid input" << endl;
			break;
		}
		}
	} while (choice != Exit);

	return 0;
}
