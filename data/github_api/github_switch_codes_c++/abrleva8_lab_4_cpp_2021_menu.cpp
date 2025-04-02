#include "menu.h"
#include "text_analyzer.h"
#include "console_input.h"
#include "file_input.h"
#include "file_output.h"
#include "test.h"
#include <iostream>

void greetings() {
	std::cout << "It is the fourth laboratory task of the first variation. "
		"The author is Levon Abramyan, Group 404, Course 1st" << std::endl << std::endl;

	std::cout << "The problem is: " << std::endl << std::endl;

	std::cout << "Count the number of characters, words, lines, paragraphs in the given text." << std::endl <<
		"Calculate the number of words in sentences and display a statistical table in which" << std::endl <<
		"the length of a sentence in words will correspond to the number of such sentences in the analyzed text."
		<< std::endl << std::endl;
}

void print_menu() {
	std::cout << std::endl << std::endl;
	std::cout << "Enter 1 to read data from console." << std::endl;
	std::cout << "Enter 2 to read data from file." << std::endl;
	std::cout << "Enter 3 to test program." << std::endl;
	std::cout << "Enter 0 to exit." << std::endl;
}

void interface_menu() {
	bool is_restart = true;
	ConsoleInput ci;
	std::unique_ptr<Input> input;

	do {
		const FileOutput fo;
		print_menu();
		switch (const int choice = ci.get_number(static_cast<int> (EXIT), static_cast<int> (TEST)); choice) {
			case EXIT:
			std::cout << "Your choice is EXIT" << std::endl;
			is_restart = false;
			continue;
			case CONSOLE: {
				input = std::make_unique<ConsoleInput>();
			}
			break;

			case FILES: {
				input = std::make_unique<FileInput>();
			}
			break;

			case TEST: {
				std::unique_ptr<Test> test(new Test);

				test->start();

				if (!test->get_is_success()) {
					std::cout << "Sorry, but there is a problem test. The program will be closed." << std::endl;
					return;
				}
				continue;
			}
			default: 
			break;
		}
		std::unique_ptr<Text> text(new Text);
		const bool is_success = input->read(*text);
		const bool is_file_input = input->get_is_file_input();
		if (is_success) {
			std::cout << std::endl << "Data read successfully!" << std::endl;
		} else {
			continue;
		}
		if (!is_file_input) {
			fo.save_input_data(*text);
		}
		text->print_info();
		text->print_sentences_info();
		fo.save_output_data(*text);

	} while (is_restart);
}