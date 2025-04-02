#include "Harl.hpp"

void	print_level(std::string level) {
	Harl	harl;
	for (size_t i = 0; i < level.size(); ++i) level[i] = toupper(level[i]);
	std::cout << "[ " << level << " ]" << std::endl;
	harl.complain(level);
	std::cout << std::endl;
}

int	main(int argc, char *argv[]) {
	if (argc == 1) {
		std::cout << "Pass an argument." << std::endl;
		return 0;
	}
	int	level_num;
	std::string	levels[] = {"DEBUG", "INFO", "WARNING", "ERROR"};
	level_num = -1;
	for (int i = 0; i < 4; ++i) {
		if (levels[i].compare(argv[1]) == 0) level_num = i;
	}
	switch (level_num) {
		case 0:
			print_level(levels[0]);
		case 1:
			print_level(levels[1]);
		case 2:
			print_level(levels[2]);
		case 3:
			print_level(levels[3]);
			break;
		default:
			std::cout << "[ Probably complaining about insignificant problems ]" << std::endl;
	}
}
