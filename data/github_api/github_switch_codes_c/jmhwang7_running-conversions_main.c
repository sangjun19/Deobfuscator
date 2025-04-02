#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include "conversions.h"
#include "utilities.h"

int main(int argc, char *argv[]) {
	bool fileInput = false;
	bool fileOutput = false;
	char *fileInputPath;
	char *fileOutputPath;
	int argCount;
	int minute, second;
	char ch;

	//Print an error if the user runs the program with no arguments.
	if (argc < 2) {
		fprintf(stderr, "Have at least 1 argument.\n");
		return 1;
	}

	//Parses out the command-line options entered.
	while ((ch = getopt(argc, argv, "i:o:h")) != EOF) {
		switch(ch) {
			case 'i':
				fileInput = true;
				fileInputPath = strdup(optarg);
				break;
			case 'o':
				fileOutput = true;
				fileOutputPath = strdup(optarg);
				break;
			case 'h':
				if (fileInput) {

					//Redirect stdin to the file given
					redirStdin(fileInputPath);

					//Redirects stdout to file given if -o was specified
					if (fileOutput) redirStdout(fileOutputPath);

					//Reads in the each of the mile times and returns mph
					while (scanf("%i:%i\n", &minute, &second) == 2) {
						printf("%.2f\n", mile_time_to_mph(minute, second));
					}
				}
				else {
					argShift(&argc, &argv, optind);

					//Redirects stdout to file given if -o was specified
					if (fileOutput) redirStdout(fileOutputPath);

					//Reads in each of the command-line arguments, and converts them.
					for (argCount = 0; argCount < argc; argCount++) {
						sscanf(argv[argCount], "%i:%i\n", &minute, &second);
						printf("%.2f\n", mile_time_to_mph(minute, second));
					}
				}
				break;
			default:
				fprintf(stderr, "Invalid option.");
				break;
		}
	}

	//Take care of memory allocated
	free(fileInputPath);
	free(fileOutputPath);

	return 0;
}