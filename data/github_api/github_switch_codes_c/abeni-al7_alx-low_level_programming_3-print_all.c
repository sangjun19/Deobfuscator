#include "variadic_functions.h"

/**
 * print_all - prints anything
 * @format: list of types of arguments passed to the function
 */

void print_all(const char * const format, ...)
{
	int i = 0;
	const char *str;
	char sep = '\0';
	va_list args;

	va_start(args, format);

	while (format && format[i])
	{
		switch (format[i])
		{
			case 'c':
				printf("%c", va_arg(args, int));
				sep = ',';
				break;
			case 'i':
				printf("%d", va_arg(args, int));
				sep = ',';
				break;
			case 'f':
				printf("%f", va_arg(args, double));
				sep = ',';
				break;
			case 's':
				str = va_arg(args, char *);
				if (str == NULL)
					str = "(nil)";
				printf("%s", str);
				sep = ',';
				break;
		}
		i++;

		if (format[i] && sep != '\0')
			printf("%c ", sep);
		sep = '\0';
	}
	printf("\n");
	va_end(args);
}
