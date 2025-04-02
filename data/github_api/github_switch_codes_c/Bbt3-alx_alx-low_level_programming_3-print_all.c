#include "variadic_functions.h"

/**
 * print_all - print everything
 * @format:  list of types of arguments passed to the function
 * Return: Anything.
 */
void print_all(const char * const format, ...)
{
	int i;
	char c;
	char *s;
	double f;
	va_list ap;
	const char *fptr;

	va_start(ap, format);
	fptr = format;
	while (*fptr != '\0')
	{
		if (*fptr != '%')
		{
			putchar(*fptr);
			continue;
		}
		switch(*++fptr)
		{
			case 'c':
				c = (char)va_arg(ap, int);
				write(1, &c, 1);
				break;
			case 'i':
				i = va_arg(ap, int);
				printf("%i", i);
				break;
			case 'f':
				f = va_arg(ap, double);
				printf("%f", f);
				break;
			case 's':
				s = va_arg(ap, char*);
				printf("%s", s);
				break;
			default:
				putchar(*fptr);
		}
	}
}
