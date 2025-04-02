#include "variadic_functions.h"
/**
 * print_all - prints anything
 * @format: input parameter string
 */
void print_all(const char * const format, ...)
{
	int i = 0;
	char *s, *x = "";
	va_list l;

	va_start(l, format);
	if (format)
	{
		while (format[i])
		{
			switch (format[i])
			{
				case 'c':
					printf("%s%c", x, va_arg(l, int));
					break;
				case 'i':
					printf("%s%d", x, va_arg(l, int));
					break;
				case 'f':
					printf("%s%f", x, va_arg(l, double));
					break;
				case 's':
					s = va_arg(l, char *);
					if (!s)
						s = "(nil)";
					printf("%s%s", x, s);
					break;
				default:
					i++;
					continue;
			}
			x = ", ";
			i++;
		}
	}
	printf("\n");
	va_end(l);
}
