// Repository: Bybalesh/testLang
// File: lang/1.cl

(function (Prism) {
	/* OpenCL kernel language */
	Prism.languages.opencl = Prism.languages.extend('c', {
		// Extracted from the official specs (2.0) and http://streamcomputing.eu/downloads/?opencl.lang (opencl-keywords, opencl-types) and http://sourceforge.net/tracker/?func=detail&aid=2957794&group_id=95717&atid=612384 (Words2, partly Words3)
		'keyword': /\b(?:(?:__)?(?:constant|global|kernel|local|private|read_only|read_write|write_only)|__attribute__|auto|(?:bool|u?(?:char|int|long|short)|half|quad)(?:2|3|4|8|16)?|break|case|complex|const|continue|(?:double|float)(?:16(?:x(?:1|2|4|8|16))?|1x(?:1|2|4|8|16)|2(?:x(?:1|2|4|8|16))?|3|4(?:x(?:1|2|4|8|16))?|8(?:x(?:1|2|4|8|16))?)?|default|do|else|enum|extern|for|goto|if|imaginary|inline|packed|pipe|register|restrict|return|signed|sizeof|static|struct|switch|typedef|uniform|union|unsigned|void|volatile|while)\b/,
		// Extracted from http://streamcomputing.eu/downloads/?opencl.lang (opencl-const)
		// Math Constants: https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/mathConstants.html
		// Macros and Limits: https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/macroLimits.html
		'number': /(?:\b0x(?:[\da-f]+(?:\.[\da-f]*)?|\.[\da-f]+)(?:p[+-]?\d+)?|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?)[fuhl]{0,4}/i,
		'boolean': /\b(?:false|true)\b/,
		'constant-opencl-kernel': {```c
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main() {
    // Загрузка ядра OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem a_mem_obj = NULL;
    cl_mem b_mem_obj = NULL;
    cl_mem c_mem_obj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_int ret;

    // Загружаем ядро OpenCL
    FILE *fp;
    char fileName[] = "./kernel.cl";
    size_t source_size;
    char *source_str;
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
