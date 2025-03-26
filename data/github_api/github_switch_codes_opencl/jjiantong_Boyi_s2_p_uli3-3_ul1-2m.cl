// Repository: jjiantong/Boyi
// File: OpenCL_applications_on_FPGAs/CEDD/SWI/s2_p_uli3-3_ul1-2m/s2_p_uli3-3_ul1-2m.cl

#define _OPENCL_COMPILER_

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#define COLS 626
#define ROWS 354

#include "support/common.h"
__constant float gaus[3][3] = {{0.0625f, 0.125f, 0.0625f}, {0.1250f, 0.250f, 0.1250f}, {0.0625f, 0.125f, 0.0625f}};
__constant int   sobx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__constant int   soby[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// https://github.com/smskelley/canny-opencl
// Gaussian Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__kernel void gaussian_kernel(	__global unsigned char *data, __global unsigned char *out,
								int cols, int rows) {
    
	unsigned char l_data[ROWS * COLS];
	
	for(int i = 0; i < rows * cols; i ++)
		l_data[i] = data[i];

	for (int row = 1; row < rows - 1; row ++){

		unsigned char l_out[COLS];

		for (int col = 1; col < cols / 2; col ++){

			int sum0 = 0;
			int sum1 = 0;
			int col0 = col * 2 - 1;
			int col1 = col * 2;
			int pos0 = row * cols + col0;
			int pos1 = row * cols + col1;

			#pragma unroll
			for (int i = 0; i < 3; i ++){
				#pragma unroll
				for (int j = 0; j < 3; j ++){
					sum0 += gaus[i][j] * l_data[(i + row - 1) * cols + j + col0 - 1];
					sum1 += gaus[i][j] * l_data[(i + row - 1) * cols + j + col1 - 1];
				}
			}
			l_out[col0] =  min(255, max(0, sum0));
			l_out[col1] =  min(255, max(0, sum1));
		}

		for(int col = 1; col < cols - 1; col ++)
			out[row * cols + col] = l_out[col];
	}
}


// Sobel kernel. Apply sobx and soby separately, then find the sqrt of their
//               squares.
// data:  image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out:   image output data (8B1C)
// theta: angle output data
__kernel 
void sobel_kernel(	__global unsigned char *data, __global unsigned char *out, __global unsigned char *theta,
					int cols, int rows) {
    // collect sums separately. we're storing them into floats because that
    // is what hypot and atan2 will expect.

	unsigned char l_data[ROWS * COLS];

	for(int i = 0; i < rows * cols; i ++)
		l_data[i] = data[i];
	
	for (int row = 1; row < rows - 1; row ++){

		unsigned char l_out[COLS];
		unsigned char l_theta[COLS];

		for (int col = 1; col < cols / 2; col ++){
			
			int col0 = col * 2 - 1;
			int col1 = col * 2;
			int pos0 = row * cols + col0;
			int pos1 = row * cols + col1;

			const float PI = 3.14159265;
			float sumx0 = 0, sumy0 = 0, angle0 = 0;
			float sumx1 = 0, sumy1 = 0, angle1 = 0;

			// find x and y derivatives
			#pragma unroll
			for (int i = 0; i < 3; i ++){
				#pragma unroll
				for (int j = 0; j < 3; j ++){
					sumx0 += sobx[i][j] * l_data[(i + row - 1) * cols + j + col0 - 1];
					sumy0 += soby[i][j] * l_data[(i + row - 1) * cols + j + col0 - 1];
					sumx1 += sobx[i][j] * l_data[(i + row - 1) * cols + j + col1 - 1];
					sumy1 += soby[i][j] * l_data[(i + row - 1) * cols + j + col1 - 1];
				}
			}

			// The output is now the square root of their squares, but they are
			// constrained to 0 <= value <= 255. Note that hypot is a built in function
			// defined as: hypot(x,y) = sqrt(x*x, y*y).			
			l_out[col0] =  min(255, max(0, (int)hypot(sumx0, sumy0)));
			l_out[col1] =  min(255, max(0, (int)hypot(sumx1, sumy1)));

			// Compute the direction angle theta in radians
			// atan2 has a range of (-PI, PI) degrees
			angle0 = atan2(sumy0, sumx0);
			angle1 = atan2(sumy1, sumx1);

			// Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
			// then store it in the theta buffer at the proper position
			//theta[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
			if(angle0 <= -7 * PI / 8)
				l_theta[col0] = 0;
			else if(angle0 <= -5 * PI / 8)
				l_theta[col0] = 45;
			else if(angle0 <= -3 * PI / 8)
				l_theta[col0] = 90;
			else if(angle0 <= - PI / 8)
				l_theta[col0] = 135;
			else if(angle0 <= PI / 8)
				l_theta[col0] = 0;
			else if(angle0 <= 3 * PI / 8)
				l_theta[col0] = 45;
			else if(angle0 <= 5 * PI / 8)
				l_theta[col0] = 90;
			else if(angle0 <= 7 * PI / 8)
				l_theta[col0] = 135;
			else
				l_theta[col0] = 0; 

			if(angle1 <= -7 * PI / 8)
				l_theta[col1] = 0;
			else if(angle1 <= -5 * PI / 8)
				l_theta[col1] = 45;
			else if(angle1 <= -3 * PI / 8)
				l_theta[col1] = 90;
			else if(angle1 <= - PI / 8)
				l_theta[col1] = 135;
			else if(angle1 <= PI / 8)
				l_theta[col1] = 0;
			else if(angle1 <= 3 * PI / 8)
				l_theta[col1] = 45;
			else if(angle1 <= 5 * PI / 8)
				l_theta[col1] = 90;
			else if(angle1 <= 7 * PI / 8)
				l_theta[col1] = 135;
			else
				l_theta[col1] = 0; 
		}

		for(int col = 1; col < cols - 1; col ++){
			out[row * cols + col] = l_out[col];
		}

		for(int col = 1; col < cols - 1; col ++){
			theta[row * cols + col] = l_theta[col];
		}
	}
}


// Non-maximum Supression Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
// theta: angle input data
__kernel void non_max_supp_kernel(__global unsigned char *data, __global unsigned char *out,
    __global unsigned char *theta, int cols, int rows) {

	unsigned char l_data[ROWS * COLS];

	for(int i = 0; i < rows * cols; i ++)
		l_data[i] = data[i];
	
	for (int row = 1; row < rows - 1; row ++){

		unsigned char l_out[COLS];

		for (int col = 1; col < cols / 2; col ++){
			
			int col0 = col * 2 - 1;
			int col1 = col * 2;
			int pos0 = row * cols + col0;
			int pos1 = row * cols + col1;
			unsigned char my_magnitude0 = l_data[pos0];
			unsigned char my_magnitude1 = l_data[pos1];

			// The following variables are used to address the matrices more easily
			switch(theta[pos0]) {
			// A gradient angle of 0 degrees = an edge that is North/South
			// Check neighbors to the East and West
			case 0:
				// supress me if my neighbor has larger magnitude
				if(my_magnitude0 <= l_data[pos0 + 1] || // east
					my_magnitude0 <= l_data[pos0 - 1]) // west
				{
					l_out[col0] = 0;
				}
				// otherwise, copy my value to the output buffer
				else {
					l_out[col0] = my_magnitude0;
				}
				break;

			// A gradient angle of 45 degrees = an edge that is NW/SE
			// Check neighbors to the NE and SW
			case 45:
				// supress me if my neighbor has larger magnitude
				if(my_magnitude0 <= l_data[(row - 1) * cols + col0 + 1] || // north east
					my_magnitude0 <= l_data[(row + 1) * cols + col0 - 1]) // south west
				{
					l_out[col0] = 0;
				}
				// otherwise, copy my value to the output buffer
				else {
					l_out[col0] = my_magnitude0;
				}
				break;

			// A gradient angle of 90 degrees = an edge that is E/W
			// Check neighbors to the North and South
			case 90:
				// supress me if my neighbor has larger magnitude
				if(my_magnitude0 <= l_data[(row - 1) * cols + col0] || // north
					my_magnitude0 <= l_data[(row + 1) * cols + col0]) // south
				{
					l_out[col0] = 0;
				}
				// otherwise, copy my value to the output buffer
				else {
					l_out[col0] = my_magnitude0;
				}
				break;

			// A gradient angle of 135 degrees = an edge that is NE/SW
			// Check neighbors to the NW and SE
			case 135:
				// supress me if my neighbor has larger magnitude
				if(my_magnitude0 <= l_data[(row - 1) * cols + col0 - 1] || // north west
					my_magnitude0 <= l_data[(row + 1) * cols + col0 + 1]) // south east
				{
					l_out[col0] = 0;
				}
				// otherwise, copy my value to the output buffer
				else {
					l_out[col0] = my_magnitude0;
				}
				break;

			default: l_out[col0] = my_magnitude0; break;
			}

			switch(theta[pos1]) {
			// A gradient angle of 0 degrees = an edge that is North/South
			// Check neighbors to the East and West
			case 0:
				// supress me if my neighbor has larger magnitude
				if(my_magnitude1 <= l_data[pos1 + 1] || // east
					my_magnitude1 <= l_data[pos1 - 1]) // west
				{
					l_out[col1] = 0;
				}
				// otherwise, copy my value to the output buffer
				else {
					l_out[col1] = my_magnitude1;
				}
				break;

			// A gradient angle of 45 degrees = an edge that is NW/SE
			// Check neighbors to the NE and SW
			case 45:
				// supress me if my neighbor has larger magnitude
				if(my_magnitude1 <= l_data[(row - 1) * cols + col1 + 1] || // north east
					my_magnitude1 <= l_data[(row + 1) * cols + col1 - 1]) // south west
				{
					l_out[col1] = 0;
				}
				// otherwise, copy my value to the output buffer
				else {
					l_out[col1] = my_magnitude1;
				}
				break;

			// A gradient angle of 90 degrees = an edge that is E/W
			// Check neighbors to the North and South
			case 90:
				// supress me if my neighbor has larger magnitude
				if(my_magnitude1 <= l_data[(row - 1) * cols + col1] || // north
					my_magnitude1 <= l_data[(row + 1) * cols + col1]) // south
				{
					l_out[col1] = 0;
				}
				// otherwise, copy my value to the output buffer
				else {
					l_out[col1] = my_magnitude1;
				}
				break;

			// A gradient angle of 135 degrees = an edge that is NE/SW
			// Check neighbors to the NW and SE
			case 135:
				// supress me if my neighbor has larger magnitude
				if(my_magnitude1 <= l_data[(row - 1) * cols + col1 - 1] || // north west
					my_magnitude1 <= l_data[(row + 1) * cols + col1 + 1]) // south east
				{
					l_out[col1] = 0;
				}
				// otherwise, copy my value to the output buffer
				else {
					l_out[col1] = my_magnitude1;
				}
				break;

			default: l_out[col1] = my_magnitude1; break;
			}

		}

		for(int col = 1; col < cols - 1; col ++)
			out[row * cols + col] = l_out[col];
	}
}


// Hysteresis Threshold Kernel
// data: image input data with each pixel taking up 1 byte (8Bit 1Channel)
// out: image output data (8B1C)
__kernel void hyst_kernel(__global unsigned char *data, __global unsigned char *out, int cols, int rows) {
	
	unsigned char l_data[ROWS * COLS];

	for(int i = 0; i < rows * cols; i ++)
		l_data[i] = data[i];
	
	for (int row = 0; row < rows; row ++){
		
		unsigned char l_out[COLS];
		
		for (int col = 0; col < cols / 2; col ++){

			int col0 = col * 2;
			int col1 = col * 2 + 1;
			int pos0 = row * cols + col0;
			int pos1 = row * cols + col1;

			float lowThresh  = 10;
			float highThresh = 70;
			const unsigned char EDGE = 255;

			unsigned char magnitude0 = l_data[pos0];
			unsigned char magnitude1 = l_data[pos1];

			if(magnitude0 >= highThresh)
				l_out[col0] = EDGE;
			else if(magnitude0 <= lowThresh)
				l_out[col0] = 0;
			else {
				float med = (highThresh + lowThresh) / 2;

				if(magnitude0 >= med)
					l_out[col0] = EDGE;
				else
					l_out[col0] = 0;
			}		

			if(magnitude1 >= highThresh)
				l_out[col1] = EDGE;
			else if(magnitude1 <= lowThresh)
				l_out[col1] = 0;
			else {
				float med = (highThresh + lowThresh) / 2;

				if(magnitude1 >= med)
					l_out[col1] = EDGE;
				else
					l_out[col1] = 0;
			}			
		}

		for(int col = 1; col < cols - 1; col ++)
			out[row * cols + col] = l_out[col];
	}
}
