// Repository: dsvua/jetracer-orbslam2
// File: src/cuda/fast.cu

#include "fast.cuh"
#include "nms.cuh"
#include "../cuda_common.h"
#include "../SlamGpuPipeline/defines.h"

namespace Jetracer
{
    // ---------------------------------------
    //                kernels
    // ---------------------------------------
    __inline__ __device__ unsigned char fast_gpu_is_corner(const unsigned int &address,
                                                           const int &min_arc_length)
    {
        int ones = __popc(address);
        if (ones < min_arc_length)
        { // if we dont have enough 1-s in the address, dont even try
            return 0;
        }
        unsigned int address_dup = address | (address << 16); //duplicate the low 16-bits at the high 16-bits
        while (ones > 0)
        {
            address_dup <<= __clz(address_dup); // shift out the high order zeros
            int lones = __clz(~address_dup);    // count the leading ones
            if (lones >= min_arc_length)
            {
                return 1;
            }
            address_dup <<= lones; // shift out the high order ones
            ones -= lones;
        }
        return 0;
    }

    __global__ void fast_gpu_calculate_lut_kernel(unsigned char *__restrict__ d_corner_lut,
                                                  const int min_arc_length)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x; // all 16 bits come from here
        d_corner_lut[x] = fast_gpu_is_corner(x, min_arc_length);
    }

    __inline__ __device__ int bresenham_circle_offset_pitch(const int &i,
                                                            const int &pitch,
                                                            const int &pitch2,
                                                            const int &pitch3)
    {
        /*
   * Note to future self and others:
   * this function is only should be called in for loops that are unrolled.
   * Due to unrollment, the if else structure disappears, and the offsets get
   * substituted.
   *
   * Order within the circle:
   *
   *      7 8  9
   *    6       10
   *  5           11
   *  4     x     12
   *  3           13
   *   2        14
   *     1 0 15
   */
        int offs = 0;
        if (i == 0)
            offs = pitch3;
        else if (i == 1)
            offs = pitch3 - 1;
        else if (i == 2)
            offs = pitch2 - 2;
        else if (i == 3)
            offs = pitch - 3;
        else if (i == 4)
            offs = -3;
        else if (i == 5)
            offs = -pitch - 3;
        else if (i == 6)
            offs = -pitch2 - 2;
        else if (i == 7)
            offs = -pitch3 - 1;
        else if (i == 8)
            offs = -pitch3;
        else if (i == 9)
            offs = -pitch3 + 1;
        else if (i == 10)
            offs = -pitch2 + 2;
        else if (i == 11)
            offs = -pitch + 3;
        else if (i == 12)
            offs = 3;
        else if (i == 13)
            offs = pitch + 3;
        else if (i == 14)
            offs = pitch2 + 2;
        else if (i == 15)
            offs = pitch3 + 1;
        return offs;
    }

    __inline __device__ unsigned int fast_gpu_prechecks(const float &c_t,
                                                        const float &ct,
                                                        const unsigned char *image_ptr,
                                                        const int &image_pitch,
                                                        const int &image_pitch2,
                                                        const int &image_pitch3)
    {
        /*
        * Note to future self:
        * using too many prechecks of course doesnt help
        */
        // (-3,0) (3,0) -> 4,12
        float px0 = (float)image_ptr[bresenham_circle_offset_pitch(4, image_pitch, image_pitch2, image_pitch3)];
        float px1 = (float)image_ptr[bresenham_circle_offset_pitch(12, image_pitch, image_pitch2, image_pitch3)];
        if ((signbit(px0 - c_t) | signbit(px1 - c_t) | signbit(ct - px0) | signbit(ct - px1)) == 0)
        {
            return 1;
        }
        // (0,3), (0,-3) -> 0, 8
        px0 = (float)image_ptr[bresenham_circle_offset_pitch(0, image_pitch, image_pitch2, image_pitch3)];
        px1 = (float)image_ptr[bresenham_circle_offset_pitch(8, image_pitch, image_pitch2, image_pitch3)];
        if ((signbit(px0 - c_t) | signbit(px1 - c_t) | signbit(ct - px0) | signbit(ct - px1)) == 0)
        {
            return 1;
        }
        return 0;
    }

    __inline__ __device__ int fast_gpu_is_corner_quick(
        const unsigned char *__restrict__ d_corner_lut,
        const float *__restrict__ px,
        const float &center_value,
        const float &threshold,
        unsigned int &dark_diff_address,
        unsigned int &bright_diff_address)
    {
        const float ct = center_value + threshold;
        const float c_t = center_value - threshold;
        dark_diff_address = 0;
        bright_diff_address = 0;

#pragma unroll 16
        for (int i = 0; i < 16; ++i)
        {
            int darker = signbit(px[i] - c_t);
            int brighter = signbit(ct - px[i]);
            dark_diff_address += signbit(px[i] - c_t) ? (1 << i) : 0;
            bright_diff_address += signbit(ct - px[i]) ? (1 << i) : 0;
        }
        return (d_corner_lut[dark_diff_address] || d_corner_lut[bright_diff_address]);
    }

    template <fast_score SCORE>
    __global__ void fast_gpu_calc_corner_response_kernel(
        const int image_width,
        const int image_height,
        const int image_pitch,
        const unsigned char *__restrict__ d_image,
        const int horizontal_border,
        const int vertical_border,
        const unsigned char *__restrict__ d_corner_lut,
        const float threshold,
        const int min_arc_length,
        const int response_pitch_elements,
        float *__restrict__ d_response)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x; // thread id X
        const int y = blockDim.y * blockIdx.y + threadIdx.y; // thread id Y
        if (x < image_width && y < image_height)
        {
            const int resp_offset = y * response_pitch_elements + x;
            d_response[resp_offset] = 0.0f;
            if ((x >= horizontal_border) &&
                (y >= vertical_border) &&
                (x < (image_width - horizontal_border)) &&
                (y < (image_height - vertical_border)))
            {
                const unsigned char *d_image_ptr = d_image + y * image_pitch + x;
                const float c = (float)(*d_image_ptr);
                const float ct = c + threshold;
                const float c_t = c - threshold;
                /*
                * Note to future self:
                * we need to create 2 differences for each of the 16 pixels
                * have 1 lookup table, and look-up both values
                *
                * c_t stands for: c - threshold (epsilon)
                * ct stands for : c + threshold (epsilon)
                *
                * Label of px:
                * - darker  if   px < c_t              (1)
                * - similar if   c_t <= px <= ct      (2)
                * - brighter if  ct < px             (3)
                *
                * Darker diff: px - c_t
                * sign will only give 1 in case of (1), and 0 in case of (2),(3)
                *
                * Similarly, brighter diff: ct - px
                * sign will only give 1 in case of (3), and 0 in case of (2),(3)
                */
                unsigned int dark_diff_address = 0;
                unsigned int bright_diff_address = 0;

                // Precalculate pitches
                const int image_pitch2 = image_pitch << 1;
                const int image_pitch3 = image_pitch + (image_pitch << 1);

                // Do a coarse corner check
                // TODO: I could use the results of the prechecks afterwards
                if (fast_gpu_prechecks(c_t, ct, d_image_ptr, image_pitch, image_pitch2, image_pitch3))
                {
                    return;
                }

                float px[16];
#pragma unroll 16
                for (int i = 0; i < 16; ++i)
                {
                    int image_ptr_offset = bresenham_circle_offset_pitch(i, image_pitch, image_pitch2, image_pitch3);
                    px[i] = (float)d_image_ptr[image_ptr_offset];
                    int darker = signbit(px[i] - c_t);
                    int brighter = signbit(ct - px[i]);
                    dark_diff_address += signbit(px[i] - c_t) ? (1 << i) : 0;
                    bright_diff_address += signbit(ct - px[i]) ? (1 << i) : 0;
                }
                // Look up these addresses, whether they qualify for a corner
                // If any of these qualify for a corner, it is a corner candidate, yaay
                if (d_corner_lut[dark_diff_address] || d_corner_lut[bright_diff_address])
                {
                    /*
                    * Note to future self:
                    * Only calculate the score once we determined that the pixel is considered
                    * a corner. This policy gave better results than computing the score
                    * for every pixel
                    */
                    if (SCORE == SUM_OF_ABS_DIFF_ALL)
                    {
                        float response = 0.0f;
#pragma unroll 16
                        for (int i = 0; i < 16; ++i)
                        {
                            response += fabsf(px[i] - c);
                        }
                        d_response[resp_offset] = response;
                    }
                    else if (SCORE == SUM_OF_ABS_DIFF_ON_ARC)
                    {
                        float response_bright = 0.0f;
                        float response_dark = 0.0f;
#pragma unroll 16
                        for (int i = 0; i < 16; ++i)
                        {
                            float absdiff = fabsf(px[i] - c) - threshold;
                            response_dark += (dark_diff_address & (1 << i)) ? absdiff : 0.0f;
                            response_bright += (bright_diff_address & (1 << i)) ? absdiff : 0.0f;
                        }
                        d_response[resp_offset] = fmaxf(response_bright, response_dark);
                    }
                    else if (SCORE == MAX_THRESHOLD)
                    {
                        // Binary search for the maximum threshold value with which the given
                        // point is still a corner
                        float min_thr = threshold + 1;
                        float max_thr = 255.0f;
                        while (min_thr <= max_thr)
                        {
                            float med_thr = floorf((min_thr + max_thr) * 0.5f);
                            // try out med_thr as a new threshold
                            if (fast_gpu_is_corner_quick(d_corner_lut,
                                                         px,
                                                         c,
                                                         med_thr,
                                                         dark_diff_address,
                                                         bright_diff_address))
                            {
                                // still a corner
                                min_thr = med_thr + 1.0f;
                            }
                            else
                            {
                                // not a corner anymore
                                max_thr = med_thr - 1.0f;
                            }
                        }
                        d_response[resp_offset] = max_thr;
                    }
                }
            }
        }
    }

    // ---------------------------------------
    //            host functions
    // ---------------------------------------
    void fast_gpu_calculate_lut(unsigned char *d_corner_lut,
                                const int &min_arc_length)
    {
        // every thread writes a byte: in total 64kB gets written
        kernel_params_t p = cuda_gen_kernel_params_1d(64 * 1024, 256);
        dim3 threads(256);
        dim3 blocks((64 * 1024 + threads.x - 1) / threads.x);
        fast_gpu_calculate_lut_kernel<<<p.blocks_per_grid, p.threads_per_block>>>(d_corner_lut,
                                                                                  min_arc_length);
        // checkCudaErrors(cudaStreamSynchronize(stream));
    }

    void fast_gpu_calc_corner_response(const int image_width,
                                       const int image_height,
                                       const int image_pitch,
                                       const unsigned char *d_image,
                                       const int horizontal_border,
                                       const int vertical_border,
                                       const unsigned char *d_corner_lut,
                                       const float threshold,
                                       const int min_arc_length,
                                       const fast_score score,
                                       const int response_pitch_elements,
                                       float *d_response,
                                       cudaStream_t stream)
    {
        // Note: I'd like to launch 128 threads / thread block
        std::size_t threads_per_x = (image_width % CUDA_WARP_SIZE == 0) ? CUDA_WARP_SIZE : 16;
        std::size_t threads_per_y = 128 / threads_per_x;
        dim3 threads(threads_per_x, threads_per_y);
        dim3 blocks((image_width + threads.x - 1) / threads.x,
                    (image_height + threads.y - 1) / threads.y);

        switch (score)
        {
        case SUM_OF_ABS_DIFF_ALL:
            fast_gpu_calc_corner_response_kernel<SUM_OF_ABS_DIFF_ALL><<<blocks, threads, 0, stream>>>(
                image_width,
                image_height,
                image_pitch,
                d_image,
                horizontal_border,
                vertical_border,
                d_corner_lut,
                threshold,
                min_arc_length,
                response_pitch_elements,
                d_response);
            break;
        case SUM_OF_ABS_DIFF_ON_ARC:
            fast_gpu_calc_corner_response_kernel<SUM_OF_ABS_DIFF_ON_ARC><<<blocks, threads, 0, stream>>>(
                image_width,
                image_height,
                image_pitch,
                d_image,
                horizontal_border,
                vertical_border,
                d_corner_lut,
                threshold,
                min_arc_length,
                response_pitch_elements,
                d_response);
            break;
        case MAX_THRESHOLD:
            fast_gpu_calc_corner_response_kernel<MAX_THRESHOLD><<<blocks, threads, 0, stream>>>(
                image_width,
                image_height,
                image_pitch,
                d_image,
                horizontal_border,
                vertical_border,
                d_corner_lut,
                threshold,
                min_arc_length,
                response_pitch_elements,
                d_response);
            break;
        }
        // checkCudaErrors(cudaStreamSynchronize(stream));

    }

    void detect(std::vector<pyramid_t> pyramid,
                const unsigned char *d_corner_lut,
                const float threshold,
                float2 *d_pos,
                float *d_score,
                int *d_level,
                cudaStream_t stream)
    {
        for (std::size_t level = 0; level < pyramid.size(); level++)
        {
            fast_gpu_calc_corner_response(pyramid[level].image_width,
                                          pyramid[level].image_height,
                                          pyramid[level].image_pitch,
                                          pyramid[level].image,
                                          3,
                                          3,
                                          d_corner_lut,
                                          threshold,
                                          SUM_OF_ABS_DIFF_ON_ARC,
                                          FAST_SCORE,
                                          pyramid[level].response_pitch / sizeof(float),
                                          pyramid[level].response,
                                          stream);
        }

        //reset score to 0 if depth is 0

        // std::cout << "<---- grid_nms" << std::endl;
        grid_nms(pyramid,
                 d_pos,
                 d_score,
                 d_level,
                 stream);
    }
} // namespace Jetracer
