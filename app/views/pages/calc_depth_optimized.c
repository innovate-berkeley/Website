/*
 * Project 2: Performance Optimization
 */

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "calc_depth_naive.h"
#include "calc_depth_optimized.h"
#include "utils.h"

#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

void calc_depth_optimized(float *depth, float *left, float *right,
        int image_width, int image_height, int feature_width,
        int feature_height, int maximum_displacement) {
    //VROOM VROOM
    int diff_image_feat_height = image_height - feature_height;
    int diff_image_feat_width = image_width - feature_width;
    #pragma omp parallel
    {
    #pragma omp for
    for (int y = 0; y < image_height; y++) {
        int prod_y_image_width = y * image_width;
        int sum_y_feat_height = y + feature_height;
        int diff_y_feat_height = y - feature_height;
        for (int x = 0; x < image_width; x++) {
            int depth_index = prod_y_image_width + x;
            if (y < feature_height || y >= diff_image_feat_height
                    || x < feature_width || x >= diff_image_feat_width) {
                depth[depth_index] = 0;
                continue;
            }
            float min_diff = -1;
            int sum_x_feat_width = x + feature_width;
            int diff_x_feat_width = x - feature_width;
            int min_displace = 0;
            for (int dy = -maximum_displacement; dy <= maximum_displacement; dy++) {
                int dy_and_diff = dy + diff_y_feat_height;
                int dy_and_sum = dy + sum_y_feat_height;
                for (int dx = -maximum_displacement; dx <= maximum_displacement; dx++) {
                    if (dy_and_diff < 0
                        || dy_and_sum >= image_height
                        || diff_x_feat_width + dx < 0
                        || sum_x_feat_width + dx >= image_width) {
                        continue;
                    }
                    float squared_diff = 0;

                    int bound_x = x - feature_width + ((feature_width + feature_width + 1) / 4 * 4);
                    __m128 sum_vector = _mm_setzero_ps();

                    for (int left_y = y - feature_height; left_y <= y + feature_height; left_y++) {
                        int left_y_prod = image_width * left_y;
                        int right_y_prod = image_width * (left_y + dy);

                        for (int left_x = x - feature_width; left_x < bound_x; left_x += 4) {
                            __m128 a = _mm_loadu_ps((float const*) (left + left_y_prod + left_x));
                            __m128 b = _mm_loadu_ps((float const*) (right + right_y_prod + left_x + dx));
                            __m128 diff = _mm_sub_ps(a, b);
                            __m128 squared = _mm_mul_ps(diff, diff);
                            sum_vector = _mm_add_ps(sum_vector, squared);
                        }
                            for (int left_x = bound_x; left_x <= x + feature_width; left_x++) {
                                int right_x = left_x + dx;
                                float unsquare_diff = left[left_y_prod + left_x] - right[right_y_prod + right_x];
                                squared_diff +=  unsquare_diff * unsquare_diff;
                            }
                      }

                      float sum_vals[4];
                      _mm_storeu_ps((float*) &sum_vals, sum_vector);
                      squared_diff += sum_vals[0] + sum_vals[1] + sum_vals[2] + sum_vals[3];


                    int curr_displace = (dx * dx) + (dy * dy);
                    if (min_diff == -1 || min_diff > squared_diff
                        || (min_diff == squared_diff
                            //Changed to non naive version of displacement
                            && curr_displace < min_displace)) {
                        min_diff = squared_diff;
                        min_displace = curr_displace;
                    }
                }
            }
            if (min_diff != -1) {
                if (maximum_displacement == 0) {
                    depth[depth_index] = 0;
                } else {
                    //Changed to non naive version of displacement
                    depth[depth_index] = (float) sqrt(min_displace);
                }
            } else {
                depth[depth_index] = 0;
            }
        }
    }
  }
}
