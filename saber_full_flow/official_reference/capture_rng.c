/* Local audit instrumentation, not an upstream SABER source file.
 * rng.c alone is compiled with -Drandombytes=official_randombytes.
 * All other upstream sources resolve randombytes to this wrapper.
 */
#include <string.h>
#include "rng.h"

int official_randombytes(unsigned char *x, unsigned long long xlen);
static unsigned char draws[4][32];
static int draw_count;
static int draw_error;

void audit_reset_capture(void) {
    memset(draws, 0, sizeof(draws));
    draw_count = 0;
    draw_error = 0;
}

int randombytes(unsigned char *x, unsigned long long xlen) {
    int result = official_randombytes(x, xlen);
    if (draw_count >= 4 || xlen != 32 || result != RNG_SUCCESS) {
        draw_error = 1;
    } else {
        memcpy(draws[draw_count], x, 32);
    }
    ++draw_count;
    return result;
}

int audit_copy_draws(unsigned char *out) {
    if (draw_count != 4 || draw_error) return -1;
    memcpy(out, draws, sizeof(draws));
    return 0;
}
