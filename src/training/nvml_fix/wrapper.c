#define _GNU_SOURCE
#include <dlfcn.h>

/*
 * Wrapper for libnvidia-ml.so.450.142.00 that adds the missing
 * nvmlDeviceGetNvLinkRemoteDeviceType symbol present in newer drivers.
 * 
 * This wrapper is loaded INSTEAD of the real library by setting
 * LD_LIBRARY_PATH to point here before /lib/x86_64-linux-gnu/.
 * All symbols not defined here fall through to the real library.
 */

/* Provide the missing symbol */
int nvmlDeviceGetNvLinkRemoteDeviceType(void *device, unsigned int link, unsigned int *devType) {
    return 13; /* NVML_ERROR_FUNCTION_NOT_FOUND */
}
