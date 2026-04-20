#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

/*
 * Wrapper for libnvidia-ml.so.1 that provides missing symbols
 * for older NVIDIA drivers (e.g. 450.x) that don't have
 * nvmlDeviceGetNvLinkRemoteDeviceType.
 *
 * Compile: gcc -shared -fPIC -o libnvidia-ml.so.1 nvml_wrapper.c -ldl
 * Use: LD_LIBRARY_PATH=<dir_containing_this>:$LD_LIBRARY_PATH
 */

static void* real_nvml = NULL;

static void* get_real_nvml(void) {
    if (!real_nvml) {
        // Try specific paths for the real library
        const char* paths[] = {
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1", 
            "/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/lib64/libnvidia-ml.so.1",
            "libnvidia-ml.so.450.142.00",
            NULL
        };
        for (int i = 0; paths[i]; i++) {
            real_nvml = dlopen(paths[i], RTLD_LAZY | RTLD_NODELETE);
            if (real_nvml) break;
        }
    }
    return real_nvml;
}

// Provide the missing symbol
int nvmlDeviceGetNvLinkRemoteDeviceType(void* device, unsigned int link, unsigned int* devType) {
    // Try to forward to real implementation
    void* real_fn = NULL;
    if (get_real_nvml()) {
        real_fn = dlsym(real_nvml, "nvmlDeviceGetNvLinkRemoteDeviceType");
    }
    if (real_fn) {
        int (*fn)(void*, unsigned int, unsigned int*) = real_fn;
        return fn(device, link, devType);
    }
    return 13; // NVML_ERROR_FUNCTION_NOT_FOUND
}
