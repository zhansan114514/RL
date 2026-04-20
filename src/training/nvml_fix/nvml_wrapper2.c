/* 
 * Provides only the missing symbol nvmlDeviceGetNvLinkRemoteDeviceType.
 * Load via LD_PRELOAD - this symbol will override/resolve before the real library.
 * All other NVML symbols come from the real libnvidia-ml.so.1 loaded by PyTorch.
 */
int nvmlDeviceGetNvLinkRemoteDeviceType(void* device, unsigned int link, unsigned int* devType) {
    return 13; /* NVML_ERROR_FUNCTION_NOT_FOUND */
}
