#ifndef LLVNR_RUNTIME_H
#define LLVNR_RUNTIME_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void *llvnr_compile(const char *model_directory, const char *cache_directory);
void *llvnr_input(void *model, const char *port, size_t *bytes);
void *llvnr_output(void *model, const char *port, size_t *bytes);
int llvnr_execute(void *model);
void llvnr_release(void *model);
// Same cross-process fcntl lock as the existing LLVNR experiment runner.
int llvnr_acquire_lock(void);
void llvnr_release_lock(int fd);
#ifdef __cplusplus
}
#endif
#endif
