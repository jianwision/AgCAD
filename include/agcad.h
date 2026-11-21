/* AgCAD C ABI (post-MVP). See docs/spec/01-architecture.md § Project layout. */
#ifndef AGCAD_H
#define AGCAD_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All returned strings must be released with agcad_free_string. */
char* agcad_version(void);
char* agcad_status_name(uint16_t status);              /* NULL if unknown (cf. cudaGetErrorName) */
char* agcad_run_program(const char* requirements_json, const char* agir_program_json);
void  agcad_free_string(char* s);
#ifdef __cplusplus
}
#endif
#endif
