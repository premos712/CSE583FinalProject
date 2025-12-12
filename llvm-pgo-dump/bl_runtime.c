// bl_runtime.c
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
  const char *name;
  uint64_t   *counts;
  uint32_t    npaths;
} __bl_rec;

extern __bl_rec __bl_table[];
extern const uint32_t __bl_table_size;

static void __bl_dump(void) {
  const char *p = getenv("BL_PROFILE_OUT");
  if (!p) p = getenv("BL_PROFILE_PATH");
  if (!p) p = "bl_profile.txt";

  FILE *fp = fopen(p, "w");
  if (!fp) return;

  fprintf(fp, "function,path_id,count\n");
  for (uint32_t i = 0; i < __bl_table_size; ++i) {
    const __bl_rec *r = &__bl_table[i];
    if (!r->name || !r->counts) continue;

    for (uint32_t pid = 0; pid < r->npaths; ++pid) {
      uint64_t c = r->counts[pid];
      if (c == 0) continue;
      fprintf(fp, "%s,%u,%llu\n",
              r->name, pid, (unsigned long long)c);
    }
  }

  fclose(fp);
}

__attribute__((constructor))
static void __bl_init(void) {
  atexit(__bl_dump);
}
