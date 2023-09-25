#ifndef GPUPARALLELBITMAPCONSTRUCTOR_H
#define GPUPARALLELBITMAPCONSTRUCTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <sys/file.h>
#include <unistd.h>
#include "GPU_ParallelBitmap.h"
#include "../general/Records.h"

typedef struct {
        unsigned char data[16];
      } uchar16;

class GPUParallelBitmapConstructor {
private:
    static GPUParallelBitmap* mGPUParallelBitmap;

public:
    static GPUParallelBitmap* construct(Record* record, int thread_num, int level_num = MAX_LEVEL);
};

#endif
