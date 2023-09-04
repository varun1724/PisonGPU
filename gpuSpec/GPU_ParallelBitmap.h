#ifndef GPUPARALLELBITMAP_H
#define GPUPARALLELBITMAP_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <sys/file.h>
#include <unistd.h>
#include "../general/Bitmap.h"
#include "GPU_LocalBitmap.h"
using namespace std;

#define MAX_THREAD 1024
#define SPECULATIVE 10
#define NONSPECULATIVE 11

class GPUParallelBitmap : public Bitmap {
    friend class GPUParallelBitmapConstructor;
    friend class GPUParallelBitmapIterator;
  public:
    GPULocalBitmap* mBitmaps[MAX_THREAD];
  private:
    int mThreadNum;
    char* mRecord;
    long mRecordLength;
    int mDepth;
    int mParallelMode;

  public:
    GPUParallelBitmap(char* record, int thread_num, int depth);
    GPUParallelBitmap(char* record, long rec_len, int thread_num, int depth);
    ~GPUParallelBitmap();
    void setRecordLength(long length);
    // SPECULATIVE or NONSPECULATIVE
    int parallelMode();
    // validation after step 3
    void rectifyStringMaskBitmaps();
    // validation after step 5
    void mergeBitmaps();
};
#endif

