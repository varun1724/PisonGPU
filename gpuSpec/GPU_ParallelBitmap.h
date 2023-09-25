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
#include "../general/Tokenizer.h"
using namespace std;

#define MAX_THREAD 128
#define SPECULATIVE 10
#define NONSPECULATIVE 11

class GPUParallelBitmap : public Bitmap {
    friend class GPUParallelBitmapConstructor;
    friend class GPUParallelBitmapIterator;
  private:
    int mThreadNum;
    char* mRecord;
    long mRecordLength;
    int mDepth;
    int mParallelMode;

    int chunkEndLocs[MAX_THREAD];

    // Declare size in the constuctor based on the number of threads
    unsigned long*** finalColonBitmaps = new unsigned long**[MAX_THREAD];
    unsigned long*** finalCommaBitmaps = new unsigned long**[MAX_THREAD];
    unsigned long** quoteBitmaps = new unsigned long*[MAX_THREAD];

    unsigned long mStartInStrBitmaps[MAX_THREAD];
    unsigned long mEndInStrBitmaps[MAX_THREAD];

    int mNumTknErrs[MAX_THREAD];
    int mNumTrials[MAX_THREAD];

    long mStartWordIds[MAX_THREAD];
    long mEndWordIds[MAX_THREAD];

    int endLevels[MAX_THREAD];


  public:
    GPUParallelBitmap(char* record, long rec_len, int thread_num, int depth);
    ~GPUParallelBitmap();
    // SPECULATIVE or NONSPECULATIVE
    int parallelMode();
    // Helper function to determine state
    int contextInference(int);
    // Sets the word id values
    void setWordIds(int);
    
};
#endif

