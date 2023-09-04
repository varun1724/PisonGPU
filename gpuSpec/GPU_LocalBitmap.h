#ifndef GPU_LOCALBITMAP_H
#define GPU_LOCALBITMAP_H
#include <string>
#include <iostream>
#include <vector>
#include <bitset>
#include <cassert>
#include <stack>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <math.h>
#include <immintrin.h>
#include "../general/Bitmap.h"
#include "../general/Tokenizer.h"

typedef struct {
        unsigned char data[16];
      } uchar16;


class GPULocalBitmap : public Bitmap {
    friend class GPUParallelBitmap;
    friend class GPUParallelBitmapIterator;
    friend class GPUParallelBitmapConstructor;
  // All need to become
  private:
    int mNumTknErr;
    int mNumTrial;
    int mThreadId;
    char* mRecord;
    // for a single large record, stream length equals to record length
    long mRecordLength;
    // each temp word has 32 bytes
    long mNumTmpWords;
    // each word has 64 bytes
    long mNumWords;
    // the deepest level of leveled bitmap indexes (starting from 0)
    int mDepth;
    // structural character bitmaps
    unsigned long *mEscapeBitmap, *mStrBitmap, *mColonBitmap, *mCommaBitmap, *mLbracketBitmap, *mRbracketBitmap, *mLbraceBitmap, *mRbraceBitmap;

    // following two variables are used for validating inference results of Step 3 (build string mask bitmap)
    // marks whether current chunk starts inside string or not
    unsigned long mStartInStrBitmap;
    // marks whether current chunk ends inside string or not
    unsigned long mEndInStrBitmap;

    // following variables are used for merging phase (after Step 5, merge leveled bitmap)
    // each thread starts with level 0, following two arrays save bitmaps for levels higher than 0 (temporary result)
    unsigned long *mLevColonBitmap[MAX_LEVEL];
    unsigned long *mLevCommaBitmap[MAX_LEVEL];
    // each thread starts with level 0, following two arrays save bitmaps for levels less than 0 (temporary result)
    unsigned long *mNegLevColonBitmap[MAX_LEVEL];
    unsigned long *mNegLevCommaBitmap[MAX_LEVEL];
    // each thread starts with level 0
    // mMaxPositiveLevel saves the maximum positive level in current thread
    int mMaxPositiveLevel;
    // mMinNegativeLevel saves the minimum negative level in current thread
    int mMinNegativeLevel;
    // saves the level after processing the whole chunk, used for parallel index construction
    int mEndLevel;

    // following variables are used by ParallelBitmapIterator
    // temporary leveled colon bitmap is mapped to the correct level, which happens during the merging phase
    unsigned long *mFinalLevColonBitmap[MAX_LEVEL];
    unsigned long *mFinalLevCommaBitmap[MAX_LEVEL];
    // structural quote bitmap, used for getting the key field when iterating bitmaps
    unsigned long *mQuoteBitmap;
    // word ids for the first and last words, often used when iterating leveled bitmap to get some information like colon, comma and key field positions
    long mStartWordId;
    long mEndWordId;

  public:
    GPULocalBitmap();
    GPULocalBitmap(char* record, int level_num);
    ~GPULocalBitmap();
    // context inference for parallel index construction (step 3).
    // if it context information couldn't be inferred, return SPECULATIVE; else return NOSPECULATIVE
    int contextInference();
    // function for non-speculative parallel index construction
    __device__ void nonSpecIndexConstruction();
    // following two functions are used for speculative parallel index construction
    __device__ void buildStringMaskBitmap();
    __device__ void buildLeveledBitmap();
    void setRecordLength(long length);
    void setThreadId(int thread_id) {mThreadId = thread_id;}

  private:
    void freeMemory();
    __device__ void store(uchar16 vector, uint16_t* v_data);
    __device__ unsigned int uchar16_movemask(const uchar16& vector);
    __device__ unsigned long clmul64(unsigned long num1, unsigned long num2);
    __device__ uchar16 make_uchar16(unsigned char value);
    __device__ uchar16 bitwiseAnd(const uchar16& v1, const uchar16& v2);
    __device__ bool addWithOverflowCheck(unsigned long a, unsigned long b, unsigned long* result);
};

#endif
