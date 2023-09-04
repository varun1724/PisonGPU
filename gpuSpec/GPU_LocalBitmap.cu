#include "GPU_LocalBitmap.h"
#include <immintrin.h>
#include <emmintrin.h>
#include <string.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/file.h>
#include <vector_types.h>
#include <unistd.h>
#include <unordered_map>
#include "cuda_runtime.h"


GPULocalBitmap::GPULocalBitmap() {

}

GPULocalBitmap::GPULocalBitmap(char* record, int level_num) {
    this->mThreadId = 0;
    this->mRecord = record;
    this->mDepth = level_num - 1;
    this->mStartWordId = 0;
    this->mEndWordId = 0;
    this->mQuoteBitmap = NULL;
    this->mEscapeBitmap = NULL;
    this->mColonBitmap = NULL;
    this->mCommaBitmap = NULL;
    this->mStrBitmap = NULL;
    this->mLbraceBitmap = NULL;
    this->mRbraceBitmap = NULL;
    this->mLbracketBitmap = NULL;
    this->mRbracketBitmap = NULL;
    for (int i = 0; i < MAX_LEVEL; ++i) {
        this->mLevColonBitmap[i] = NULL;
        this->mLevCommaBitmap[i] = NULL;
        this->mNegLevColonBitmap[i] = NULL;
        this->mNegLevCommaBitmap[i] = NULL;
        this->mFinalLevColonBitmap[i] = NULL;
        this->mFinalLevCommaBitmap[i] = NULL;
    }
    this->mStartInStrBitmap = 0ULL;
    this->mEndInStrBitmap = 0ULL;
    this->mMaxPositiveLevel = 0;
    this->mMinNegativeLevel = -1;

    this->mNumTknErr = 0;
    this->mNumTrial = 0;
}

void GPULocalBitmap::freeMemory()
{
    for(int m = 0; m < MAX_LEVEL; ++m){
        if (mLevColonBitmap[m]) {
            free(mLevColonBitmap[m]);
            mLevColonBitmap[m] = NULL;
        }
        if (mLevCommaBitmap[m]) {
            free(mLevCommaBitmap[m]);
            mLevCommaBitmap[m] = NULL;
        }
        if (mNegLevColonBitmap[m]) {
            free(mNegLevColonBitmap[m]);
            mNegLevColonBitmap[m] = NULL;
        }
        if (mNegLevCommaBitmap[m]) {
            free(mNegLevCommaBitmap[m]);
            mNegLevCommaBitmap[m] = NULL;
        }
    }
    if (mQuoteBitmap) {
        free(mQuoteBitmap);
        mQuoteBitmap = NULL;
    }
    if (mEscapeBitmap) {
        free(mEscapeBitmap);
        mEscapeBitmap = NULL;
    }
    if (mStrBitmap) {
        free(mStrBitmap);
        mStrBitmap = NULL;
    }
    if (mColonBitmap) {
        free(mColonBitmap);
        mColonBitmap = NULL;
    }
    if (mCommaBitmap) {
        free(mCommaBitmap);
        mCommaBitmap = NULL;
    }
    if (mLbraceBitmap) {
        free(mLbraceBitmap);
        mLbraceBitmap = NULL;
    }
    if (mRbraceBitmap) {
        free(mRbraceBitmap);
        mRbraceBitmap = NULL;
    }
    if (mLbracketBitmap) {
        free(mLbracketBitmap);
        mLbracketBitmap = NULL;
    }
    if (mRbracketBitmap) {
        free(mRbracketBitmap);
        mRbracketBitmap = NULL;
    }
}

GPULocalBitmap::~GPULocalBitmap()
{
    freeMemory();
}

void GPULocalBitmap::setRecordLength(long length) {
    this->mRecordLength = length;
    this->mNumTmpWords = length / 32;
    this->mNumWords = length / 64;
    this->mQuoteBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
}

int GPULocalBitmap::contextInference() {
    Tokenizer tkn;
    int start_states[2] = {OUT, IN};
    bool getStartState = false;
    int start_state = OUT;
    for (int j = 0; j < 2; ++j) {
        ++mNumTrial;
        int state = start_states[j];
        tkn.createIterator(mRecord, state);
        while (true) {
            int tkn_status = tkn.hasNextToken();
            if (tkn_status == END)
                break;
            if (tkn_status == ERROR) {
                ++mNumTknErr;
                start_state = tkn.oppositeState(state);
                getStartState = true;
                break;
            }
            tkn.nextToken();
        }
        if (getStartState == true) break;
    }
    if (start_state == IN) {
        mStartInStrBitmap = 0xffffffffffffffffULL;
    } else {
        mStartInStrBitmap = 0ULL;
    }
    //cout<<"inference result num of trails: "<<mNumTrial<<" num of token error "<<mNumTknErr<<endl;
    //cout<<"inference result "<<start_state<<" "<<getStartState<<endl;
    if (getStartState == true) return start_state;
    return UNKNOWN;
}
