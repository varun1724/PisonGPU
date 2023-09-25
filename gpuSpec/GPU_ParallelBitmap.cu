#include "GPU_ParallelBitmap.h"

GPUParallelBitmap::GPUParallelBitmap(char* record, long rec_len, int thread_num, int depth) {
    mRecord = record;
    mDepth = depth;
    mThreadNum = thread_num;

    // Location of the starting character of the current chunk
    char* start_chunk = record;

    // Calculates the length of each chunk without padding
    mRecordLength = rec_len;
    int chunk_len = rec_len / thread_num;
    if (chunk_len % 64 > 0) {
        chunk_len = chunk_len + 64 - chunk_len % 64;
    }

    // Tracks the current total length and sets the base value of the location array to 0 to make things easier in the GPU function
    int cur_len = 0;

    mParallelMode = NONSPECULATIVE;
    for (int i = 0; i < thread_num; ++i) {

       // Assign memory for the bitmap arrays
        finalColonBitmaps[i] = new unsigned long*[MAX_LEVEL];
        finalCommaBitmaps[i] = new unsigned long*[MAX_LEVEL];
        for (int j = 0; j < MAX_LEVEL; ++j) {
            finalColonBitmaps[i][j] = new unsigned long[1];
            finalCommaBitmaps[i][j] = new unsigned long[1];
        }


        // Assign values for inference results variables
        mStartInStrBitmaps[i] = 0ULL;
        mEndInStrBitmaps[i] = 0ULL;

        if (i < thread_num - 1) {
            int pad_len = 0;
            // escaped backslashes are not separated into different chunks
            while (start_chunk[chunk_len + pad_len - 1] == '\\') {
                pad_len += 64;
            }

           // Assign memory for quote bitmaps
           quoteBitmaps[i] = new unsigned long[(chunk_len + pad_len) / 64];

            // Set ending location of the end of each chunk
            chunkEndLocs[i] = cur_len + chunk_len + pad_len;

            start_chunk = start_chunk + chunk_len + pad_len;
            cur_len += (chunk_len + pad_len);
        } else {
            chunkEndLocs[i] = rec_len;
            quoteBitmaps[i] = new unsigned long[(rec_len - chunkEndLocs[i-1]) / 64];
        }
        // perform context inference and decide whether the program runs in speculative mode
        if (contextInference(i) == UNKNOWN) {
            mParallelMode = SPECULATIVE;
        }
    }
}

int GPUParallelBitmap::contextInference(int chunk) {
    Tokenizer tkn;
    int start_states[2] = {OUT, IN};
    bool getStartState = false;
    int start_state = OUT;
    for (int j = 0; j < 2; ++j) {
        mNumTrials[chunk] += 1;
        int state = start_states[j];
        tkn.createIterator(mRecord, state);
        while (true) {
            int tkn_status = tkn.hasNextToken();
            if (tkn_status == END)
                break;
            if (tkn_status == ERROR) {
                mNumTknErrs[chunk] += 1;
                start_state = tkn.oppositeState(state);
                getStartState = true;
                break;
            }
            tkn.nextToken();
        }
        if (getStartState == true) break;
    }
    if (start_state == IN) {
        mStartInStrBitmaps[chunk] = 0xffffffffffffffffULL;
    } else {
        mStartInStrBitmaps[chunk] = 0ULL;
    }
    //cout<<"inference result num of trails: "<<mNumTrial<<" num of token error "<<mNumTknErr<<endl;
    //cout<<"inference result "<<start_state<<" "<<getStartState<<endl;
    if (getStartState == true) return start_state;
    return UNKNOWN;
}

GPUParallelBitmap::~GPUParallelBitmap() {

    // Step 1: Deallocate memory for each variable
    for (int i = 0; i < MAX_THREAD; ++i) {
        // Step 2: Deallocate memory for finalColonBitmaps and finalCommaBitmaps (two levels of arrays)
        if (finalColonBitmaps[i]) {
            for (int j = 0; j < MAX_LEVEL; ++j) {
                // Step 3: Deallocate memory for the individual elements within the inner arrays
                if (finalColonBitmaps[i][j]) {
                    delete[] finalColonBitmaps[i][j];
                }
                if (finalCommaBitmaps[i][j]) {
                    delete[] finalCommaBitmaps[i][j];
                }
            }
            // Deallocate memory for the inner arrays
            delete[] finalColonBitmaps[i];
            delete[] finalCommaBitmaps[i];
        }

        // Step 2: Deallocate memory for quoteBitmaps (one level of array)
        if (quoteBitmaps[i]) {
            // Step 3: Deallocate memory for the individual elements within the inner array
            delete[] quoteBitmaps[i];
        }
    }

    // Step 1: Deallocate memory for the outer arrays
    delete[] finalColonBitmaps;
    delete[] finalCommaBitmaps;
    delete[] quoteBitmaps;

}

int GPUParallelBitmap::parallelMode() {
    return mParallelMode;
}

void GPUParallelBitmap::setWordIds(int thread_num) {

    long offset = 0;
    long mNumWords = chunkEndLocs[0] / 64;
    offset += mNumWords;

    // Set for first chunk
    mStartWordIds[0] = 0;
    mEndWordIds[0] = offset;

    for (int i = 1; i < thread_num; ++i) {
        mStartWordIds[i] = offset;
        mNumWords = (chunkEndLocs[i] - chunkEndLocs[i-1]) / 64;
        mEndWordIds[i] = offset + mNumWords;
        offset += mNumWords;
    }

}