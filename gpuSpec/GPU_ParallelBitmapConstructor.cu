#include "GPU_ParallelBitmapConstructor.h"


//////////////////////
//////////////////////
// LOCAL BITMAP FUNCTIONS
//////////////////////
//////////////////////

// Function to store the uchar16 vector into the memory pointed by uint16_t* pointer
__device__ void store(uchar16 vector, uint16_t* v_data) {
    unsigned char* uchar_ptr = (unsigned char*)v_data;
    for (int i = 0; i < 16; i++) {
        uchar_ptr[i] = vector.data[i];
    }
}

__device__ unsigned int uchar16_movemask(const uchar16& vector) {
    // Convert uchar16 to uint16_t (16-bit unsigned integers)
    uint16_t v_data[16];
    store(vector, v_data);

    // Convert 16-bit unsigned integers to a 16-bit mask (unsigned short)
    unsigned short mask = 0;
    for (int i = 0; i < 16; ++i) {
        mask |= (v_data[i] << i);
    }

    // Use CUDA's __popc to count the number of set bits in the mask
    return __popc(mask);
}

// Perform carry-less multiplication
__device__ unsigned long clmul64(unsigned long num1, unsigned long num2) {
    unsigned long result = 0;

    for (int i = 0; i < 64; i++) {
        unsigned long bit = (num2 >> i) & 1;
        result ^= (num1 << i) * bit;
    }

    // Returns the lower 64 bits of the result
    return result & 0xFFFFFFFFFFFFFFFF;
}


// Function to set all values in the uchar16 type to the specified value
__device__ uchar16 make_uchar16(unsigned char value) {
    uchar16 result;
    for (int i = 0; i < 16; i++) {
        result.data[i] = value;
    }
    return result;
}

// Function to perform a bitwise and between two uchar16 types
__device__ uchar16 bitwiseAnd(const uchar16& v1, const uchar16& v2) {
    uchar16 result;
    for (int i = 0; i < 16; i++) {
        result.data[i] = v1.data[i] & v2.data[i];
    }
    return result;
}

// Checks for overflow
__device__ bool addWithOverflowCheck(unsigned long a, unsigned long b, unsigned long* result) {
    unsigned long long sum = a + b;
    bool overflow = (sum < a) || (sum < b); // Check if the sum is less than any of the operands
    *result = sum;
    return overflow;
}

// Creates the final leveled bitmaps on different threads
// TODO: Edit function to account for flattened inputs
__global__ void nonSpecIndexConstructionKernel(
    int* chunkEndLocs, char* record, int mDepth, unsigned long* finalColonBitmaps, 
    unsigned long* finalCommaBitmaps, unsigned long* quoteBitmaps,  
    unsigned long* mStartInStrBitmaps, int* endLevels) {
        

    // Based on the array that is passed in, the chunk that the record needs to eval is determined
    int recordLength = 0;
    if (threadIdx.x == 0) {
        recordLength = chunkEndLocs[0];
    } else {
        record += chunkEndLocs[threadIdx.x-1];
        recordLength = chunkEndLocs[threadIdx.x] - chunkEndLocs[threadIdx.x-1];
    }

    // Assign number of temp words variables
    long mNumTmpWords = recordLength / 32;
    long mNumWords = recordLength / 64;

    // Variables for tracking maximum positive and minimum negative levels in threads
    int mMaxPositiveLevel = 0;
    int mMinNegativeLevel = -1;

    // Create variables to store temp leveled bitmaps
    // each thread starts with level 0, following two arrays save bitmaps for levels higher than 0 (temporary result)
    unsigned long *mLevColonBitmap[MAX_LEVEL];
    unsigned long *mLevCommaBitmap[MAX_LEVEL];
    // each thread starts with level 0, following two arrays save bitmaps for levels less than 0 (temporary result)
    unsigned long *mNegLevColonBitmap[MAX_LEVEL];
    unsigned long *mNegLevCommaBitmap[MAX_LEVEL];


    // vectors for structural characters
    // Creates a 128 bit vector type containing 4 32 bit unsigned integers
    // Front bits
    uchar16 v_quote0 = make_uchar16(0x22);
    uchar16 v_colon0 = make_uchar16(0x3a);
    uchar16 v_escape0 = make_uchar16(0x5c);
    uchar16 v_lbrace0 = make_uchar16(0x7b);
    uchar16 v_rbrace0 = make_uchar16(0x7d);
    uchar16 v_comma0 = make_uchar16(0x2c);
    uchar16 v_lbracket0 = make_uchar16(0x5b);
    uchar16 v_rbracket0 = make_uchar16(0x5d);
    // Creates the rest of the 128 bit vectors to have 32 bits in total
    // Back bits
    uchar16 v_quote1 = make_uchar16(0x22);
    uchar16 v_colon1 = make_uchar16(0x3a);
    uchar16 v_escape1 = make_uchar16(0x5c);
    uchar16 v_lbrace1 = make_uchar16(0x7b);
    uchar16 v_rbrace1 = make_uchar16(0x7d);
    uchar16 v_comma1 = make_uchar16(0x2c);
    uchar16 v_lbracket1 = make_uchar16(0x5b);
    uchar16 v_rbracket1 = make_uchar16(0x5d);

    // variables for saving temporary results in the first four steps
    unsigned long colonbitT, quotebitT, escapebitT, lbracebitT, rbracebitT, commabitT, lbracketbitT, rbracketbitT;
    unsigned long colonbit, quotebit, escapebit, lbracebit, rbracebit, commabit, lbracketbit, rbracketbit;
    unsigned long colonbit0, quotebit0, escapebit0, lbracebit0, rbracebit0, commabit0, lbracketbit0, rbracketbit0;
    unsigned long colonbit1, quotebit1, escapebit1, lbracebit1, rbracebit1, commabit1, lbracketbit1, rbracketbit1;
    unsigned long str_mask;

    // variables for saving temporary results in the last step
    unsigned long lb_mask, rb_mask, cb_mask;
    unsigned long lb_bit, rb_bit, cb_bit;
    unsigned long first, second;
    int cur_level = -1;

    // variables for saving context information among different words
    int top_word = -1;
    uint64_t prev_iter_ends_odd_backslash = 0ULL;
    uint64_t prev_iter_inside_quote = mStartInStrBitmaps[threadIdx.x];
    const uint64_t even_bits = 0x5555555555555555ULL;
    const uint64_t odd_bits = ~even_bits;


    for (int j = 0; j < mNumTmpWords; ++j) {
        colonbit = 0, quotebit = 0, escapebit = 0, lbracebit = 0, rbracebit = 0, commabit = 0, lbracketbit = 0, rbracketbit = 0;
        unsigned long i = j * 32;
        // step 1: build structural character bitmaps
        uchar16 v_text0 = make_uchar16(0);
        uchar16 v_text1 = make_uchar16(0);

        for (int j = 0; j < sizeof(uchar16); j++) {
            v_text0.data[j] = record[i + j];
            v_text0.data[j] = record[i + 16 + j];
        }

        colonbit0 = uchar16_movemask(bitwiseAnd(v_text0, v_colon0));
        colonbit1 = uchar16_movemask(bitwiseAnd(v_text1, v_colon1));
        colonbit = static_cast<unsigned long>((static_cast<uint32_t>(colonbit0) << 16) | static_cast<uint32_t>(colonbit1));

        quotebit0 = uchar16_movemask(bitwiseAnd(v_text0, v_quote0));
        quotebit1 = uchar16_movemask(bitwiseAnd(v_text1, v_quote1));
        quotebit = static_cast<unsigned long>((static_cast<uint32_t>(quotebit0) << 16) | static_cast<uint32_t>(quotebit1));

        escapebit0 = uchar16_movemask(bitwiseAnd(v_text0, v_escape0));
        escapebit1 = uchar16_movemask(bitwiseAnd(v_text1, v_escape1));
        escapebit = static_cast<unsigned long>((static_cast<uint32_t>(escapebit0) << 16) | static_cast<uint32_t>(escapebit1));

        lbracebit0 = uchar16_movemask(bitwiseAnd(v_text0, v_lbrace0));
        lbracebit1 = uchar16_movemask(bitwiseAnd(v_text1, v_lbrace1));
        lbracebit = static_cast<unsigned long>((static_cast<uint32_t>(lbracebit0) << 16) | static_cast<uint32_t>(lbracebit1));

        rbracebit0 = uchar16_movemask(bitwiseAnd(v_text0, v_rbrace0));
        rbracebit1 = uchar16_movemask(bitwiseAnd(v_text1, v_rbrace1));
        rbracebit = static_cast<unsigned long>((static_cast<uint32_t>(rbracebit0) << 16) | static_cast<uint32_t>(rbracebit1));

        commabit0 = uchar16_movemask(bitwiseAnd(v_text0, v_comma0));
        commabit1 = uchar16_movemask(bitwiseAnd(v_text1, v_comma1));
        commabit = static_cast<unsigned long>((static_cast<uint32_t>(commabit0) << 16) | static_cast<uint32_t>(commabit1));

        lbracketbit0 = uchar16_movemask(bitwiseAnd(v_text0, v_lbracket0));
        lbracketbit1 = uchar16_movemask(bitwiseAnd(v_text1, v_lbracket1));
        lbracketbit = static_cast<unsigned long>((static_cast<uint32_t>(lbracketbit0) << 16) | static_cast<uint32_t>(lbracketbit1));

        rbracketbit0 = uchar16_movemask(bitwiseAnd(v_text0, v_rbracket0));
        rbracketbit1 = uchar16_movemask(bitwiseAnd(v_text1, v_rbracket1));
        rbracketbit = static_cast<unsigned long>((static_cast<uint32_t>(rbracketbit0) << 16) | static_cast<uint32_t>(rbracketbit1));

        // first half of the word (lowest 32 bits)
        if (j % 2 == 0) {
            colonbitT = colonbit;
            quotebitT = quotebit;
            escapebitT = escapebit;
            lbracebitT = lbracebit;
            rbracebitT = rbracebit;
            commabitT = commabit;
            lbracketbitT = lbracketbit;
            rbracketbitT = rbracketbit;
            continue;
        } else {
            // highest 32 bits inside a word
            colonbit = (colonbit << 32) | colonbitT;
            quotebit = (quotebit << 32) | quotebitT;
            escapebit = (escapebit << 32) | escapebitT;
            lbracebit = (lbracebit << 32) | lbracebitT;
            rbracebit = (rbracebit << 32) | rbracebitT;
            commabit = (commabit << 32) | commabitT;
            lbracketbit = (lbracketbit << 32) | lbracketbitT;
            rbracketbit = (rbracketbit << 32) | rbracketbitT;

            // step 2: update structural quote bitmaps
            uint64_t bs_bits = escapebit;
            uint64_t start_edges = bs_bits & ~(bs_bits << 1);
            int64_t even_start_mask = even_bits ^ prev_iter_ends_odd_backslash;
            uint64_t even_starts = start_edges & even_start_mask;
            uint64_t odd_starts = start_edges & ~even_start_mask;
            uint64_t even_carries = bs_bits + even_starts;
            int64_t odd_carries;
            bool iter_ends_odd_backslash = addWithOverflowCheck(bs_bits, odd_starts, (unsigned long *)&odd_carries);
            odd_carries |= prev_iter_ends_odd_backslash;
            prev_iter_ends_odd_backslash = iter_ends_odd_backslash ? 0x1ULL : 0x0ULL;
            uint64_t even_carry_ends = even_carries & ~bs_bits;
            uint64_t odd_carry_ends = odd_carries & ~bs_bits;
            uint64_t even_start_odd_end = even_carry_ends & odd_bits;
            uint64_t odd_start_even_end = odd_carry_ends & even_bits;
            uint64_t odd_ends = even_start_odd_end | odd_start_even_end;
            int64_t quote_bits = quotebit & ~odd_ends;


            top_word += 1;
            int startPos = 0;
            if (threadIdx.x != 0) {
                startPos = chunkEndLocs[threadIdx.x-1] / 64;
            }
            quoteBitmaps[startPos + top_word] = static_cast<unsigned long>(quote_bits);

            unsigned long long allOnes64Bit = ULLONG_MAX;
            str_mask = clmul64(quote_bits, allOnes64Bit);
            str_mask ^= prev_iter_inside_quote;
            prev_iter_inside_quote = static_cast<uint64_t>(static_cast<int64_t>(str_mask) >> 63);

            // step 4: update structural character bitmaps
            unsigned long tmp = (~str_mask);
            colonbit = colonbit & tmp;
            lbracebit = lbracebit & tmp;
            rbracebit = rbracebit & tmp;
            commabit = commabit & tmp;
            lbracketbit = lbracketbit & tmp;
            rbracketbit = rbracketbit & tmp;

            // step 5: generate leveled bitmaps
            lb_mask = lbracebit | lbracketbit;
            rb_mask = rbracebit | rbracketbit;
            cb_mask = lb_mask | rb_mask;
            lb_bit = lb_mask & (-lb_mask);
            rb_bit = rb_mask & (-rb_mask);
            if (!cb_mask) {
                if (cur_level >= 0 && cur_level <= mDepth) {
                    if (!mLevColonBitmap[cur_level]) {
                        mLevColonBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                    }
                    if (!mLevCommaBitmap[cur_level]) {
                        mLevCommaBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                    }
                    if (colonbit) {
                        mLevColonBitmap[cur_level][top_word] = colonbit;
                    } else {
                        mLevCommaBitmap[cur_level][top_word] = commabit;
            }
        } else if (cur_level < 0) {
                    if (!mNegLevColonBitmap[-cur_level]) {
                        mNegLevColonBitmap[-cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                        // before finding the first bracket, update minimum negative level
                        if (cur_level < mMinNegativeLevel) {
                            mMinNegativeLevel = cur_level;
                        }
                    }
                    if (!mNegLevCommaBitmap[-cur_level]) {
                        mNegLevCommaBitmap[-cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                    }
                    if (colonbit) {
                        mNegLevColonBitmap[-cur_level][top_word] = colonbit;
                    } else {
                        mNegLevCommaBitmap[-cur_level][top_word] = commabit;
                    }
                }
            } else {
                first = 1;
                while (cb_mask || first) {
                    if (!cb_mask) {
                        second = 1UL<<63;
                    } else {
                        cb_bit = cb_mask & (-cb_mask);
                        second = cb_bit;
                    }
                    if (cur_level >= 0 && cur_level <= mDepth) {
                        if (!mLevColonBitmap[cur_level]) {
                            mLevColonBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                        }
                        if (!mLevCommaBitmap[cur_level]) {
                            mLevCommaBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                        }
                        unsigned long mask = second - first;
                        if (!cb_mask) mask = mask | second;
                        unsigned long colon_mask = mask & colonbit;
                        if (colon_mask) {
                            mLevColonBitmap[cur_level][top_word] |= colon_mask;
                        } else {
                            mLevCommaBitmap[cur_level][top_word] |= (commabit & mask);
                        }
                        if (cb_mask) {
                            if (cb_bit == rb_bit) {
                                mLevColonBitmap[cur_level][top_word] |= cb_bit;
                                mLevCommaBitmap[cur_level][top_word] |= cb_bit;
                            }
                            else if (cb_bit == lb_bit && cur_level + 1 <= mDepth) {
                                if (!mLevCommaBitmap[cur_level + 1]) {
                                    mLevCommaBitmap[cur_level + 1] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                                }
                                mLevCommaBitmap[cur_level + 1][top_word] |= cb_bit;
                            }
                        }
                    } else if (cur_level < 0) {
                        if (!mNegLevColonBitmap[-cur_level]) {
                            mNegLevColonBitmap[-cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                        }
                        if (!mNegLevCommaBitmap[-cur_level]) {
                            mNegLevCommaBitmap[-cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                        }
                        unsigned long mask = second - first;
                        if (!cb_mask) mask = mask | second;
                        unsigned long colon_mask = mask & colonbit;
                        if (colon_mask) {
                            mNegLevColonBitmap[-cur_level][top_word] |= colon_mask;
                        } else {
                            mNegLevCommaBitmap[-cur_level][top_word] |= (commabit & mask);
                        }
                        if (cb_mask) {
                            if (cb_bit == rb_bit) {
                                mNegLevColonBitmap[-cur_level][top_word] |= cb_bit;
                                mNegLevCommaBitmap[-cur_level][top_word] |= cb_bit;
                            }
                            else if (cb_bit == lb_bit) {
                                if (cur_level + 1 == 0) {
                                    if (!mLevCommaBitmap[0]) {
                                        mLevCommaBitmap[0] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                                    }
                                    mLevCommaBitmap[0][top_word] |= cb_bit;
                                } else {
                                    if (!mNegLevCommaBitmap[-(cur_level + 1)]) {
                                        mNegLevCommaBitmap[-(cur_level + 1)] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                                    }
                                    mNegLevCommaBitmap[-(cur_level + 1)][top_word] |= cb_bit;
                                }
                            }
                        }
                    }
                    if (cb_mask) {
                        if (cb_bit == lb_bit) {
                            lb_mask = lb_mask & (lb_mask - 1);
                            lb_bit = lb_mask & (-lb_mask);
                            ++cur_level;
                            if (threadIdx.x == 0 && cur_level == 0) {
                                // JSON record at the top level could be an array
                                if (!mLevCommaBitmap[cur_level]) {
                                    mLevCommaBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                                }
                                mLevCommaBitmap[cur_level][top_word] |= cb_bit;
                            }
                        } else if (cb_bit == rb_bit) {
                            rb_mask = rb_mask & (rb_mask - 1);
                            rb_bit = rb_mask & (-rb_mask);
                            --cur_level;
                        }
                        first = second;
                        cb_mask = cb_mask & (cb_mask - 1);
                        if (cur_level > mMaxPositiveLevel) {
                            mMaxPositiveLevel = cur_level;
                        } else if (cur_level < mMinNegativeLevel) {
                            mMinNegativeLevel = cur_level;
                        }
                    } else {
                        first = 0;
                    }
                }
        }
        }
    }
    if (mDepth == MAX_LEVEL - 1) mDepth = mMaxPositiveLevel;
    endLevels[threadIdx.x] = cur_level;
    
    __syncthreads();

    // TODO: Merge correctly
    // Merge bitmaps for each chunk into final bitmaps
    int curLevel = endLevels[0];

    // if (threadIdx.x != 0) {
    //     for (int i = 0; i <= mMaxPositiveLevel; ++i) {
    //         *finalColonBitmaps[threadIdx.x][i] = mLevColonBitmap[i];
    //         *finalCommaBitmaps[threadIdx.x][i] = mLevCommaBitmap[i];
    //     }
    // } else {
    //     // Set cur_level to the correct value
    //     for (int i = 1; i < threadIdx.x; ++i) {
    //         curLevel += (endLevels[i] + 1);
    //     }

    //     for(int j = 1; j <= -mMinNegativeLevel && (curLevel - j + 1) >= 0; ++j) {
    //         *finalColonBitmaps[threadIdx.x][curLevel - j + 1] = mNegLevColonBitmap[j];
    //         *finalCommaBitmaps[threadIdx.x][curLevel - j + 1] = mNegLevCommaBitmap[j];
    //     }
    //     for(int j = 0; j <= mMaxPositiveLevel && (curLevel + j + 1) >= 0; ++j) {
    //         *finalColonBitmaps[threadIdx.x][curLevel + j + 1] = mLevColonBitmap[j];
    //         *finalCommaBitmaps[threadIdx.x][curLevel + j + 1] = mLevCommaBitmap[j];
    //     }
    // }

    if (threadIdx.x != 0) {
        for (int i = 0; i <= mMaxPositiveLevel; ++i) {
            finalColonBitmaps[threadIdx.x * (mMaxPositiveLevel + 1) + i] = *mLevColonBitmap[i];
            finalCommaBitmaps[threadIdx.x * (mMaxPositiveLevel + 1) + i] = *mLevCommaBitmap[i];
        }
    } else {
        // Set cur_level to the correct value
        for (int i = 1; i < threadIdx.x; ++i) {
            curLevel += (endLevels[i] + 1);
        }

        for (int j = 1; j <= -mMinNegativeLevel && (curLevel - j + 1) >= 0; ++j) {
            finalColonBitmaps[threadIdx.x * (mMaxPositiveLevel + 1) + curLevel - j + 1] = *mNegLevColonBitmap[j];
            finalCommaBitmaps[threadIdx.x * (mMaxPositiveLevel + 1) + curLevel - j + 1] = *mNegLevCommaBitmap[j];
        }
        for (int j = 0; j <= mMaxPositiveLevel && (curLevel + j + 1) >= 0; ++j) {
            finalColonBitmaps[threadIdx.x * (mMaxPositiveLevel + 1) + curLevel + j + 1] = *mLevColonBitmap[j];
            finalCommaBitmaps[threadIdx.x * (mMaxPositiveLevel + 1) + curLevel + j + 1] = *mLevCommaBitmap[j];
        }
    }
    
    // Free all temp bitmaps that were created
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

    return;

}



///////////////////////////////////
///////////////////////////////////
/// BUILD LEVELED BITMAPS FUNCTION
/// For speculative mode
///////////////////////////////////
///////////////////////////////////

__global__ void specIndexConstructionKernel(
    int* chunkEndLocs, char* record, int mDepth, unsigned long**** finalColonBitmaps, 
    unsigned long**** finalCommaBitmaps, unsigned long*** quoteBitmaps,  
    unsigned long* mStartInStrBitmaps, unsigned long* mEndInStrBitmaps, int* endLevels) {

    // Based on the array that is passed in, the chunk that the record needs to eval is determined
    int recordLength = 0;
    if (threadIdx.x == 0) {
        recordLength = chunkEndLocs[0];
    } else {
        record += chunkEndLocs[threadIdx.x-1];
        recordLength = chunkEndLocs[threadIdx.x] - chunkEndLocs[threadIdx.x-1];
    }

    // Assign number of temp words variables
    long mNumTmpWords = recordLength / 32;
    long mNumWords = recordLength / 64;

    // Variables for tracking maximum positive and minimum negative levels in threads
    int mMaxPositiveLevel = 0;
    int mMinNegativeLevel = -1;

    // Create variables to store temp leveled bitmaps
    // each thread starts with level 0, following two arrays save bitmaps for levels higher than 0 (temporary result)
    unsigned long *mLevColonBitmap[MAX_LEVEL];
    unsigned long *mLevCommaBitmap[MAX_LEVEL];
    // each thread starts with level 0, following two arrays save bitmaps for levels less than 0 (temporary result)
    unsigned long *mNegLevColonBitmap[MAX_LEVEL];
    unsigned long *mNegLevCommaBitmap[MAX_LEVEL];
    

    // allocate memory space for saving results
    unsigned long* mColonBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    unsigned long* mCommaBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    unsigned long* mStrBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    unsigned long* mLbraceBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    unsigned long* mRbraceBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    unsigned long* mLbracketBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    unsigned long* mRbracketBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    

    // vectors for structural characters
    // Creates a 128 bit vector type containing 4 32 bit unsigned integers
    // Front bits
    uchar16 v_quote0 = make_uchar16(0x22);
    uchar16 v_colon0 = make_uchar16(0x3a);
    uchar16 v_escape0 = make_uchar16(0x5c);
    uchar16 v_lbrace0 = make_uchar16(0x7b);
    uchar16 v_rbrace0 = make_uchar16(0x7d);
    uchar16 v_comma0 = make_uchar16(0x2c);
    uchar16 v_lbracket0 = make_uchar16(0x5b);
    uchar16 v_rbracket0 = make_uchar16(0x5d);
    // Creates the rest of the 128 bit vectors to have 32 bits in total
    // Back bits
    uchar16 v_quote1 = make_uchar16(0x22);
    uchar16 v_colon1 = make_uchar16(0x3a);
    uchar16 v_escape1 = make_uchar16(0x5c);
    uchar16 v_lbrace1 = make_uchar16(0x7b);
    uchar16 v_rbrace1 = make_uchar16(0x7d);
    uchar16 v_comma1 = make_uchar16(0x2c);
    uchar16 v_lbracket1 = make_uchar16(0x5b);
    uchar16 v_rbracket1 = make_uchar16(0x5d);


    // variables for saving temporary results
    unsigned long colonbitT, quotebitT, escapebitT, lbracebitT, rbracebitT, commabitT, lbracketbitT, rbracketbitT;
    unsigned long colonbit, quotebit, escapebit, lbracebit, rbracebit, commabit, lbracketbit, rbracketbit;
    unsigned long colonbit0, quotebit0, escapebit0, lbracebit0, rbracebit0, commabit0, lbracketbit0, rbracketbit0;
    unsigned long colonbit1, quotebit1, escapebit1, lbracebit1, rbracebit1, commabit1, lbracketbit1, rbracketbit1;
    unsigned long str_mask;

    // variables for saving context information among different words
    int top_word = -1;
    uint64_t prev_iter_ends_odd_backslash = 0ULL;
    uint64_t prev_iter_inside_quote = mStartInStrBitmaps[threadIdx.x];
    const uint64_t even_bits = 0x5555555555555555ULL;
    const uint64_t odd_bits = ~even_bits;

    for (int j = 0; j < mNumTmpWords; ++j) {
        colonbit = 0, quotebit = 0, escapebit = 0, lbracebit = 0, rbracebit = 0, commabit = 0, lbracketbit = 0, rbracketbit = 0;
        unsigned long i = j * 32;
        // step 1: build structural character bitmaps
        uchar16 v_text0 = make_uchar16(0);
        uchar16 v_text1 = make_uchar16(0);

        for (int j = 0; j < sizeof(uchar16); j++) {
            v_text0.data[j] = record[i + j];
            v_text0.data[j] = record[i + 16 + j];
        }

        colonbit0 = uchar16_movemask(bitwiseAnd(v_text0, v_colon0));
        colonbit1 = uchar16_movemask(bitwiseAnd(v_text1, v_colon1));
        colonbit = static_cast<unsigned long>((static_cast<uint32_t>(colonbit0) << 16) | static_cast<uint32_t>(colonbit1));

        quotebit0 = uchar16_movemask(bitwiseAnd(v_text0, v_quote0));
        quotebit1 = uchar16_movemask(bitwiseAnd(v_text1, v_quote1));
        quotebit = static_cast<unsigned long>((static_cast<uint32_t>(quotebit0) << 16) | static_cast<uint32_t>(quotebit1));

        escapebit0 = uchar16_movemask(bitwiseAnd(v_text0, v_escape0));
        escapebit1 = uchar16_movemask(bitwiseAnd(v_text1, v_escape1));
        escapebit = static_cast<unsigned long>((static_cast<uint32_t>(escapebit0) << 16) | static_cast<uint32_t>(escapebit1));

        lbracebit0 = uchar16_movemask(bitwiseAnd(v_text0, v_lbrace0));
        lbracebit1 = uchar16_movemask(bitwiseAnd(v_text1, v_lbrace1));
        lbracebit = static_cast<unsigned long>((static_cast<uint32_t>(lbracebit0) << 16) | static_cast<uint32_t>(lbracebit1));

        rbracebit0 = uchar16_movemask(bitwiseAnd(v_text0, v_rbrace0));
        rbracebit1 = uchar16_movemask(bitwiseAnd(v_text1, v_rbrace1));
        rbracebit = static_cast<unsigned long>((static_cast<uint32_t>(rbracebit0) << 16) | static_cast<uint32_t>(rbracebit1));

        commabit0 = uchar16_movemask(bitwiseAnd(v_text0, v_comma0));
        commabit1 = uchar16_movemask(bitwiseAnd(v_text1, v_comma1));
        commabit = static_cast<unsigned long>((static_cast<uint32_t>(commabit0) << 16) | static_cast<uint32_t>(commabit1));

        lbracketbit0 = uchar16_movemask(bitwiseAnd(v_text0, v_lbracket0));
        lbracketbit1 = uchar16_movemask(bitwiseAnd(v_text1, v_lbracket1));
        lbracketbit = static_cast<unsigned long>((static_cast<uint32_t>(lbracketbit0) << 16) | static_cast<uint32_t>(lbracketbit1));

        rbracketbit0 = uchar16_movemask(bitwiseAnd(v_text0, v_rbracket0));
        rbracketbit1 = uchar16_movemask(bitwiseAnd(v_text1, v_rbracket1));
        rbracketbit = static_cast<unsigned long>((static_cast<uint32_t>(rbracketbit0) << 16) | static_cast<uint32_t>(rbracketbit1));
        // first half of the word (lowest 32 bits)
        if (j % 2 == 0) {
            colonbitT = colonbit;
            quotebitT = quotebit;
            escapebitT = escapebit;
            lbracebitT = lbracebit;
            rbracebitT = rbracebit;
            commabitT = commabit;
            lbracketbitT = lbracketbit;
            rbracketbitT = rbracketbit;
            continue;
        } else {
            // highest 32 bits inside a word
            colonbit = (colonbit << 32) | colonbitT;
            quotebit = (quotebit << 32) | quotebitT;
            escapebit = (escapebit << 32) | escapebitT;
            lbracebit = (lbracebit << 32) | lbracebitT;
            rbracebit = (rbracebit << 32) | rbracebitT;
            commabit = (commabit << 32) | commabitT;
            lbracketbit = (lbracketbit << 32) | lbracketbitT;
            rbracketbit = (rbracketbit << 32) | rbracketbitT;
            mColonBitmap[++top_word] = colonbit;
            mCommaBitmap[top_word] = commabit;
            mLbraceBitmap[top_word] = lbracebit;
            mRbraceBitmap[top_word] = rbracebit;
            mLbracketBitmap[top_word] = lbracketbit;
            mRbracketBitmap[top_word] = rbracketbit;

            // step 2: update structural quote bitmaps
            uint64_t bs_bits = escapebit;
            uint64_t start_edges = bs_bits & ~(bs_bits << 1);
            int64_t even_start_mask = even_bits ^ prev_iter_ends_odd_backslash;
            uint64_t even_starts = start_edges & even_start_mask;
            uint64_t odd_starts = start_edges & ~even_start_mask;
            uint64_t even_carries = bs_bits + even_starts;
            int64_t odd_carries;
            bool iter_ends_odd_backslash = addWithOverflowCheck(bs_bits, odd_starts, (unsigned long *)&odd_carries);
            odd_carries |= prev_iter_ends_odd_backslash;
            prev_iter_ends_odd_backslash = iter_ends_odd_backslash ? 0x1ULL : 0x0ULL;
            uint64_t even_carry_ends = even_carries & ~bs_bits;
            uint64_t odd_carry_ends = odd_carries & ~bs_bits;
            uint64_t even_start_odd_end = even_carry_ends & odd_bits;
            uint64_t odd_start_even_end = odd_carry_ends & even_bits;
            uint64_t odd_ends = even_start_odd_end | odd_start_even_end;
            int64_t quote_bits = quotebit & ~odd_ends;
            *quoteBitmaps[threadIdx.x][top_word] = static_cast<unsigned long>(quote_bits);

            // step 3: build string mask bitmaps
            unsigned long long allOnes64Bit = ULLONG_MAX;
            str_mask = clmul64(quote_bits, allOnes64Bit);
            str_mask ^= prev_iter_inside_quote;
            mStrBitmap[top_word] = str_mask;
            prev_iter_inside_quote = static_cast<uint64_t>(static_cast<int64_t>(str_mask) >> 63);
        }
    }
    mEndInStrBitmaps[threadIdx.x] = prev_iter_inside_quote;

    // Sync threads to be able to access into all arrays
    __syncthreads();

    // Rectifty bitmaps logic
    prev_iter_inside_quote = mEndInStrBitmaps[0];
    if (threadIdx.x > 1) {
        prev_iter_inside_quote = mEndInStrBitmaps[threadIdx.x-1];
    }
    if (prev_iter_inside_quote != mStartInStrBitmaps[threadIdx.x] && threadIdx.x > 0) {
        mStartInStrBitmaps[threadIdx.x] = prev_iter_inside_quote;
        // flip string mask bitmaps
        //cout<<"flip for "<<i<<"th thread "<<endl;
        for (int j = 0; j < mNumWords; ++j) {
            mStrBitmap[j] = ~mStrBitmap[j];
        }
        if (mEndInStrBitmaps[threadIdx.x] == 0) {
            mEndInStrBitmaps[threadIdx.x] = 0xffffffffffffffffULL;
        } else {
            mEndInStrBitmaps[threadIdx.x] = 0ULL;
        }
    }

    __syncthreads();

    // variables for saving temporary results in the last step
    unsigned long lb_mask, rb_mask, cb_mask;
    unsigned long lb_bit, rb_bit, cb_bit;
    unsigned long first, second;
    int cur_level = -1;

    for (int j = 0; j < mNumWords; ++j) {
        // get input info
        colonbit = mColonBitmap[j];
        commabit = mCommaBitmap[j];
        lbracebit = mLbraceBitmap[j];
        rbracebit = mRbraceBitmap[j];
        lbracketbit = mLbracketBitmap[j];
        rbracketbit = mRbracketBitmap[j];
        str_mask = mStrBitmap[j];

        // step 4: update structural character bitmaps
        unsigned long tmp = (~str_mask);
        colonbit = colonbit & tmp;
        lbracebit = lbracebit & tmp;
        rbracebit = rbracebit & tmp;
        commabit = commabit & tmp;
        lbracketbit = lbracketbit & tmp;
        rbracketbit = rbracketbit & tmp;

        // step 5: generate leveled bitmaps
        lb_mask = lbracebit | lbracketbit;
        rb_mask = rbracebit | rbracketbit;
        cb_mask = lb_mask | rb_mask;
        lb_bit = lb_mask & (-lb_mask);
        rb_bit = rb_mask & (-rb_mask);
        int top_word = j;
        if (!cb_mask) {
            if (cur_level >= 0 && cur_level <= mDepth) {
                if (!mLevColonBitmap[cur_level]) {
                    mLevColonBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                }
                if (!mLevCommaBitmap[cur_level]) {
                    mLevCommaBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                }
                if (colonbit) {
                    mLevColonBitmap[cur_level][top_word] = colonbit;
                } else {
                    mLevCommaBitmap[cur_level][top_word] = commabit;
                }
            } else if (cur_level < 0) {
                if (!mNegLevColonBitmap[-cur_level]) {
                    mNegLevColonBitmap[-cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                }
                if (!mNegLevCommaBitmap[-cur_level]) {
                    mNegLevCommaBitmap[-cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                }
                if (colonbit) {
                    mNegLevColonBitmap[-cur_level][top_word] = colonbit;
                } else {
                    mNegLevCommaBitmap[-cur_level][top_word] = commabit;
                }
            }
        } else {
            first = 1;
            while (cb_mask || first) {
                if (!cb_mask) {
                    second = 1UL<<63;
                } else {
                    cb_bit = cb_mask & (-cb_mask);
                    second = cb_bit;
                }
                if (cur_level >= 0 && cur_level <= mDepth) {
                    if (!mLevColonBitmap[cur_level]) {
                        mLevColonBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                    }
                    if (!mLevCommaBitmap[cur_level]) {
                        mLevCommaBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                    }
                    unsigned long mask = second - first;
                    if (!cb_mask) mask = mask | second;
                    unsigned long colon_mask = mask & colonbit;
                    if (colon_mask) {
                        mLevColonBitmap[cur_level][top_word] |= colon_mask;
                    } else {
                        mLevCommaBitmap[cur_level][top_word] |= (commabit & mask);
                    }
                    if (cb_mask) {
                        if (cb_bit == rb_bit) {
                            mLevColonBitmap[cur_level][top_word] |= cb_bit;
                            mLevCommaBitmap[cur_level][top_word] |= cb_bit;
                        }
                        else if (cb_bit == lb_bit && cur_level + 1 <= mDepth) {
                            if (!mLevCommaBitmap[cur_level + 1]) {
                                mLevCommaBitmap[cur_level + 1] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                            }
                            mLevCommaBitmap[cur_level + 1][top_word] |= cb_bit;
                        }
                    }
                } else if (cur_level < 0) {
                    if (!mNegLevColonBitmap[-cur_level]) {
                        mNegLevColonBitmap[-cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                    }
                    if (!mNegLevCommaBitmap[-cur_level]) {
                        mNegLevCommaBitmap[-cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                    }
                    unsigned long mask = second - first;
                    if (!cb_mask) mask = mask | second;
                    unsigned long colon_mask = mask & colonbit;
                    if (colon_mask) {
                        mNegLevColonBitmap[-cur_level][top_word] |= colon_mask;
                    } else {
                        mNegLevCommaBitmap[-cur_level][top_word] |= (commabit & mask);
                    }
                    if (cb_mask) {
                        if (cb_bit == rb_bit) {
                            mNegLevColonBitmap[-cur_level][top_word] |= cb_bit;
                            mNegLevCommaBitmap[-cur_level][top_word] |= cb_bit;
                        }
                        else if (cb_bit == lb_bit) {
                            if (cur_level + 1 == 0) {
                                if (!mLevCommaBitmap[0]) {
                                    mLevCommaBitmap[0] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                                }
                                mLevCommaBitmap[0][top_word] |= cb_bit;
                            } else {
                                if (!mNegLevCommaBitmap[-(cur_level + 1)]) {
                                    mNegLevCommaBitmap[-(cur_level + 1)] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                                }
                                mNegLevCommaBitmap[-(cur_level + 1)][top_word] |= cb_bit;
                            }
                        }
                    }
                }
                if (cb_mask) {
                    if (cb_bit == lb_bit) {
                        lb_mask = lb_mask & (lb_mask - 1);
                        lb_bit = lb_mask & (-lb_mask);
                        ++cur_level;
                        if (threadIdx.x == 0 && cur_level == 0) {
                            // JSON record at the top level could be an array
                            if (!mLevCommaBitmap[cur_level]) {
                                mLevCommaBitmap[cur_level] = (unsigned long*)malloc(mNumWords * sizeof(unsigned long));
                            }
                            mLevCommaBitmap[cur_level][top_word] |= cb_bit;
                        }
                    } else if (cb_bit == rb_bit) {
                        rb_mask = rb_mask & (rb_mask - 1);
                        rb_bit = rb_mask & (-rb_mask);
                        --cur_level;
                    }
                    first = second;
                    cb_mask = cb_mask & (cb_mask - 1);
                    if (cur_level > mMaxPositiveLevel) {
                        mMaxPositiveLevel = cur_level;
                    } else if (cur_level < mMinNegativeLevel) {
                        mMinNegativeLevel = cur_level;
                    }
                } else {
                    first = 0;
                }
            }
        }
    }
    if (mDepth == MAX_LEVEL - 1) mDepth = mMaxPositiveLevel;
    endLevels[threadIdx.x] = cur_level;

    // Sync threads to access endLevels on all threads
    __syncthreads();

    // Merge bitmaps here and save only the final ones. This works because each thread is running individually
    int curLevel = endLevels[0];

    if (threadIdx.x != 0) {
    for (int i = 0; i <= mMaxPositiveLevel; ++i) {
            *finalColonBitmaps[threadIdx.x][i] = mLevColonBitmap[i];
            *finalCommaBitmaps[threadIdx.x][i] = mLevCommaBitmap[i];
        }
    } else {
        // Set cur_level to the correct value
        for (int i = 1; i < threadIdx.x; ++i) {
            curLevel += (endLevels[i] + 1);
        }

        for(int j = 1; j <= -mMinNegativeLevel && (curLevel - j + 1) >= 0; ++j) {
            *finalColonBitmaps[threadIdx.x][curLevel - j + 1] = mNegLevColonBitmap[j];
            *finalCommaBitmaps[threadIdx.x][curLevel - j + 1] = mNegLevCommaBitmap[j];
        }
        for(int j = 0; j <= mMaxPositiveLevel && (curLevel + j + 1) >= 0; ++j) {
            *finalColonBitmaps[threadIdx.x][curLevel + j + 1] = mLevColonBitmap[j];
            *finalCommaBitmaps[threadIdx.x][curLevel + j + 1] = mLevCommaBitmap[j];
        }
    }


    // Free all local temp bitmaps that were created
    for (int m = 0; m < MAX_LEVEL; ++m){
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

    return;
}


__global__ void test(unsigned long* quoteBitmaps, int recordLength) {
    unsigned long count = 0;
    for (int i = 0; i < (int)(recordLength /64); ++i) {
        quoteBitmaps[i] = count;
        ++count;
    }
}

// Host data and functions

GPUParallelBitmap* GPUParallelBitmapConstructor::mGPUParallelBitmap = NULL;


GPUParallelBitmap* GPUParallelBitmapConstructor::construct(Record* record, int thread_num, int level_num) {
    cout << "Depth entered: " << level_num << endl;
    char* record_text = NULL;
    long length = 0;
    if (record->rec_start_pos > 0) record_text = record->text + record->rec_start_pos;
    else record_text = record->text;
    if (record->rec_length > 0) length = record->rec_length;
    else length = strlen(record->text);

    mGPUParallelBitmap = new GPUParallelBitmap(record_text, length, thread_num, level_num);
    int mode = mGPUParallelBitmap->parallelMode();


    // Settng up CUDA 
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);
    cudaError_t err = cudaSuccess;

    // 1. Create cuda variables
    int* dev_chunkEndLocs;
    char* dev_record;
    int localDepth = level_num - 1;

    unsigned long* dev_mStartInStrBitmaps;
    unsigned long* dev_mEndInStrBitmaps;

    int* dev_endLevels;


    // Allocate memory for cuda pointers and copy for pointer arrays
    cudaMalloc((void**)&dev_chunkEndLocs, MAX_THREAD * sizeof(int));
    cudaMalloc(&dev_record, sizeof(record_text));
    cudaMalloc((void**)&dev_mStartInStrBitmaps, MAX_THREAD * sizeof(unsigned long));
    cudaMalloc((void**)&dev_mEndInStrBitmaps, MAX_THREAD * sizeof(unsigned long));
    cudaMalloc((void**)&dev_endLevels, MAX_THREAD * sizeof(int));

    cudaMemcpy(dev_chunkEndLocs, mGPUParallelBitmap->chunkEndLocs, MAX_THREAD * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_record, record_text, sizeof(record_text), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mStartInStrBitmaps, mGPUParallelBitmap->mStartInStrBitmaps, MAX_THREAD * sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mEndInStrBitmaps, mGPUParallelBitmap->mStartInStrBitmaps, MAX_THREAD * sizeof(unsigned long), cudaMemcpyHostToDevice);\
    cudaMemcpy(dev_endLevels, mGPUParallelBitmap->endLevels, MAX_THREAD * sizeof(int), cudaMemcpyHostToDevice);


    int size = MAX_THREAD * MAX_LEVEL * 1;
    int quoteSize = int(length / 64);
    unsigned long* d_finalColonBitmaps;
    unsigned long* d_finalCommaBitmaps;
    unsigned long* d_quoteBitmaps;
    cudaMalloc((void**)&d_finalColonBitmaps, size * sizeof(unsigned long));
    cudaMalloc((void**)&d_finalCommaBitmaps, size * sizeof(unsigned long));
    cudaMalloc((void**)&d_quoteBitmaps, quoteSize * sizeof(unsigned long));


    unsigned long* h_flattenedColonBitmaps = new unsigned long[size];
    unsigned long* h_flattenedCommaBitmaps = new unsigned long[size];
    unsigned long* h_flattenedQuoteBitmaps = new unsigned long[quoteSize];
    
    cout << "Memory allocated successfully for leveled bitmaps" << endl;

    cudaMemcpy(d_finalColonBitmaps, h_flattenedColonBitmaps, size * sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_finalCommaBitmaps, h_flattenedCommaBitmaps, size * sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quoteBitmaps, h_flattenedQuoteBitmaps, quoteSize * sizeof(unsigned long), cudaMemcpyHostToDevice);

    cout << "Memory copied for leveled bitmaps" << endl;


   

    if (mode == NONSPECULATIVE) {
        cout << "nonspec" << endl;
        // 3. Call Kernel function with parallel threads
        nonSpecIndexConstructionKernel<<<1, thread_num>>>(
            dev_chunkEndLocs, 
            dev_record, 
            localDepth, 
            d_finalColonBitmaps, 
            d_finalCommaBitmaps, 
            d_quoteBitmaps,
            dev_mStartInStrBitmaps,
            dev_endLevels
            );

        // test<<<1, thread_num>>>(d_quoteBitmaps, mGPUParallelBitmap->chunkEndLocs[127]);
        cudaDeviceSynchronize();

        // Do this later
        //mGPUParallelBitmap->setWordIds(thread_num);

    } else {
        // TODO: fix spec function as well

        // specIndexConstructionKernel<<<1, thread_num>>>(
        //         dev_chunkEndLocs, 
        //         dev_record, 
        //         localDepth, 
        //         d_finalColonBitmaps, 
        //         d_finalCommaBitmaps, 
        //         d_quoteBitmaps,
        //         dev_mStartInStrBitmaps,
        //         dev_mEndInStrBitmaps,
        //         dev_endLevels
        //         );

        // TODO: Figure out where to put this
        //mGPUParallelBitmap->setWordIds(thread_num);
    }

    cout << "Finishes thread calls" << endl;

    cudaMemcpy(h_flattenedColonBitmaps, d_finalColonBitmaps, size * sizeof(unsigned long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flattenedCommaBitmaps, d_finalCommaBitmaps, size * sizeof(unsigned long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flattenedQuoteBitmaps, d_quoteBitmaps, quoteSize * sizeof(unsigned long), cudaMemcpyDeviceToHost);

    cout << "Finished memory copying back to host variables" << endl;

    // TODO: Correctly reassign memory for quoteBitmaps
    // This step is different for quote bitmaps because size varies depending on the pad length (values stored in chunkEndLocs)

    int index = 0;
    for (int i = 0; i < MAX_THREAD; ++i) {
        for (int j = 0; j < MAX_LEVEL; ++j) {
            for (int k = 0; k < 1; ++k) {
                mGPUParallelBitmap->finalColonBitmaps[i][j][k] = h_flattenedColonBitmaps[index];
                mGPUParallelBitmap->finalCommaBitmaps[i][j][k] = h_flattenedCommaBitmaps[index];
                ++index;
            }
        }
    }

    index = 0;
    for (int i = 0; i < MAX_THREAD; ++i) {
        int maxNum = 0;
        if (i == 0) {
            maxNum = mGPUParallelBitmap->chunkEndLocs[0];
        } else {
            maxNum = mGPUParallelBitmap->chunkEndLocs[i] - mGPUParallelBitmap->chunkEndLocs[i-1];
        }
        cout << "MaxNum: " << maxNum << endl;

        for (int j = 0; j < (int)(maxNum/64); ++j) {
            mGPUParallelBitmap->quoteBitmaps[i][j] = h_flattenedQuoteBitmaps[index];
            //cout << "Value at i: " << i << ", j: " << j << " is: " << mGPUParallelBitmap->quoteBitmaps[i][j] << endl;
            ++index;
        }
    }

    cout << "Memory copied to parallelBitmap vars" << endl;

    cout << "Test: " << mGPUParallelBitmap->finalCommaBitmaps[0][0][0] << endl;


    for (int p = 0; p < 128; ++p) {
        for (int i = 0; i < MAX_LEVEL; ++i) {
            if (mGPUParallelBitmap->finalCommaBitmaps[p][i][0] != 0) {
                cout << "Test: " << mGPUParallelBitmap->finalColonBitmaps[p][i][0] << endl;
            }
        }
    }

    cout << "Finishes" << endl;
    

    //exit(EXIT_FAILURE);


    // 5. Clean up memory
    // TODO: Correctly clean up memory
    cudaFree(dev_chunkEndLocs);
    cudaFree(dev_record);
    cudaFree(dev_mStartInStrBitmaps);
    cudaFree(dev_mEndInStrBitmaps);
    cudaFree(dev_endLevels);


    //exit(EXIT_FAILURE);

    return mGPUParallelBitmap;
}
