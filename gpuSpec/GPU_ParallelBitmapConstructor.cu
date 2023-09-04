#include "GPU_ParallelBitmapConstructor.h"
#include "GPU_LocalBitmap.h"


//////////////////////
//////////////////////
// LOCAL BITMAP FUCNTIONS
//////////////////////
//////////////////////

// Function to store the uchar16 vector into the memory pointed by uint16_t* pointer
__device__ void GPULocalBitmap::store(uchar16 vector, uint16_t* v_data) {
    unsigned char* uchar_ptr = (unsigned char*)v_data;
    for (int i = 0; i < 16; i++) {
        uchar_ptr[i] = vector.data[i];
    }
}

__device__ unsigned int GPULocalBitmap::uchar16_movemask(const uchar16& vector) {
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
__device__ unsigned long GPULocalBitmap::clmul64(unsigned long num1, unsigned long num2) {
    unsigned long result = 0;

    for (int i = 0; i < 64; i++) {
        unsigned long bit = (num2 >> i) & 1;
        result ^= (num1 << i) * bit;
    }

    // Returns the lower 64 bits of the result
    return result & 0xFFFFFFFFFFFFFFFF;
}


// Function to set all values in the uchar16 type to the specified value
__device__ uchar16 GPULocalBitmap::make_uchar16(unsigned char value) {
    uchar16 result;
    for (int i = 0; i < 16; i++) {
        result.data[i] = value;
    }
    return result;
}

// Function to perform a bitwise and between two uchar16 types
__device__ uchar16 GPULocalBitmap::bitwiseAnd(const uchar16& v1, const uchar16& v2) {
    uchar16 result;
    for (int i = 0; i < 16; i++) {
        result.data[i] = v1.data[i] & v2.data[i];
    }
    return result;
}

// Checks for overflow
__device__ bool GPULocalBitmap::addWithOverflowCheck(unsigned long a, unsigned long b, unsigned long* result) {
    unsigned long long sum = a + b;
    bool overflow = (sum < a) || (sum < b); // Check if the sum is less than any of the operands
    *result = sum;
    return overflow;
}

__device__ void nonSpecIndexConstruction() {

    // Based on the array that is passed in, the chunk that the record needs to eval is determined



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
    uint64_t prev_iter_inside_quote = mStartInStrBitmap;
    const uint64_t even_bits = 0x5555555555555555ULL;
    const uint64_t odd_bits = ~even_bits;


    for (int j = 0; j < mNumTmpWords; ++j) {
        colonbit = 0, quotebit = 0, escapebit = 0, lbracebit = 0, rbracebit = 0, commabit = 0, lbracketbit = 0, rbracketbit = 0;
        unsigned long i = j * 32;
        // step 1: build structural character bitmaps
        uchar16 v_text0 = make_uchar16(0);
        uchar16 v_text1 = make_uchar16(0);

        for (int j = 0; j < sizeof(uchar16); j++) {
            v_text0.data[j] = mRecord[i + j];
            v_text0.data[j] = mRecord[i + 16 + j];
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
        if(j % 2 == 0) {
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
            mQuoteBitmap[++top_word] = quote_bits;

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
                            if (mThreadId == 0 && cur_level == 0) {
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
    mEndLevel = cur_level;
    return;
}



///////////////////////////////////
///////////////////////////////////
/// BUILD STRING MASK BITMAP FUNCTION
/// For speculative mode
///////////////////////////////////
///////////////////////////////////


__device__ void GPULocalBitmap::buildStringMaskBitmap() {
    // allocate memory space for saving results
    if (!mQuoteBitmap) {
        mQuoteBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    }
    if (!mColonBitmap) {
        mColonBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    }
    if (!mCommaBitmap) {
        mCommaBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    }
    if (!mStrBitmap) {
        mStrBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    }
    if (!mLbraceBitmap) {
        mLbraceBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    }
    if (!mRbraceBitmap) {
        mRbraceBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    }
    if (!mLbracketBitmap) {
        mLbracketBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    }
    if (!mRbracketBitmap) {
        mRbracketBitmap = (unsigned long*)malloc((mNumWords) * sizeof(unsigned long));
    }

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
    uint64_t prev_iter_inside_quote = mStartInStrBitmap;
    const uint64_t even_bits = 0x5555555555555555ULL;
    const uint64_t odd_bits = ~even_bits;

    for (int j = 0; j < mNumTmpWords; ++j) {
        colonbit = 0, quotebit = 0, escapebit = 0, lbracebit = 0, rbracebit = 0, commabit = 0, lbracketbit = 0, rbracketbit = 0;
        unsigned long i = j * 32;
        // step 1: build structural character bitmaps
        uchar16 v_text0;
        uchar16 v_text1;
        memcpy(&v_text0, mRecord + i, sizeof(uchar16));
        memcpy(&v_text1, mRecord + i + 16, sizeof(uchar16));

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
        if(j % 2 == 0) {
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
            mQuoteBitmap[top_word] = quote_bits;

            // step 3: build string mask bitmaps
            unsigned long long allOnes64Bit = ULLONG_MAX;
            str_mask = clmul64(quote_bits, allOnes64Bit);
            str_mask ^= prev_iter_inside_quote;
            mStrBitmap[top_word] = str_mask;
            prev_iter_inside_quote = static_cast<uint64_t>(static_cast<int64_t>(str_mask) >> 63);
        }
    }
    mEndInStrBitmap = prev_iter_inside_quote;
}


///////////////////////////////////
///////////////////////////////////
/// BUILD LEVELED BITMAP FUNCTION
/// For speculative mode
///////////////////////////////////
///////////////////////////////////


__device__ void GPULocalBitmap::buildLeveledBitmap() {
    // variables for saving temporary results in the first four steps
    unsigned long colonbit, lbracebit, rbracebit, commabit, lbracketbit, rbracketbit;
    unsigned long str_mask;

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
                        if (mThreadId == 0 && cur_level == 0) {
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
    mEndLevel = cur_level;
}

GPUParallelBitmap* GPUParallelBitmapConstructor::mGPUParallelBitmap = NULL;

__global__ void nonSpecIndexConstructionKernel(GPUParallelBitmap* mGPUParallelBitmap) {

    mGPUParallelBitmap->mBitmaps[threadIdx.x]->nonSpecIndexConstruction();
    // cout<<thread_id<<"th thread finishes structural index construction."<<endl;
    __syncthreads();
}

__global__ void buildStringMaskBitmapKernel(GPUParallelBitmap* mGPUParallelBitmap) {

    //cout<<thread_id<<"th thread starts building string mask bitmap."<<endl;
    mGPUParallelBitmap->mBitmaps[threadIdx.x]->buildStringMaskBitmap();
    //cout<<thread_id<<"th thread finishes building string mask bitmap."<<endl;
    __syncthreads();
}

__global__ void buildLeveledBitmapKernel(GPUParallelBitmap* mGPUParallelBitmap) {

    mGPUParallelBitmap->mBitmaps[threadIdx.x]->buildLeveledBitmap();
    //cout<<thread_id<<"th thread finishes building leveled bitmap."<<endl;
    __syncthreads();

}


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

    int chunk_len = rec_len / thread_num;
    if (chunk_len % 64 > 0) {
        chunk_len = chunk_len + 64 - chunk_len % 64;
    }


    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64*1024*1024);

    if (mode == NONSPECULATIVE) {

        // TODO: Create correct cuda variables




        nonSpecIndexConstructionKernel<<<1, thread_num>>>(mGPUParallelBitmap);
        mGPUParallelBitmap->mergeBitmaps();
    } else {
        buildStringMaskBitmapKernel<<<1, thread_num>>>(mGPUParallelBitmap);
        mGPUParallelBitmap->rectifyStringMaskBitmaps();
        buildLeveledBitmapKernel<<<1, thread_num>>>(mGPUParallelBitmap);
        mGPUParallelBitmap->mergeBitmaps();
    }
    cout << "Depth at end of constructor: " << mGPUParallelBitmap->mDepth << endl;
    return mGPUParallelBitmap;
}
