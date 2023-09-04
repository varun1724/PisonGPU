#ifndef BITMAPCONSTRUCTOR_H
#define BITMAPCONSTRUCTOR_H

#include <string>
#include "Bitmap.h"
#include "BitmapIterator.h"
#include "../gpuSpec/GPU_ParallelBitmapConstructor.h"
#include "../gpuSpec/GPU_ParallelBitmapIterator.h"
#include "Records.h"

class BitmapConstructor {
  public:
    // construct leveled bitmaps for a JSON record
    static Bitmap* construct(Record* record, int thread_num = 1, int level_num = MAX_LEVEL);
    // get bitmap iterator for given bitmap index
    static BitmapIterator* getIterator(Bitmap* bi);
};

#endif
