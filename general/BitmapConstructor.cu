#include "BitmapConstructor.h"

Bitmap* BitmapConstructor::construct(Record* record, int thread_num, int level_num) {
    Bitmap* bm = NULL;
    bm = GPUParallelBitmapConstructor::construct(record, thread_num, level_num);
    bm->type = PARALLEL;
    return bm;
}

BitmapIterator* BitmapConstructor::getIterator(Bitmap* bm) {
    BitmapIterator* bi = NULL;
    bi = new GPUParallelBitmapIterator((GPUParallelBitmap*)bm);
    bi->type = PARALLEL;
    return bi;
}
