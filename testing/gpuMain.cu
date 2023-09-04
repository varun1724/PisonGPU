#include <chrono>
#include "../general/RecordLoader.h"
#include "../general/BitmapIterator.h"
#include "../general/BitmapConstructor.h"
using namespace std::chrono;

// $[*].user.id
std::string query(BitmapIterator* iter) {
    std::string output = "";
    while (iter->isArray() && iter->moveNext() == true) {
        if (iter->down() == false) continue;  /* array element on the top level */
        if (iter->isObject() && iter->moveToKey((char*)"user")) {
            if (iter->down() == false) continue; /* value of "user" */
            if (iter->isObject() && iter->moveToKey((char*)"id")) {
                // value of "id"
                char* value = iter->getValue();
                output.append(value).append(";").append("\n");
                if (value) free(value);
            }
            iter->up();
        }
        iter->up();
    }
    return output;
}

int main() {
    // PATH TO LOCAL SAMPLE TEST FILE: "dataset/twitter_sample_large_record.json"
    const char* file_path = "/content/drive/MyDrive/pthreads/data/twitter_large_record.json";
    Record* rec = RecordLoader::loadSingleRecord(file_path);
    if (rec == NULL) {
        std::cout << "record loading fails." << std::endl;
        return -1;
    }

    // set the number of threads for parallel gpu bitmap construction
    int thread_num = 128;

    /* set the number of levels of bitmaps to create, either based on the
     * query or the JSON records. E.g., query $[*].user.id needs three levels
     * (level 0, 1, 2), but the record may be of more than three levels
     */
    int level_num = 3;

    cout << "Running on " << thread_num << " threads" << endl;

    // Start clock to measure speeds
    auto start = high_resolution_clock::now();


    /* process the input record: first build bitmap, then perform
     * the query with a bitmap iterator
     */
    Bitmap* bm = BitmapConstructor::construct(rec, thread_num, level_num);
    BitmapIterator* iter = BitmapConstructor::getIterator(bm);
    std::string output = query(iter);

    // End clock
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Duration: " << duration.count() << endl;

    // Uncomment to see matches
    // std::cout << "matches are: " << output << std::endl;

    delete iter;
    delete bm;


    return 0;
}
