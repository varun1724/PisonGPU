#include <sys/time.h>
#include <fstream>
#include "RecordLoader.h"

#define MAX_PAD 64

void* aligned_malloc(size_t, size_t);

// Opens binary files
// First have to convert the data to binary files??
Record* RecordLoader::loadSingleRecord(const char* file_path) {
    unsigned long size;
    FILE* fp = fopen(file_path, "r");
    if (fp == NULL) {
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    rewind(fp);

    void* p = aligned_malloc(64, (size + MAX_PAD) * sizeof(char));
    if (p == NULL) {
        std::cout << "Fail to allocate memory space for input record." << std::endl;
    }
    char* record_text = static_cast<char*>(p);
    size_t load_size = fread(record_text, 1, size, fp);
    if (load_size == 0) {
        std::cout << "Fail to load the input record into memory" << std::endl;
    }
    int remain = 64 - (size % 64);
    int counter = 0;
    // pad the input data where its size can be divided by 64
    while (counter < remain) {
        record_text[size + counter] = 'd';
        counter++;
    }
    record_text[size + counter] = '\0';
    fclose(fp);

    // only one single record
    Record* record = new Record();
    record->text = record_text;
    record->rec_start_pos = 0;
    record->rec_length = strlen(record_text);
    return record;
}



RecordSet* RecordLoader::loadRecords(const char* file_path) {
    FILE *fp = fopen(file_path, "r");
    RecordSet* rs = new RecordSet();
    if (fp) {
        char line[MAX_RECORD_SIZE];
        std::string str;
        int start_pos = 0;
        while (fgets(line, sizeof(line), fp) != NULL) {
            if (strlen(line) <= MIN_RECORD_SIZE) continue;
            int remain = 64 - strlen(line) % 64;
            int top = strlen(line);
            while (remain > 0) {
                line[top++] = 'd';
                --remain;
            }
            line[top] = '\0';
            if (strlen(line) > MIN_RECORD_SIZE) {
                // concating a sequence of record texts into one single string generates the best performance for indexing and querying
                str.append(line);
                Record* record = new Record();
                record->rec_start_pos = start_pos;
                record->rec_length = strlen(line);
                start_pos += strlen(line);
                rs->recs.push_back(record);
                ++rs->num_recs;
            }
        }
        void* p = aligned_malloc(64, str.size()*sizeof(char));
        //void* p;
        if (p == NULL) {
            std::cout<<"Fail to allocate memory space for records from input file."<<std::endl;
        }
        for (int i = 0; i < rs->recs.size(); ++i) {
            // all record objects points to the same input text which contacts a sequence of JSON records
            rs->recs[i]->text = (char*) p;
            if (i == 0) strcpy(rs->recs[0]->text, str.c_str());
            // deconstructor in the last record object can delete input text
            if (i < rs->recs.size() - 1) rs->recs[i]->can_delete_text = false;
        }
        fclose(fp);
        return rs;
    }
    printf("Error: %d (%s)\n", errno, strerror(errno));
    std::cout<<"Fail open the file."<<std::endl;
    return rs;
}


void* aligned_malloc(size_t alignment, size_t size) {

    uintptr_t mask = alignment - 1;
    void* ptr = malloc(size + alignment + sizeof(void*));
    if (ptr) {
        void* aligned_ptr = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(ptr) & ~mask) + alignment);
        *(reinterpret_cast<void**>(aligned_ptr) - 1) = ptr;
        return aligned_ptr;
    }

    return nullptr;
}
