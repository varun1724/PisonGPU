#ifndef _RECORDLOADER_H
#define _RECORDLOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <thread>
#include <sys/time.h>
#include <sys/file.h>
#include <malloc.h>
#include <iostream>
#include <string>
#include <vector>
#include "Records.h"

class RecordLoader{
  public:
    static Record* loadSingleRecord(const char* file_path);
    static RecordSet* loadRecords(const char* file_path);

};
#endif
