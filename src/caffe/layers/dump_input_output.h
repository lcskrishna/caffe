#include <iostream>
#include <stdio.h>


#ifndef ENABLE_DUMP_INPUT_OUTPUT
        #define ENABLE_DUMP_INPUT_OUTPUT 1
#endif


void formatFileName(std::string& str, const std::string& from, const std::string& to);

std::string getFilePath();

std::string getInputFilePath();

