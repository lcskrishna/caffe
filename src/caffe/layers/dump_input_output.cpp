#include "dump_input_output.h"

std::string filePath= "/home/svcbuild/Work/verify/caffe/out/";

std::string input_file_path = "/home/svcbuild/Work/verify/caffe/out/";

void formatFileName(std::string& str, const std::string& from, const std::string& to) {
    //Written to avoid conflicts with file creation with filenames that contain "/"
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

std::string getFilePath() {

    return filePath;
}

std::string getInputFilePath() {
	return input_file_path;
}
