#include "dump_data.h"

void formatFileName(std::string& str, const std::string& from, const std::string& to){
    //Written to avoid conflicts with file creation with filenames that contain "/"
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

/*
void dump_top_data(const float * output_data, int size_of_data, std::string layer_name){


    //FILE * fs;
    std::string temp = layer_name;
    formatFileName(layer_name,"/","_");
    std::string fileName = "out/"+ layer_name +".f32";
    std::ofstream outfile(fileName, std::ofstream::binary);
    //fs = std::fopen(fileName.c_str(),"wb");
    //if(!fs){
    //    std::cerr << "Error in creating the file." << std::endl;
    //
    std::cout << "The size of the layer:" << temp << " is " << size_of_data << std::endl;
    for(int i=0;i<size_of_data;i++){
        float out_val = output_data[i];
        outfile.write((char *)&out_val, sizeof(float));
        //std::cout << i << " " << out_val << std::endl;
        /*std::fwrite(&out_val,sizeof(float),1, fs);
    }

    std::cout <<"Data for the layer " << temp << " is written into the out folder." << std::endl;

} */
