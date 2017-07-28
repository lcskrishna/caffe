#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"
#include "dump_data.h"
namespace caffe {


template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int_tp count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int_tp i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }

  vector<int_tp> output_shape = top[0]->shape();
  int output_dim=1;
  for(int i = 0 ; i < output_shape.size();i++){
    output_dim = output_dim * output_shape[i];
  }

  std::string layer_name = this->layer_param_.name();
  std::string temp = layer_name;
  formatFileName(layer_name,"/","_");
  std::string fileName = "/home/svcbuild/Work/caffe/examples/CIFAR_TEST/out/"+ layer_name +".f32";
  std::ofstream outfile(fileName, std::ios::out | std::ios::binary);
  if(outfile){
      std::cout <<"File is created." << std::endl;
  }else{
      std::cout <<"File is not created." << std::endl;
  }
  std::cout << "The size of the layer:" << temp << " is " << output_dim << std::endl;
    const float * output_data = (const float *) top[0]->cpu_data();
    for(int j=0;j<output_dim;j++){
        float out_val = output_data[j];
        outfile.write((char *)&out_val, sizeof(float));
    }

}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int_tp count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int_tp i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
