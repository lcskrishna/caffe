#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "dump_data.h"

namespace caffe {

template<typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int_tp* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int_tp* stride_data = this->stride_.cpu_data();
  const int_tp* pad_data = this->pad_.cpu_data();
  const int_tp* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int_tp i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int_tp input_dim = this->input_shape(i + 1);
    const int_tp kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1)
        + 1;
    const int_tp output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);

  }

  std::cout << " kernel : ";
  for(int i=0;i < this->num_spatial_axes_;i++){
    std::cout << kernel_shape_data[i] << " ";
  }

  std::cout << "stride : ";
  for(int i=0;i< this->num_spatial_axes_;i++){
      std::cout << stride_data[i] << " ";
  }

  std::cout << "pad : ";
  for(int i=0;i< this->num_spatial_axes_;i++){
      std::cout << pad_data[i] << " ";
  }

  std::cout << "dilation : ";
  for(int i=0;i<this->num_spatial_axes_;i++){
      std::cout << dilation_data[i] << " ";
  }


}


template<typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int_tp n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                             top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }


  vector<int_tp> bottom_shape = bottom[0]->shape();
  int bottom_dim = 1;
  std::cout << "Bottom Shape is : " << std::endl;
  for(int i=0;i < bottom_shape.size();i++){
    std::cout << bottom_shape[i] << " ";
    bottom_dim = bottom_dim * bottom_shape[i];
  }
  std::cout << std::endl;

  std::string bottom_layer_name = "bottom_" + this->layer_param_.name();
  formatFileName(bottom_layer_name,"/","_");
  std::string bottomFileName = "/home/svcbuild/Work/caffe/examples/CIFAR_TEST/out/"+ bottom_layer_name +".f32";
  std::ofstream bottomFile(bottomFileName, std::ios::out | std::ios::binary);
  if(bottomFile){
      std::cout <<"File is created." << std::endl;
  }else{
      std::cout <<"File is not created." << std::endl;
  }
  std::cout << "The size of the layer:" << bottom_layer_name << " is " << bottom_dim << std::endl;
    const float * bottom_data = (const float *) bottom[0]->cpu_data();
    for(int j=0;j<bottom_dim;j++){
        float out_val = bottom_data[j];
        bottomFile.write((char *)&out_val, sizeof(float));
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

template<typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                  bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
