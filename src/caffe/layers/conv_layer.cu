#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "dump_data.h"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // Multi queue execution, all previous work needs to be done first
    this->device_->FinishQueues();
    for (int_tp n = 0; n < this->num_; ++n) {
      // Multi queue execution, go through work queues
      this->device_->SwitchQueue(n);
      this->forward_gpu_gemm(bottom_data, n * this->bottom_dim_, weight,
          top_data, n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data, n * this->top_dim_, bias);
      }
    }
    // Multi queue execution, finish all queues
    this->device_->FinishQueues();
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
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff, n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data, n * this->bottom_dim_,
              top_diff, n * this->top_dim_, weight_diff);
        }
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[i]) {
        // Multi queue execution, all previous work needs to be done first
        this->device_->FinishQueues();
        for (int_tp n = 0; n < this->num_; ++n) {
          // Multi queue execution, go through work queues
          this->device_->SwitchQueue(n);
          this->backward_gpu_gemm(top_diff, n * this->top_dim_, weight,
                                  bottom_diff, n * this->bottom_dim_);
        }
        // Multi queue execution, finish all queues
        this->device_->FinishQueues();
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
