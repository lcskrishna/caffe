#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "dump_data.h"
namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ConcatParameter& concat_param = this->layer_param_.concat_param();

  //std::cout << this->layer_param_.name() << std::endl;

  CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
      << "Either axis or concat_dim should be specified; not both.";
}

template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int_tp num_axes = bottom[0]->num_axes();
  const ConcatParameter& concat_param = this->layer_param_.concat_param();
  if (concat_param.has_concat_dim()) {
    concat_axis_ = static_cast<int_tp>(concat_param.concat_dim());
    // Don't allow negative indexing for concat_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(concat_axis_, 0) << "casting concat_dim from uint32 to int32 "
        << "produced negative result; concat_dim must satisfy "
        << "0 <= concat_dim < " << kMaxBlobAxes;
    CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis());
  }
  // Initialize with the first blob.
  vector<int_tp> top_shape = bottom[0]->shape();
  num_concats_ = bottom[0]->count(0, concat_axis_);
  concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
  int_tp bottom_count_sum = bottom[0]->count();
  for (int_tp i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num_axes, bottom[i]->num_axes())
        << "All inputs must have the same #axes.";
    for (int_tp j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) { continue; }
      CHECK_EQ(top_shape[j], bottom[i]->shape(j))
          << "All inputs must have the same shape, except at concat_axis.";
    }
    bottom_count_sum += bottom[i]->count();
    top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
  }
  top[0]->Reshape(top_shape);
  CHECK_EQ(bottom_count_sum, top[0]->count());
  if (bottom.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }

  std::cout << "dim concat output : ";
  for(int i=0;i<top_shape.size();i++){
    std::cout << top_shape[i] << " ";
  }
  std::cout << std::endl;

}


template <typename Dtype>
void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) { return; }
  Dtype* top_data = top[0]->mutable_cpu_data();
  int_tp offset_concat_axis = 0;
  const int_tp top_concat_axis = top[0]->shape(concat_axis_);
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const int_tp bottom_concat_axis = bottom[i]->shape(concat_axis_);
    for (int_tp n = 0; n < num_concats_; ++n) {
      caffe_cpu_copy(bottom_concat_axis * concat_input_size_,
          bottom_data + n * bottom_concat_axis * concat_input_size_,
          top_data + (n * top_concat_axis + offset_concat_axis)
              * concat_input_size_);
    }
    offset_concat_axis += bottom_concat_axis;
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
void ConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (bottom.size() == 1) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  int_tp offset_concat_axis = 0;
  const int_tp top_concat_axis = top[0]->shape(concat_axis_);
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const int_tp bottom_concat_axis = bottom[i]->shape(concat_axis_);
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      for (int_tp n = 0; n < num_concats_; ++n) {
        caffe_cpu_copy(bottom_concat_axis * concat_input_size_, top_diff +
            (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
            bottom_diff + n * bottom_concat_axis * concat_input_size_);
      }
    }
    offset_concat_axis += bottom_concat_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConcatLayer);
#endif

INSTANTIATE_CLASS(ConcatLayer);
REGISTER_LAYER_CLASS(Concat);

}  // namespace caffe
