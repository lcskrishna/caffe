#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
 /* std::cout << "dim RelU output : ";
  vector<int_tp> bottom_shape = bottom[0]->shape();
  for(int i=0;i<bottom_shape.size();i++){
    std::cout << bottom_shape[i]<< " ";
  }
  std::cout << std::endl;*/
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
