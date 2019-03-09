#include "nutil.h"

#include <ctime>

namespace ML {
  void copy(Nnet &dest, Nnet &src) {
    if(dest.data_size != src.data_size) {
      error("copy");
    }
    memcpy(dest.data, src.data, dest.data_size);
  }

  void rand_float_array(u64 *data, uint size, u64 min, u64 max, f64 factor) {
    if(factor > 1 || factor < 0) {
      error("rand_float_array");
    }
    //srand(time(0));
    u64 val = 0;
    for(uint i = 0; i < size; i++) {
      if(data[i] > max || data[i] < min) {
        data[i] = (max - min)/2;
      }
      val = ((static_cast<f64>(rand())/RAND_MAX) * factor * (max - min))/2;
      data[i] = data[i] + val <= max ? data[i] + val : data[i] - val;
    }
  }

  void rand_weight(Layer &layer, f64 factor) {
    rand_float_array(layer.w.data(), layer.w.size(), 0, U64_MAX / layer.w.cols, factor);
  }

  void rand_bias(Layer &layer, f64 factor) {
    rand_float_array(layer.b.data(), layer.b.size(), 0, U64_MAX, factor);
  }

  void rand_layer(Layer &layer, f64 factor) {
    rand_weight(layer, factor);
    rand_bias(layer, factor);
  }

  void rand_weight(Nnet &net, f64 factor) {
    for(Layer &layer : *net.lv) {
      rand_weight(layer, factor);
    }
  }

  void rand_bias(Nnet &net, f64 factor) {
    for(Layer &layer : *net.lv) {
      rand_bias(layer, factor);
    }
  }

  void rand_net(Nnet &net, f64 factor) {
    for(Layer &layer : *net.lv) {
      rand_layer(layer, factor);
    }
  }

  void rand_weight(Nnet &net, uint layer_num, f64 factor) {
    if(layer_num >= net.lv->size()) {
      error("rand_weight");
    }
    rand_weight(net.lv->at(layer_num), factor);
  }

  void rand_bias(Nnet &net, uint layer_num, f64 factor) {
    if(layer_num >= net.lv->size()) {
      error("rand_weight");
    }
    rand_bias(net.lv->at(layer_num), factor);
  }

  void rand_layer(Nnet &net, uint layer_num, f64 factor) {
    if(layer_num >= net.lv->size()) {
      error("rand_weight");
    }
    rand_layer(net.lv->at(layer_num), factor);
  }
}
