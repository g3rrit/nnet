#include "nutil.h"

#include <ctime>

namespace ML {
  void copy(Nnet &dest, Nnet &src) {
    if(dest.data_size != src.data_size) {
      error("copy");
    }
    memcpy(dest.data, src.data, dest.data_size);
  }

  void rand_float_array(f64 *data, uint size, f64 min, f64 max, f64 factor) {
    if(factor > 1 || factor < 0) {
      error("rand_float_array");
    }
    srand(time(0));
    f64 val = 0;
    for(uint i = 0; i < size; i++) {
      if(data[i] > max || data[i] < min) {
        data[i] = (max - min)/2;
      }
      val = (((f64) rand()/RAND_MAX)* (max - min) * factor)/2;
      data[i] = data[i] + val <= max ? data[i] + val : data[i] - val;
    }
  }

  void rand_weight(Layer &layer, f64 factor) {
    rand_float_array(layer.w.data(), layer.w.size(), WEIGHT_MIN, WEIGHT_MAX, factor);
  }

  void rand_bias(Layer &layer, f64 factor) {
    rand_float_array(layer.b.data(), layer.b.size(), BIAS_MIN, BIAS_MAX, factor);
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
