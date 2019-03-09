#pragma once

#include "vmath.h"
#include "nnet.h"
#include "mstd.h"

namespace ML {

  void copy(Nnet &dest, Nnet &src);

  void rand_float_array(u64 *data, uint size, u64 min, u64 max, f64 factor);

  void rand_weight(Layer &layer, f64 factor);

  void rand_bias(Layer &layer, f64 factor);

  void rand_layer(Layer &layer, f64 factor);

  void rand_weight(Nnet &net, f64 factor);

  void rand_bias(Nnet &net, f64 factor);

  void rand_net(Nnet &net, f64 factor);

  void rand_weight(Nnet &net, uint layer_num, f64 factor);

  void rand_bias(Nnet &net, uint layer_num, f64 factor);

  void rand_layer(Nnet &net, uint layer_num, f64 factor);
}
