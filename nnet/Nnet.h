#pragma once

#include <iostream>

#include "std.h"
#include "Math.h"

using Math::Vec;
using Math::Mat;

//in_nodes := amount of input nodes
//nodes    := amount of nodes in layer
template<typename T, usize nodes, usize in_nodes>
struct Layer {
  Mat<T, nodes, in_nodes> w;
  Vec<T, nodes>           b;
  Vec<T, nodes>           a;

  Vec<T, nodes> *get_activ(Vec<T, > &input) {
    mult(a, w, input);
    add(a, b);
    sigmoid(a, a);
    return &a;
  }
};

//in_count  := amount of input nodes
//h_count   := amount of hidden layer nodes
//h_layer   := amount of hidden layers
//out_count := amount of output nodes
template<typename T = double, usize in_count, usize h_count, usize h_layer, usize out_count>
struct Nnet
{
  Vec<T, in_count>  input;
  Vec<T, out_count> output;
  Layer<T, 

  Nnet();
  ~Nnet();
};

