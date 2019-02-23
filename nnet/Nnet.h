#pragma once

#include <vector>
#include <iostream>

#include "std.h"
#include "Math.h"

#include <vector>

using Math::Vec;
using Math::Mat;
using std::vector;

namespace ML {
  //in_nodes := amount of input nodes
  //nodes    := amount of nodes in layer
  struct Layer {
    Mat<f64> w;
    Vec<f64> b;
    Vec<f64> a;

    Layer(uint in_nodes, uint nodes);
    Layer(uint in_nodes, uint nodes, f64 *buffer);
    Vec<f64> *get_active(Vec<f64> &input);
  };

  struct Layer_Builder {
    vector<Layer> *lv;
    uint          p_nodes;

    Layer_Builder(uint s_nodes);
    void add(uint nodes, f64 *data);
    vector<Layer> *build();
  };

  struct Nnet_Structure : public vector<uint> {
    uint data_size();
  };

  struct Nnet {
    f64            *data;
    uint           data_size;
    Nnet_Structure structure;
    vector<Layer>  *lv;

    Nnet(Nnet_Structure _structure);
    ~Nnet();

    Vec<f64> *get_output(Vec<f64> &in);
  };

  f64 cost(Vec<f64> &v, Vec<f64> &exp);

  std::ostream &operator<<(std::ostream &os, Layer &l);
  std::ostream &operator<<(std::ostream& os, Nnet &n);
}
