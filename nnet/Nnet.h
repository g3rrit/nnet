#pragma once

#include <vector>
#include <iostream>

#include "mstd.h"
#include "vmath.h"

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

  struct Nnet_Structure : public vector<uint> {
    uint data_size();
    vector<Layer> *build(f64 *data);
    void load(FILE *file);
    void save(FILE *file);
  };

  struct Nnet {
    f64            *data;
    uint           data_size;
    Nnet_Structure structure;
    vector<Layer>  *lv;

    Nnet(Nnet_Structure _structure);
    Nnet(string &path);
    ~Nnet();

    Vec<f64> *get_output(Vec<f64> &in);

    void load(FILE *file);
    void save(FILE *file);
  };

  f64 cost(Vec<f64> &v, Vec<f64> &exp);

  std::ostream &operator<<(std::ostream &os, Layer &l);
  std::ostream &operator<<(std::ostream& os, Nnet &n);
}
