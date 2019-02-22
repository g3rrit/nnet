#pragma once

#include <vector>
#include <iostream>

#include "std.h"
#include "Math.h"

using Math::Vec;
using Math::Mat;
using std::vector;

//in_nodes := amount of input nodes
//nodes    := amount of nodes in layer
template<typename T>
struct Layer {
  Mat<T> w;
  Vec<T> b;
  Vec<T> a;

  Layer(uint in_nodes, uint nodes)
    : w(nodes, in_nodes),
      b(nodes),
      a(nodes) {}

  Layer(uint in_nodes, uint nodes, T *buffer) 
    : w(nodes, in_nodes, buffer),
      b(nodes, buffer + (in_nodes * nodes)),
      a(nodes) {}

  Vec<T> *get_activ(Vec<T> &input) {
    mult(a, w, input);
    add(a, b);
    sigmoid(a, a);
    return &a;
  }
};

template<typename T>
struct Layer_Builder {
  vector<Layer<T>> *lv;
  uint             p_nodes;

  Layer_Builder(uint s_nodes) 
     : p_nodes(s_nodes) {
    lv = new vector<Layer<T>>();
  }
  void add(uint nodes, T *data) {
    lv.emplace_back(p_nodes, nodes, data);
    p_nodes = nodes;
  }
  vector<Layer<T>> *build() {
    return lv;
  }
};

template<typename T = double>
struct Nnet {
  T                *data;
  vector<Layer<T>> *lv;

  Nnet(uint in_nodes, vector<Layer<T>> *_lv, T *_data)
    : data(_data),
      lv(_lv) {}

  ~Nnet() {
    delete lv;
  }

  Vec<T> *get_output(Vec<T> &in) {
    Vec<T> *current = &in;
    for(Layer<T> &layer : lv) {
      current = layer.get_active(current);
    }
    return current;
  }
};

template<typename T>
double cost(Vec<T> &v, Vec<T> &exp) {
  double cost = 0;
  if(v.rows != exp.rows) {
    error("get_cost");
  }
  for(uint i = 0; i < v.rows; i++) {
    cost += (v.get(i) - exp.get(i)) * (v.get(i) - exp.get(i));
  }
  return cost;
}
