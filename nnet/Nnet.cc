#include "Nnet.h"
#include "Math.h"

namespace ML {

  Layer::Layer(uint in_nodes, uint nodes)
    : w(nodes, in_nodes),
      b(nodes),
      a(nodes) {}

  Layer::Layer(uint in_nodes, uint nodes, f64 *buffer) 
    : w(nodes, in_nodes, buffer),
      b(nodes, buffer + (in_nodes * nodes)),
      a(nodes) {}

  Vec<f64> *Layer::get_active(Vec<f64> &input) {
    mult(a, w, input);
    add(a, a, b);
    sigmoid(a, a);
    return &a;
  }


  Layer_Builder::Layer_Builder(uint s_nodes) 
    : p_nodes(s_nodes) {
    lv = new vector<Layer>();
    lv->reserve(20); // TODO: figure out why vector doesnt automatically reserve space on emplace back
  }

  void Layer_Builder::add(uint nodes, f64 *data) {
    lv->emplace_back(p_nodes, nodes, data);
    p_nodes = nodes;
  }

  vector<Layer> *Layer_Builder::build() {
    return lv;
  }

  uint Nnet_Structure::data_size() {
    auto si = std::begin(*this);
    uint size = 0;
    uint prev = *si;
    for(si++; si != std::end(*this); si++) {
      size += prev * (*si) + (*si);
      prev = *si;
    }
    return size;
  }


  Nnet::Nnet(Nnet_Structure _structure)
    : structure(_structure),
      data_size(0) {
    data_size = structure.data_size();
    data = new f64[data_size];
    
    //build layers
    auto si = std::begin(structure);
    uint prev = *si;
    Layer_Builder l_builder{ prev };    
    uint data_offset = 0;
    for(si++; si != std::end(structure); si++) {
      l_builder.add(*si, data + data_offset);
      data_offset += prev * (*si) + (*si);
      prev = *si;
    }
    lv = l_builder.build();
  }

  Nnet::~Nnet() {
    delete []data;
    delete lv;
  }

  Vec<f64> *Nnet::get_output(Vec<f64> &in) {
    Vec<f64> *current = &in;
    for(Layer &layer : *lv) {
      current = layer.get_active(*current);
    }
    return current;
  }


  f64 cost(Vec<f64> &v, Vec<f64> &exp) {
    f64 cost = 0;
    if(v.rows != exp.rows) {
      error("get_cost");
    }
    for(uint i = 0; i < v.rows; i++) {
      cost += (v.get(i) - exp.get(i)) * (v.get(i) - exp.get(i));
    }
    return cost;
  }

  
//---------------------------------------
// OUTPUT
//---------------------------------------
  std::ostream &operator<<(std::ostream &os, Layer &l) {
    os << "--------------------\n"
      << "WEIGHTSV:\n" << l.w << "\nBIASEV:\n" << l.b << "\n"
      << "--------------------\n";
    return os;
  }

  std::ostream &operator<<(std::ostream& os, Nnet &n) {
    os << "--------------------\n"
      << "Neural Net:\n";
    uint count = 0;
    for(Layer &layer : *n.lv) {
      os << "Layer[" << count << "]:\n" << layer;
      count++;
    }
    os << "--------------------\n";
    return os;
  }
}

