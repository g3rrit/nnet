#include "nnet.h"
#include "vmath.h"

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

  vector<Layer> *Nnet_Structure::build(f64 *data) {
    auto si = std::begin(*this);
    uint prev = *si;

    vector<Layer> *lv = new vector<Layer>();
    lv->reserve(20); //TODO bug if this line is removed
    //prob because reserve doesent extend the vector
    
    uint data_offset = 0;
    for(si++; si != std::end(*this); si++) {
      lv->emplace_back(prev, *si, data + data_offset);
      data_offset += prev * (*si) + (*si);
      prev = *si;
    }    
    return lv;
  }

  void Nnet_Structure::load(FILE *file) {
    this->clear();
    
    uint num_layers = 0;
    if(fread(&num_layers, sizeof(uint), 1, file) != 1) {
      error("unable to read number of layers");
    }
    uint nodes = 0;
    for(uint i = 0; i < num_layers; i++) {
      if(fread(&nodes, sizeof(uint), 1, file) != 1) {
        error("unable to read number of nodes");
      }
      this->push_back(nodes);
    }
  }

  void Nnet_Structure::save(FILE *file) {
    uint num_layers = this->size();
    if(fwrite(&num_layers, sizeof(uint), 1, file) != 1) {
      error("unable to write number of layers");
    }
    for(uint &nodes : *this) {
      if(fwrite(&nodes, sizeof(uint), 1, file) != 1) {
        error("unable to write number of nodes");
      }
    }
  }


  Nnet::Nnet(Nnet_Structure _structure)
    : structure(_structure),
      data_size(0) {
    data_size = structure.data_size();
    data = new f64[data_size];
    
    lv = structure.build(data);
  }

  Nnet::Nnet(string &path) 
    : data(0),
      data_size(0),
      lv(0) {
    FILE *file = fopen(path.c_str(), "rb");
    if(!file) {
      error("unable to open nnet file");
    }
    load(file);
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

  void Nnet::load(FILE *file) {
    structure.load(file);
    data_size = structure.data_size();
    delete []data;
    data = new f64[data_size];

    if(fread(data, sizeof(uint), data_size, file) != data_size) {
      error("unable to read data");
    }
    
    delete lv;
    lv = structure.build(data);
  }

  void Nnet::save(FILE *file) {
    structure.save(file);
    if(fwrite(data, sizeof(uint), data_size, file) != data_size) {
      error("unable to write data");
    }
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

