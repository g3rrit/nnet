#pragma once

#include "std.h"
#include "Math.h"
#include "Nnet.h"

using Math::Vec;

namespace ML {
  typedef pair<Vec<f64>, Vec<f64>> vec_pair;
  
  struct Training_Data {
    f64              *data;
    vector<vec_pair> v;
    uint             pair_count;
    uint             vec_size;
    uint             data_size;

    Training_Data(uint _pair_count, uint _vec_size);
    ~Training_Data();

    void sample(vector<vec_pair*> &s, uint sample_size);
  };

  
  struct Trainer {
    Training_Data  *t_data;
    vector<Nnet>   netv;
    Nnet_Structure structure;
    vector<f64>    costv;

    Trainer(Nnet_Structure _structure, uint net_count);
    ~Trainer();

    void set_training_data(Training_Data *_t_data);
    void load_training_data(string &path);
    void save_training_data(string &path);

    vector<f64> *calc_costv(uint block_size, uint rounds);
    Nnet *get_best();
    void dup_best();

    vector<f64> *train(uint block_size, uint rounds);
  };
}
