#include "trainer.h"

#include <cstdio>
#include <thread>
#include <ctime>

#include "nutil.h"

using std::thread;

namespace ML {

  Training_Data::Training_Data(uint _pair_count, uint _in_vec_size, uint _out_vec_size)
    : pair_count(_pair_count),
      in_vec_size(_in_vec_size),
      out_vec_size(_out_vec_size),
      data_size((_in_vec_size + _out_vec_size) * _pair_count) {
    data = new f64[data_size];
  
    uint offset = 0;
    for(uint i = 0; i < pair_count; i++) {
      v.push_back(std::make_pair(Vec<f64>(_in_vec_size, data + offset), Vec<f64>(_out_vec_size, data + offset + in_vec_size)));
      offset += in_vec_size + out_vec_size;
    }
  }

  Training_Data::~Training_Data() {
    delete []data;
  }

  void Training_Data::sample(vector<vec_pair*> &s, uint sample_size) {
    srand(time(0));
    for(uint i = 0; i < sample_size; i++) {
      uint pos = rand() % v.size();
      s.push_back(&v.at(pos));
    }
  }
    

  Trainer::Trainer(Nnet_Structure _structure, uint net_count)
    : structure(_structure),
      t_data(0) {
    netv.reserve(net_count);
    costv.reserve(net_count);
    for(uint i = 0; i < net_count; i++) {
      netv.emplace_back(structure);
      costv.push_back(0);
    }
  }

  Trainer::Trainer(Nnet &net, uint net_count) {
    netv.reserve(net_count);
    costv.reserve(net_count);
    for(uint i = 0; i < net_count; i++) {
      netv.emplace_back(net.structure);
      copy(netv.at(i), net);
      costv.push_back(0);
    }
  }

  Trainer::~Trainer() {
    delete t_data;
  }

  void Trainer::set_training_data(Training_Data *_t_data) {
    t_data = _t_data;
  }

  void Trainer::load_training_data(string &path) {
    FILE *file = fopen(path.c_str(), "rb");
    if(!file) {
      save_training_data(path);
      error("unable to read training_data");
    }
    uint pair_count = 0;
    uint in_vec_size = 0;
    uint out_vec_size = 0;
    if(fread(&pair_count, sizeof(uint), 1, file) != 1) {
      error("unable to read pair_count");
    }
    if(fread(&in_vec_size, sizeof(uint), 1, file) != 1) {
      error("unable to read in_vec_size");
    }
    if(fread(&out_vec_size, sizeof(uint), 1, file) != 1) {
      error("unable to read out_vec_size");
    }

    t_data = new Training_Data(pair_count, in_vec_size, out_vec_size);
    if(fread(t_data->data, sizeof(f64), t_data->data_size, file) != t_data->data_size) {
      error("unable to read training data");
    }

    fclose(file);
  }

  void Trainer::save_training_data(string &path) {
    if(!t_data) {
      error("training data not loaded");
    }
    FILE *file = fopen(path.c_str(), "wb");
    if(!file) {
      error("unable to save training data");
    }
    if(fwrite(&t_data->pair_count, sizeof(uint), 1, file) != 1) {
      error("unable to save pair_count");
    }
    if(fwrite(&t_data->in_vec_size, sizeof(uint), 1, file) != 1) {
      error("unable to save in_vec_size");
    }
    if(fwrite(&t_data->out_vec_size, sizeof(uint), 1, file) != 1) {
      error("unable to save out_vec_size");
    }
    if(fwrite(t_data->data, sizeof(f64), t_data->data_size, file) != t_data->data_size) {
      error("unable to save training_data");
    }

    fclose(file);
  }

  vector<f64> *Trainer::calc_costv(uint block_size, uint rounds) {
    //lets make this threaded what could possibly go wrong
    auto tf = [] (Nnet *net, vector<vec_pair*> *td, f64 *c) {
                f64 temp_cost = 0;
                uint count = 0;
                for(vec_pair *vp : *td) {
                  temp_cost += cost(*net->get_output(vp->first), vp->second);
                  count++;
                }
                *c += temp_cost/(f64)count;
              };

    for(f64 &cost : costv) {
      cost = 0;
    }

    for(uint i = 0; i < rounds; i++) {
      vector<thread> thread_v;
      thread_v.reserve(netv.size());

      vector<vec_pair*> td;
      t_data->sample(td, block_size);

      for(uint n = 0; n < netv.size(); n++) {
        thread_v.at(n) = thread(tf, &netv.at(n), &td, &costv.at(n));
      }
      for(uint n = 0; n < netv.size(); n++) {
        thread_v.at(n).join();
      }
    }
    //average out cost vector
    for(f64 &cost : costv) {
      cost /= rounds;
    }

    return &costv;
  }

  Nnet *Trainer::get_best() {
    Nnet *res = nullptr;
    f64 min_cost = 10000; //TODO
    for(uint i = 0; i < netv.size(); i++) {
      if(costv.at(i) <= min_cost) {
        res = &netv.at(i);
        min_cost = costv.at(i);
      }
    }
    if(res == nullptr) {
      error("no minmal net found");
    }
    return res;
  }

  void Trainer::dup_best() {
    Nnet *best = get_best();
    for(Nnet &net : netv) {
      copy(net, *best);
    }
  }

  vector<f64> *Trainer::train(uint block_size, uint rounds, f64 t_factor) {
    vector<f64> *res = calc_costv(block_size, rounds);
    dup_best();
    for(Nnet &net : netv) {
      rand_net(net, t_factor);
    }
    return res;
  }
}
