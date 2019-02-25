#include "vmath.h"
#include <iostream>

#include "nnet.h"
#include "nutil.h"

using Math::Vec;
using Math::Mat;
using ML::Nnet_Structure;
using ML::Nnet;

int main() {
  Vec<double> v1{ 2 };
  Vec<double> v2{ 2 };
  Mat<double> m{ 2, 2 };

  m.set(0,0, 1);
  m.set(0,1, 2);
  m.set(1,0, 3);
  m.set(1,1, 4);

  v1.set(0, 1);
  v1.set(1, 2);

  v2.set(0, 5);
  v2.set(1, 10);

  Vec<double> res{ 2 };
  add(res, v1, v2);

  std::cout << v1 << "\n" << v2 << "\n" << m << "\n";
  std::cout << "res:\n" << res << "\n";

  mult(res, m, v1);

  std::cout << "res:\n" << res << "\n";

  Nnet_Structure structure;
  structure.push_back(10);
  structure.push_back(20);
  structure.push_back(10);
  structure.push_back(30);
  Nnet net{structure};
  rand_net(net, 1);
  std::cout << net;
  rand_net(net, 0.1);
  std::cout << net;
  rand_net(net, 0.3);
  std::cout << net;

  getchar();
  return 0;
}
