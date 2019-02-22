#include "Math.h"
#include <iostream>

using Math::Vec;
using Math::Mat;

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

  getchar();
  return 0;
}
