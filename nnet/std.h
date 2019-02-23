#pragma once

#include <iostream>

typedef unsigned int uint;

typedef double f64;

inline void error(const char *str) {
  std::cout << "error:\n" << str << "\n";
  getchar();
  exit(-1);
}
