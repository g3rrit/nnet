#pragma once

#include <iostream>

typedef unsigned int uint;

inline void error(const char *str) {
  std::cout << "error:\n" << str << "\n";
  getchar();
  exit(-1);
}
