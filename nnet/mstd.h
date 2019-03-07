#pragma once

#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <limits>

using std::pair;
using std::vector;
using std::string;

typedef unsigned int uint;

typedef double f64;
#define F64_MAX DBL_MAX

inline void error(const char *str) {
  std::cout << "error:\n" << str << "\n";
  getchar();
  exit(-1);
}
