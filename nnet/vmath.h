#pragma once

#include "mstd.h"
#include <iostream>
#include <cmath>

namespace Math {
  template<typename T>
  struct Mat {
    T     *elem;
    bool  owned;
    uint  rows;
    uint  cols;

    Mat(uint _rows, uint _cols) 
      : elem(new T[_rows * _cols]),
        owned(true),
        rows(_rows),
        cols(_cols) {}

    Mat(uint _rows, uint _cols, T *buffer)
      : elem(buffer),
        owned(false),
        rows(_rows),
        cols(_cols) {}

    ~Mat() {
      if(owned) {
        delete []elem;
      }
    }

    inline T get(uint _row, uint _col) {
      if(_row >= rows || _col >= cols) {
        error("mat get");
      }
      return elem[cols * _row + _col];
    }
    inline void set(uint _row, uint _col, T val) {
      if(_row >= rows || _col >= cols) {
        error("mat set");
      }
      elem[cols * _row + _col] = val;
    }

    inline T *data() {
      return elem;
    }
    
    inline uint size() {
      return rows * cols;
    }

    friend std::ostream& operator<<(std::ostream &os, Mat &m) {
      os << "Mat:\n";
      for(uint i = 0; i < m.rows; i++) {
        os << "| ";
        for(uint n = 0; n < m.cols; n++) {
          os << m.get(i, n) << " ";
        }
        os << "|\n";
      }
      return os;
    }
  };
  
  template<typename T>
  struct Vec : Mat<T> {
    Vec(uint _rows)
      : Mat<T>(_rows, 1) {}

    Vec(uint _rows, T *buffer)
      : Mat<T>(_rows, 1, buffer) {}

    inline void set(uint _row, T val){
      Mat<T>::set(_row, 0, val);
    }
    inline T get(uint _row) {
      return Mat<T>::get(_row, 0);
    }
  };

  template<typename T, typename F>
  void for_all(Mat<T> &res, Mat<T> &m, F &f) {
    for(uint i = 0; i < res.rows; i++) {
      for(uint n = 0; n < res.cols; n++) {
        res.set(i, n, f(m.get(i,n)));
      }
    }
  }
  
  template<typename T>
  void add(Mat<T> &res, Mat<T> &m1, Mat<T> &m2) {
    for(uint i = 0; i < res.rows; i++) {
      for(uint n = 0; n < res.cols; n++) {
        res.set(i, n, m1.get(i, n) + m2.get(i,n));
      }
    }
  }

  template<typename T>
  void mult(Mat<T> &res, Mat<T> &m1, Mat<T> &m2) {
    if(m1.rows != res.rows || m2.cols != res.cols || m1.cols != m2.rows) {
      error("Mat mult");
    }
    uint w = m1.cols;
    for(uint i = 0; i < res.rows; i++) {
      for(uint n = 0; n < res.cols; n++) {
        T val{ 0 };
        for(uint x = 0; x < w; x++) {
          val += m1.get(i, x) * m2.get(x, n);
        }
        res.set(i, n, val);
      }
    }
  }

  //normalizes all values in matrix
  template<typename T>
  void sigmoid(Mat<T> &res, Mat<T> &m) {
    auto sf = [] (T val) {
      return (1 + tanh(val/2))/2;
    };
    for_all(res, m, sf);
  }

  template<typename T>
  void avg(Mat<T> &res, Mat<T> &m) {
    T count = m.rows * m.cols;
    for(uint i = 0; i < m.rows; i++) {
      for(uint n = 0; n < m.cols; n++) {
        res.set(i, n, m.get(i, n) / count);
      }
    }
  }

  template<typename T>
  void div(Mat<T> &res, Mat<T> &m, T val) {
    for(uint i = 0; i < m.rows; i++) {
      for(uint n = 0; n < m.cols; n++) {
        res.set(i, n, m.get(i, n) / val);
      }
    }
  }
  
  //res matrix consists of 0 and 1 if element in m is >= element in b
  template<typename T>
  void bin_norm(Mat<T> &res, Mat<T> &m, Mat<T> &b) {
    T val = 0;
    for(uint i = 0; i < m.rows; i++) {
      for(uint n = 0; n < m.cols; n++) {
        res.set(i, n, (m.get(i,n) >= b.get(i,n) ? 1 : 0));
      }
    }
  }
}
