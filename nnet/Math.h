#pragma once

#include "std.h"
#include <iostream>
#include <cmath>

namespace Math {
  template<typename T, usize _rows, usize _cols>
  struct Mat {
    T *elem;
    bool owned;

    Mat() {
      elem = new T[_rows * _cols];
      owned = true;
    }
    ~Mat() {
      if(owned) {
        delete []elem;
      }
    }

    inline T get(usize _row, usize _col) {
      if(_row >= _rows || _col >= _cols) {
        error("mat get");
      }
      return elem[_cols * _row + _col];
    }
    inline void set(usize _row, usize _col, T val) {
      if(_row >= _rows || _col >= _cols) {
        error("mat set");
      }
      elem[_cols * _row + _col] = val;
    }
    inline usize rows() {
      return _rows;
    }
    inline usize cols() {
      return _cols; 
    }

    friend std::ostream& operator<<(std::ostream &os, Mat &m) {
      os << "Mat:\n";
      for(int i = 0; i < m.rows(); i++) {
        os << "| ";
        for(int n = 0; n < m.cols(); n++) {
          os << m.get(i, n) << " ";
        }
        os << "|\n";
      }
      return os;
    }
  };
  
  template<typename T, usize _rows>
  struct Vec : Mat<T, _rows, 1> {
    inline void set(usize _row, T val){
      Mat<T, _rows, 1>::set(_row, 0, val);
    }
    inline T get(usize _row) {
      return Mat::get(_row, 0);
    }
  };

  template<typename T, usize _rows, usize _cols, typename F>
  void for_all(Mat<T, _rows, _cols> &res, Mat<T, _rows, _cols> &m, F &f) {
    for(usize i = 0; i < _rows; i++) {
      for(usize n = 0; n < _cols; n++) {
        res.set(i, n, f(m.get(i,n)));
      }
    }
  }
  
  template<typename T, usize _rows, usize _cols>
  void add(Mat<T, _rows, _cols> &res, Mat<T, _rows, _cols> &m1, Mat<T, _rows, _cols> &m2) {
    for(int i = 0; i < _rows; i++) {
      for(int n = 0; n < _cols; n++) {
        res.set(i, n, m1.get(i, n) + m2.get(i,n));
      }
    }
  }

  template<typename T, usize _rows, usize _cols, usize _w>
  void mult(Mat<T, _rows, _cols> &res, Mat<T, _rows, _w> &m1, Mat<T, _w, _cols> &m2) {
    for(usize i = 0; i < _rows; i++) {
      for(usize n = 0; n < _cols; n++) {
        T val{ 0 };
        for(usize x = 0; x < _w; x++) {
          val += m1.get(i, x) * m2.get(x, n);
        }
        res.set(i, n, val);
      }
    }
  }

  //normalizes all values in matrix
  template<typename T, usize _rows, usize _cols>
  void sigmoid(Mat<T, _rows, _cols> &res, Mat<T, _rows, _cols> &m) {
    for_all(res, m, [] (T val) {
      return (1 + tanh(val/2))/2;
    });
  }
}
