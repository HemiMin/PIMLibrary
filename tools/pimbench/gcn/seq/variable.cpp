#include "variable.h"
#include "rand.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <iostream>

Variable::Variable(int size, bool requires_grad):
    data(size), grad(requires_grad ? size : 0) {}

void Variable::glorot(int in_size, int out_size) {
    float range = sqrtf(6.0f / (in_size + out_size));

    for(int i = 0; i < data.size(); i++) {
        const float rand = float(RAND()) / MY_RAND_MAX - 0.5;
        data[i] = rand * range * 2;
    }
}

void Variable::zero() {
    for(int i = 0; i < data.size(); i++)
        data[i] = 0;
}

void Variable::zero_grad() {
    for(int i = 0; i < grad.size(); i++)
        grad[i] = 0;
}

void Variable::print(int col) {
    int count = 0;
    for(float x: data) {
        printf("%.4f ", x);
        count++;
        if(count % col == 0) printf("\n");
    }
}

void Variable::write2txt(std::string name, int col) {
  std::ofstream out(name);
    int count = 0;
    for(float x: data) {
      out << x << " ";
        count++;
        if(count % col == 0) out << "\n";
    }
}

void Variable::write2bin(std::string name) {
  std::ofstream out(name, std::ios::out | std::ios::binary);
  for(float x: data) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(float));
  }
  out.close();
}

void Variable::readbin(std::string name) {
  std::ifstream in(name, std::ios::binary);
  int i = 0;
  float x;
  while(in.read(reinterpret_cast<char*>(&x), sizeof(float))){
    data[i] = x;
    i++;
  }
}

float Variable::grad_norm() {
    float norm = 0;
    for(float x: grad) norm += x * x;
    return sqrtf(norm);
}
