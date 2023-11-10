#ifndef VARIABLE_H
#include <vector>
#include <fstream>

struct Variable {
    std::vector<float> data, grad;
    Variable(int size, bool requires_grad=true);
    void glorot(int in_size, int out_size);
    void zero();
    void zero_grad();
    void print(int col=0x7fffffff);
    void write2txt(std::string name, int col=0x7fffffff);
    void write2bin(std::string name);
    void readbin(std::string name);
    float grad_norm();
};

#define VARIABLE_H
#endif
