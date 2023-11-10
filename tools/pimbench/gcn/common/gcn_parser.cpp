//
// Created by Chengze Fan on 2019-04-17.
//

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include "gcn_parser.h"

using namespace std;
GCNParser::GCNParser(GCNParams *gcnParams, GCNData *gcnData, std::string graph_name) {
    string root = "data/";
    try {
    this->graph_file.open(root + graph_name + ".graph");
    this->split_file.open(root + graph_name + ".split");
    this->svmlight_file.open(root + graph_name + ".svmlight");
    } catch (std::ios_base::failure& e) {
      std::cerr << e.what() << '\n';
    }
    this->gcnParams = gcnParams;
    this->gcnData = gcnData;
}

void GCNParser::parseGraph() {
    auto &graph_sparse_index = this->gcnData->graph;

    graph_sparse_index.indptr.push_back(0);
    int node = 0;
    while(true) {
        std::string line;
        getline(graph_file, line);
        if (graph_file.eof()) break;
        
        // Implicit self connection
        graph_sparse_index.indices.push_back(node);
        graph_sparse_index.indptr.push_back(graph_sparse_index.indptr.back() + 1);
        node++;

        std::istringstream ss(line);
        while (true) {
            int neighbor;
            ss >> neighbor;
            if (ss.fail()) break;
            graph_sparse_index.indices.push_back(neighbor);
            graph_sparse_index.indptr.back() += 1;
        }
    }
    
    gcnParams->num_nodes = node;
    std::vector<float> adj(node*node, 0);

    for (int src = 0; src < graph_sparse_index.indptr.size() - 1; src++) {
        for (int i = graph_sparse_index.indptr[src]; i < graph_sparse_index.indptr[src + 1]; i++) {
            int dst = graph_sparse_index.indices[i];
            float coef = 1.0 / sqrtf(
                    (graph_sparse_index.indptr[src + 1] - graph_sparse_index.indptr[src]) * (graph_sparse_index.indptr[dst + 1] - graph_sparse_index.indptr[dst])
            );
            if (adj[src*node + dst] != 0) adj[src*node + dst] += coef;
            else adj[src * node + dst] = coef;
        }
    }

    //std::ofstream out2("data/cora_adj.txt");
    //for (int i = 0; i < node; i++) {
    //  for (int j = 0; j < node; j++) {
    //    out2 << adj[i*node + j] << " ";
    //  }
    //  out2 << "\n";
    //}
    //std::ofstream out("data/cora_adj.dat", std::ios::out | std::ios::binary);
    //for (auto f : adj) {
    //  out.write(reinterpret_cast<const char*>(&f), sizeof(float));
    //}
    //out.close();
}

bool GCNParser::isValidInput() {
    return graph_file.is_open() && split_file.is_open() && svmlight_file.is_open();
}

void GCNParser::parseNode() {
    auto &feature_sparse_index = this->gcnData->feature_index;
    auto &feature_val = this->gcnData->feature_value;
    auto &labels = this->gcnData->label;

    feature_sparse_index.indptr.push_back(0);

    int max_idx = 0, max_label = 0;
    while(true) {
        std::string line;
        getline(svmlight_file, line);
        if (svmlight_file.eof()) break;
        feature_sparse_index.indptr.push_back(feature_sparse_index.indptr.back());
        std::istringstream ss(line);

        int label = -1;
        ss >> label;
        labels.push_back(label);
        if (ss.fail()) continue;
        max_label = max(max_label, label);

        while (true) {
            string kv;
            ss >> kv;
            if(ss.fail()) break;
            std::istringstream kv_ss(kv);

            int k;
            float v;
            char col;
            kv_ss >> k >> col >> v;

            feature_val.push_back(v);
            feature_sparse_index.indices.push_back(k);
            feature_sparse_index.indptr.back() += 1;
            max_idx = max(max_idx, k);
        }
    }
    gcnParams->input_dim = max_idx + 1;
    gcnParams->output_dim = max_label + 1;
    std::vector<float> feature_matrix(gcnParams->num_nodes*gcnParams->input_dim, 0);

    for (int i = 0; i < feature_sparse_index.indptr.size() - 1; i++) {
        for (int jj = feature_sparse_index.indptr[i]; jj < feature_sparse_index.indptr[i + 1]; jj++) {
            int j = feature_sparse_index.indices[jj];
            feature_matrix[i * gcnParams->input_dim + j] = feature_val[jj];
        }
    }

    //std::ofstream out1("data/cora_input.txt");
    //for (int i = 0; i < gcnParams->num_nodes; i++) {
    //  for (int j = 0; j < gcnParams->input_dim; j++) {
    //    out1 << feature_matrix[i*gcnParams->input_dim + j] << " ";
    //  }
    //  out1 << "\n";
    //}

    //std::ofstream out("data/cora_input.dat", std::ios::out | std::ios::binary);
    //for (auto f : feature_matrix) {
    //  out.write(reinterpret_cast<const char*>(&f), sizeof(float));
    //}
    //out.close();
    
}

void GCNParser::parseSplit() {
    auto &split = this->gcnData->split;

    while (true) {
        std::string line;
        getline(split_file, line);
        if (split_file.eof()) break;
        split.push_back(std::stoi(line));
    }
}

void vprint(std::vector<int> v){
    for(int i:v)printf("%i ", i);
    printf("\n");
}

bool GCNParser::parse() {
    if (!isValidInput()) return false;
    this->parseGraph();
    std::cout << "Parse Graph Succeeded." << endl;
    this->parseNode();
    std::cout << "Parse Node Succeeded." << endl;
    this->parseSplit();
    std::cout << "Parse Split Succeeded." << endl;
    return true;
}
