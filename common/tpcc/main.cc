#include "omp.h"
#include "mkl.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <functional>
#include <valarray>
#include <algorithm>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstdlib>
#include <string>
#include "math.h"
#include "worker.h"

using namespace std;

const std::array<std::string,8> __partition_names = {
  string("lb"),string("bl"),string("fl"),string("lf"),
  string("rf"),string("fr"),string("br"),string("rb")
};
const double __partition_size = 2 * M_PI / __partition_names.size();

// provides tpcc result performed on temporal differences
std::string tpcc_string(double angle1, double angle2) {
  double anglediff = angle2 - angle1;
  int partition = (int) floor( anglediff / __partition_size ) + 4;
  return __partition_names[partition];
}

// concordance of the estimated model, computational model using intervals to determine the tpcc
// assign responsible order for Concordance
double ModelConcordance(double * u, double * grad,
const int start_index, const int end_index, const int norm, const double dx, 
std::vector<int>& vec, std::map<string, int>& tpcc_map, const float graddx = 1.0f) {
  double sum = 0.0f;
  double numerator = 0.0f;
  double denominator = 0.0f;
  var index1;
  var index2;
  for(int i = start_index; i < end_index-1; i++) {
    index1 = loop_index((var) i, dx, calibrated_length);
    index2 = loop_index((var) (i+1), dx, calibrated_length);
    numerator = std::pow(grad[i] * graddx, norm);
    denominator = std::pow(u[i], norm);
    sum += numerator / denominator;
    if ((denominator <= std::pow(quantisation_factor, norm)) && grad[i] > 0) {
      vec[0] += 1;
    } else {
      vec[1] += 1;
    }
    double angle1 = arg((double) index1, u[i], dx, grad[i] + (grad[i+1] - grad[i]) * dx);
    double angle2 = arg((double) index2, u[i+1], dx, grad[i+1]);
    std::string tpcc = tpcc_string(angle1, angle2);
    tpcc_map[tpcc] += 1;
  }
  return sum;
}

