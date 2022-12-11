#ifndef EX_FAISS_CLUSTERING_H_
#define EX_FAISS_CLUSTERING_H_

#include "index.h"
#include <faiss/Clustering.h>

namespace ex_faiss {

class ExFaissClustering {
 public:
  ExFaissClustering(int d, int k);

  // TODO: Handle weights
  void Train(int64_t n, const float * x, ExFaissIndex * index);

  std::vector<float> centroids() { return clustering_->centroids; }
  size_t dimensionality() { return clustering_->d; }
  size_t n_centroids() { return clustering_->k; }
  std::vector<faiss::ClusteringIterationStats> iteration_stats() { return clustering_->iteration_stats; }

 private:
  std::unique_ptr<faiss::Clustering> clustering_;
};

} // namespace ex_faiss
#endif