#include <memory>

#include <faiss/Clustering.h>

#include "index.h"
#include "clustering.h"

namespace ex_faiss {

  ExFaissClustering::ExFaissClustering(int d, int k) {
    clustering_ = std::make_unique<faiss::Clustering>(d, k);
  }

  void ExFaissClustering::Train(int64_t n, const float * x, ExFaissIndex * index) {
    clustering_->train(n, x, *(index->index()));
  }

} // namespace ex_faiss