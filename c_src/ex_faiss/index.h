#ifndef EX_FAISS_INDEX_H_
#define EX_FAISS_INDEX_H_

#include <memory>
#include <cstdint>
#include <faiss/Index.h>

namespace ex_faiss {

class ExFaissIndex {
 public:
  ExFaissIndex(faiss::Index * index);

  ExFaissIndex(int d, const char * description, faiss::MetricType metric_type);

  void Add(int64_t n, const float * x);

  void AddWithIds(int64_t n, const float * x, const int64_t * xids);

  void Search(int64_t n,
              const float * x,
              int64_t k,
              float * distances,
              int64_t * labels);

  void Train(int64_t n, const float * x);

  void WriteToFile(const char * fname);

  ExFaissIndex * Clone();

  ExFaissIndex * CloneToGpu(int device);

  void Reset();

  void ReconstructBatch(int64_t n, const int64_t * keys, float * recons);

  void ComputeResiduals(int64_t n, const float * data, float * resid, const int64_t * keys);

  faiss::Index * index() { return index_.get(); }
  int dim() { return index_->d; }
  int64_t n_total() { return index_->ntotal; }

 private:
  std::unique_ptr<faiss::Index> index_;
};

ExFaissIndex * ReadIndexFromFile(const char * fname, int io_flags);

} // namespace ex_faiss
#endif