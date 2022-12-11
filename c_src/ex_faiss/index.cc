#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/clone_index.h>

#if defined(__CUDA__)
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#endif

#include "index.h"

namespace ex_faiss {

ExFaissIndex::ExFaissIndex(faiss::Index * index) {
  index_ = std::unique_ptr<faiss::Index>(index);
}

ExFaissIndex::ExFaissIndex(int dim,
                           const char * description,
                           faiss::MetricType metric_type) {
  faiss::Index * index = faiss::index_factory(dim, description, metric_type);
  index_ = std::unique_ptr<faiss::Index>(index);
}

ExFaissIndex * ExFaissIndex::Clone() {
  faiss::Index * index = faiss::clone_index(index_.get());
  return new ExFaissIndex(index);
}

ExFaissIndex * ExFaissIndex::CloneToGpu(int device) {
  #if defined(__CUDA__)
    faiss::gpu::StandardGpuResources res;
    faiss::Index * index = faiss::gpu::index_cpu_to_gpu(&res, device, index_.get());
    return new ExFaissIndex(index);
  #else
    return nullptr;
  #endif
}

void ExFaissIndex::Add(int64_t n, const float * x) {
  index_->add(n, x);
}

void ExFaissIndex::AddWithIds(int64_t n, const float * x, const int64_t * xids) {
  index_->add_with_ids(n, x, xids);
}

void ExFaissIndex::Search(int64_t n, const float * x, int64_t k, float * distances, int64_t * labels) {
  index_->search(n, x, k, distances, labels);
}

void ExFaissIndex::Train(int64_t n, const float * x) {
  index_->train(n, x);
}

void ExFaissIndex::Reset() {
  index_->reset();
}

void ExFaissIndex::ReconstructBatch(int64_t n, const int64_t * keys, float * recons) {
  index_->reconstruct_batch(n, keys, recons);
}

void ExFaissIndex::ComputeResiduals(int64_t n, const float * data, float * resid, const int64_t * keys) {
  index_->compute_residual_n(n, data, resid, keys);
}

void ExFaissIndex::WriteToFile(const char * fname) {
  faiss::write_index(index_.get(), fname);
}

ExFaissIndex * ReadIndexFromFile(const char * fname, int io_flags) {
  faiss::Index * index = faiss::read_index(fname, io_flags);
  return new ExFaissIndex(index);
}
} // namespace ex_faiss