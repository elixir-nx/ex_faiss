#include <cstring>
#include <vector>

#include "ex_faiss/nif_util.h"
#include "ex_faiss/index.h"
#include "ex_faiss/clustering.h"

#if defined(__CUDA__)
#include <faiss/gpu/utils/DeviceUtils.h>
#endif

void free_ex_faiss_index(ErlNifEnv * env, void * obj) {
  ex_faiss::ExFaissIndex ** index = (ex_faiss::ExFaissIndex **) obj;
  if (*index != nullptr) {
    delete *index;
    *index = nullptr;
  }
}

void free_ex_faiss_clustering(ErlNifEnv * env, void * obj) {
  ex_faiss::ExFaissClustering ** clustering = (ex_faiss::ExFaissClustering **) obj;
  if (*clustering != nullptr) {
    delete *clustering;
    *clustering = nullptr;
  }
}

static int open_resources(ErlNifEnv* env) {
  const char * mod = "ExFaiss";

  if (!nif::open_resource<ex_faiss::ExFaissIndex *>(env, mod, "Index", free_ex_faiss_index)) {
    return -1;
  }
  if (!nif::open_resource<ex_faiss::ExFaissClustering *>(env, mod, "Clustering", free_ex_faiss_clustering)) {
    return -1;
  }

  return 1;
}

static int load(ErlNifEnv* env, void ** priv, ERL_NIF_TERM load_info) {
  if (open_resources(env) == -1) return -1;

  return 0;
}

ERL_NIF_TERM new_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    nif::error(env, "Bad argument count.");
  }

  int64_t dim;
  std::string description;
  faiss::MetricType metric_type;

  if (!nif::get(env, argv[0], &dim)) {
    return nif::error(env, "Unable to get dimensionality.");
  }
  if (!nif::get(env, argv[1], description)) {
    return nif::error(env, "Unable to get string.");
  }
  if (!nif::get_metric_type(env, argv[2], &metric_type)) {
    return nif::error(env, "Unable to get metric type.");
  }

  ex_faiss::ExFaissIndex * index = new ex_faiss::ExFaissIndex(dim, description.c_str(), metric_type);
  return nif::ok(env, nif::make<ex_faiss::ExFaissIndex *>(env, index));
}

ERL_NIF_TERM clone_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }

  ex_faiss::ExFaissIndex * cloned = (*index)->Clone();
  return nif::ok(env, nif::make<ex_faiss::ExFaissIndex *>(env, cloned));
}

ERL_NIF_TERM write_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;
  std::string fname;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }
  if (!nif::get(env, argv[1], fname)) {
    return nif::error(env, "Unable to get fname.");
  }

  (*index)->WriteToFile(fname.c_str());

  return nif::ok(env);
}

ERL_NIF_TERM read_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  std::string fname;
  int32_t io_flags;

  if (!nif::get(env, argv[0], fname)) {
    return nif::error(env, "Unable to get fname.");
  }
  if (!nif::get(env, argv[1], &io_flags)) {
    return nif::error(env, "Unable to get IO flags.");
  }

  ex_faiss::ExFaissIndex * index = ex_faiss::ReadIndexFromFile(fname.c_str(), io_flags);

  return nif::ok(env, nif::make<ex_faiss::ExFaissIndex *>(env, index));
}

ERL_NIF_TERM add_to_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;
  int64_t n;
  ErlNifBinary data;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }
  if (!nif::get(env, argv[1], &n)) {
    return nif::error(env, "Unable to get n.");
  }
  if (!nif::get_binary(env, argv[2], &data)) {
    return nif::error(env, "Unable to get data.");
  }

  (*index)->Add(n, reinterpret_cast<float *>(data.data));

  return nif::ok(env);
}

ERL_NIF_TERM add_with_ids_to_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;
  int64_t n;
  ErlNifBinary data;
  ErlNifBinary ids;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }
  if (!nif::get(env, argv[1], &n)) {
    return nif::error(env, "Unable to get n.");
  }
  if (!nif::get_binary(env, argv[2], &data)) {
    return nif::error(env, "Unable to get data.");
  }
  if (!nif::get_binary(env, argv[3], &ids)) {
    return nif::error(env, "Unable to get ids.");
  }

  (*index)->AddWithIds(n, reinterpret_cast<float *>(data.data), reinterpret_cast<int64_t *>(ids.data));

  return nif::ok(env);
}

ERL_NIF_TERM search_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;
  int64_t n;
  ErlNifBinary data;
  int64_t k;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }
  if (!nif::get(env, argv[1], &n)) {
    return nif::error(env, "Unable to get n.");
  }
  if (!nif::get_binary(env, argv[2], &data)) {
    return nif::error(env, "Unable to get data.");
  }
  if (!nif::get(env, argv[3], &k)) {
    return nif::error(env, "Unable to get k.");
  }

  ErlNifBinary distances, labels;
  enif_alloc_binary(n * k * sizeof(float), &distances);
  enif_alloc_binary(n * k * sizeof(int64_t), &labels);

  (*index)->Search(n,
                   reinterpret_cast<float *>(data.data),
                   k,
                   reinterpret_cast<float *>(distances.data),
                   reinterpret_cast<int64_t *>(labels.data));

  ERL_NIF_TERM distances_term = nif::make(env, distances);
  ERL_NIF_TERM labels_term = nif::make(env, labels);

  return nif::ok(env, enif_make_tuple2(env, distances_term, labels_term));
}

ERL_NIF_TERM train_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;
  int64_t n;
  ErlNifBinary data;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }
  if (!nif::get(env, argv[1], &n)) {
    return nif::error(env, "Unable to get n.");
  }
  if (!nif::get_binary(env, argv[2], &data)) {
    return nif::error(env, "Unable to get data.");
  }

  (*index)->Train(n, reinterpret_cast<float *>(data.data));

  return nif::ok(env);
}

ERL_NIF_TERM reconstruct_batch_from_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;
  int64_t n;
  ErlNifBinary keys;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }
  if (!nif::get(env, argv[1], &n)) {
    return nif::error(env, "Unable to get n.");
  }
  if (!nif::get_binary(env, argv[2], &keys)) {
    return nif::error(env, "Unable to get keys.");
  }

  int64_t d = (*index)->dim();

  ErlNifBinary reconstruction;
  enif_alloc_binary(n * d * sizeof(float), &reconstruction);

  (*index)->ReconstructBatch(n, reinterpret_cast<int64_t *>(keys.data), reinterpret_cast<float *>(reconstruction.data));

  return nif::ok(env, nif::make(env, reconstruction));
}

ERL_NIF_TERM compute_residuals_from_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;
  int64_t n;
  ErlNifBinary data;
  ErlNifBinary keys;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }
  if (!nif::get(env, argv[1], &n)) {
    return nif::error(env, "Unable to get n.");
  }
  if (!nif::get_binary(env, argv[2], &data)) {
    return nif::error(env, "Unable to get data.");
  }
  if (!nif::get_binary(env, argv[3], &keys)) {
    return nif::error(env, "Unable to get keys.");
  }

  int64_t d = (*index)->dim();

  ErlNifBinary residuals;
  enif_alloc_binary(n * d * sizeof(float), &residuals);

  (*index)->ComputeResiduals(n, 
                             reinterpret_cast<float *>(data.data),
                             reinterpret_cast<float *>(residuals.data),
                             reinterpret_cast<int64_t *>(keys.data));

  return nif::ok(env, nif::make(env, residuals));
}

ERL_NIF_TERM reset_index(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }

  (*index)->Reset();

  return nif::ok(env);
}

ERL_NIF_TERM get_index_dim(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }

  int dim = (*index)->dim();

  return nif::ok(env, nif::make(env, dim));
}

ERL_NIF_TERM get_index_n_vectors(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }

  int64_t n_total = (*index)->n_total();

  return nif::ok(env, nif::make(env, n_total));
}

ERL_NIF_TERM index_cpu_to_gpu(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissIndex ** index;
  int device;

  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[0], index)) {
    return nif::error(env, "Unable to get index.");
  }
  if (!nif::get(env, argv[1], &device)) {
    return nif::error(env, "Unable to get device.");
  }

  ex_faiss::ExFaissIndex * gpu_index = (*index)->CloneToGpu(device);

  return nif::ok(env, nif::make<ex_faiss::ExFaissIndex *>(env, gpu_index));
}

ERL_NIF_TERM get_num_gpus(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return nif::error(env, "Bad argument count.");
  }

  int gpus;

  #if defined(__CUDA__)
    gpus = faiss::gpu::getNumDevices();
  #else
    gpus = 0;
  #endif

  return nif::ok(env, nif::make(env, gpus));
}

ERL_NIF_TERM new_clustering(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return nif::error(env, "Bad argument count.");
  }

  int d;
  int k;

  if (!nif::get(env, argv[0], &d)) {
    return nif::error(env, "Unable to get d.");
  }
  if (!nif::get(env, argv[1], &k)) {
    return nif::error(env, "Unable to get k.");
  }

  ex_faiss::ExFaissClustering * clustering = new ex_faiss::ExFaissClustering(d, k);

  return nif::ok(env, nif::make<ex_faiss::ExFaissClustering *>(env, clustering));
}

ERL_NIF_TERM train_clustering(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissClustering ** clustering;
  int64_t n;
  ErlNifBinary data;
  ex_faiss::ExFaissIndex ** index;

  if (!nif::get<ex_faiss::ExFaissClustering *>(env, argv[0], clustering)) {
    return nif::error(env, "Unable to get clustering.");
  }
  if (!nif::get(env, argv[1], &n)) {
    return nif::error(env, "Unable to get n.");
  }
  if (!nif::get_binary(env, argv[2], &data)) {
    return nif::error(env, "Unable to get data.");
  }
  if (!nif::get<ex_faiss::ExFaissIndex *>(env, argv[3], index)) {
    return nif::error(env, "Unable to get index.");
  }

  (*clustering)->Train(n, reinterpret_cast<float *>(data.data), *index);

  return nif::ok(env);
}

ERL_NIF_TERM get_clustering_centroids(ErlNifEnv * env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return nif::error(env, "Bad argument count.");
  }

  ex_faiss::ExFaissClustering ** clustering;

  if (!nif::get<ex_faiss::ExFaissClustering *>(env, argv[0], clustering)) {
    return nif::error(env, "Unable to get clustering.");
  }

  size_t d, k;
  d = (*clustering)->dimensionality();
  k = (*clustering)->n_centroids();

  ErlNifBinary data;
  enif_alloc_binary(d * k * sizeof(float), &data);

  std::vector<float> centroids = (*clustering)->centroids();
  std::memcpy(data.data, centroids.data(), data.size);

  return nif::ok(env, nif::make(env, data));
}

static ErlNifFunc ex_faiss_funcs[] = {
  // Index CPU
  {"new_index", 3, new_index},
  {"clone_index", 1, clone_index},
  {"write_index", 2, write_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"read_index", 2, read_index, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"add_to_index", 3, add_to_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"add_with_ids_to_index", 4, add_with_ids_to_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"search_index", 4, search_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"train_index", 3, train_index, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"reset_index", 1, reset_index},
  {"reconstruct_batch_from_index", 3, reconstruct_batch_from_index},
  {"compute_residuals_from_index", 4, compute_residuals_from_index},
  {"get_index_dim", 1, get_index_dim},
  {"get_index_n_vectors", 1, get_index_n_vectors},
  // Index GPU
  {"index_cpu_to_gpu", 2, index_cpu_to_gpu, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"get_num_gpus", 0, get_num_gpus},
  // Clustering CPU
  {"new_clustering", 2, new_clustering, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"train_clustering", 4, train_clustering, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"get_clustering_centroids", 1, get_clustering_centroids}
};

ERL_NIF_INIT(Elixir.ExFaiss.NIF, ex_faiss_funcs, &load, NULL, NULL, NULL);