#include "nif_util.h"

namespace nif {

  ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term) {
    return enif_make_tuple2(env, ok(env), term);
  }

  ERL_NIF_TERM ok(ErlNifEnv* env) {
    return enif_make_atom(env, "ok");
  }

  ERL_NIF_TERM error(ErlNifEnv * env, const char * msg) {
    ERL_NIF_TERM atom = enif_make_atom(env, "error");
    ERL_NIF_TERM msg_term = enif_make_string(env, msg, ERL_NIF_LATIN1);
    return enif_make_tuple2(env, atom, msg_term);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, int var) {
    return enif_make_int(env, var);
  }

  ERL_NIF_TERM make(ErlNifEnv * env, int64_t var) {
    return enif_make_int64(env, var);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary var) {
    return enif_make_binary(env, &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int32_t * var) {
    return enif_get_int(env, term,
                        reinterpret_cast<int32_t *>(var));
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int64_t * var) {
    return enif_get_int64(env, term,
                          reinterpret_cast<nif_int64_t *>(var));
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var) {
    unsigned len;
    int ret = enif_get_list_length(env, term, &len);

    if (!ret) {
      ErlNifBinary bin;
      ret = enif_inspect_binary(env, term, &bin);
      if (!ret) {
        return 0;
      }
      var = std::string((const char*)bin.data, bin.size);
      return ret;
    }

    var.resize(len+1);
    ret = enif_get_string(env, term, &*(var.begin()), var.size(), ERL_NIF_LATIN1);

    if (ret > 0) {
      var.resize(ret-1);
    } else if (ret == 0) {
      var.resize(0);
    } else {}

    return ret;
  }

  int get_metric_type(ErlNifEnv * env, ERL_NIF_TERM term, faiss::MetricType * metric_type) {
    int value;
    if (!enif_get_int(env, term, &value)) return 0;
    *metric_type = faiss::MetricType(value);
    return 1;
  }

  int get_binary(ErlNifEnv * env, ERL_NIF_TERM term, ErlNifBinary * var) {
    return enif_inspect_binary(env, term, var);
  }

  int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<int64_t> &var) {
    unsigned int length;
    if (!enif_get_list_length(env, list, &length)) return 0;
    var.reserve(length);
    ERL_NIF_TERM head, tail;

    while (enif_get_list_cell(env, list, &head, &tail)) {
      int64_t elem;
      if (!get(env, head, &elem)) return 0;
      var.push_back(elem);
      list = tail;
    }
    return 1;
  }

}