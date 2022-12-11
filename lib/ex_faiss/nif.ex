defmodule ExFaiss.NIF do
  @on_load :__on_load__

  def __on_load__ do
    path = :filename.join(:code.priv_dir(:ex_faiss), ~c"libex_faiss")
    :erlang.load_nif(path, 0)
  end

  # Index operations
  def new_index(_dim, _description, _metric), do: :erlang.nif_error(:undef)
  def clone_index(_index), do: :erlang.nif_error(:undef)
  def add_to_index(_index, _dim, _data), do: :erlang.nif_error(:undef)
  def add_with_ids_to_index(_index, _dim, _data, _ids), do: :erlang.nif_error(:undef)
  def search_index(_index, _n, _data, _k), do: :erlang.nif_error(:undef)
  def train_index(_index, _n, _data), do: :erlang.nif_error(:undef)
  def reset_index(_index), do: :erlang.nif_error(:undef)
  def reconstruct_batch_from_index(_index, _n, _data), do: :erlang.nif_error(:undef)
  def compute_residuals_from_index(_index, _n, _data, _keys), do: :erlang.nif_error(:undef)
  def write_index(_index, _fname), do: :erlang.nif_error(:undef)
  def read_index(_fname, _io_flags), do: :erlang.nif_error(:undef)
  def get_index_dim(_index), do: :erlang.nif_error(:undef)
  def get_index_n_vectors(_index), do: :erlang.nif_error(:undef)

  # Gpu operations
  def index_cpu_to_gpu(_index, _device), do: :erlang.nif_error(:undef)
  def get_num_gpus(), do: :erlang.nif_error(:undef)

  # Clustering operations
  def new_clustering(_dim, _k), do: :erlang.nif_error(:undef)
  def train_clustering(_clustering, _n, _data, _index), do: :erlang.nif_error(:undef)
  def get_clustering_centroids(_clustering), do: :erlang.nif_error(:undef)
end
