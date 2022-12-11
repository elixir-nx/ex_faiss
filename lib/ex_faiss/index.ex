defmodule ExFaiss.Index do
  @moduledoc """
  Wraps references to a Faiss index.
  """
  alias __MODULE__
  import ExFaiss.Shared

  defstruct [:dim, :ref, :device]

  # TODO: In all of these results, we copy the underlying data with to_binary
  # but FAISS does not take ownership of the data, so there must be a way we
  # can just provide a view of the data without copying. I think it may
  # be backend specific though

  # TODO: Anything that just returns :ok right now should return
  # the index

  # TODO: Handle selectors in search

  # TODO: Handle :errors from C++ exceptions

  # TODO: Change order of new arguments
  # TODO: Add description to struct

  @doc """
  Creates a new Faiss index which stores vectors
  of the given dimensionality `dim`.

  ## Options

    * `:metric` - metric type. One of [:l2]

    * `:device` - device type. One of `:host`, `:cuda`, or
      `{:cuda, device}` where device is an integer device
      ordinal
  """
  def new(dim, description, opts \\ []) when is_integer(dim) and dim > 0 do
    # TODO: Handle Index factory description as options
    # TODO: Maybe have sigil to construct factory descriptions
    opts = Keyword.validate!(opts, metric: :l2, device: :host)
    metric_type = metric_type_to_int(opts[:metric])

    ref = ExFaiss.NIF.new_index(dim, description, metric_type) |> unwrap!()

    case opts[:device] do
      :cuda ->
        new_gpu_index(ref, dim, 0)

      {:cuda, device} ->
        new_gpu_index(ref, dim, device)

      :host ->
        %Index{dim: dim, ref: ref, device: :host}

      device ->
        raise ArgumentError, "invalid device #{inspect(device)}"
    end
  end

  # TODO: Handle replicated index
  defp new_gpu_index(ref, dim, device) do
    devices = ExFaiss.NIF.get_num_gpus() |> unwrap!()

    cond do
      devices <= 0 ->
        raise ArgumentError,
              "no gpu devices found, please ensure you've set" <>
                " the environment variable USE_CUDA=true, and" <>
                " that you have CUDA enabled devices"

      devices < device or device < 0 ->
        raise ArgumentError,
              "device #{inspect(device)} is out of bounds for" <>
                " number of devices #{inspect(devices)}"

      true ->
        ref = ExFaiss.NIF.index_cpu_to_gpu(ref, device) |> unwrap!()
        %Index{ref: ref, dim: dim, device: {:cuda, device}}
    end
  end

  @doc """
  Adds the given tensors to the given index.
  """
  def add(%Index{dim: dim, ref: ref} = index, %Nx.Tensor{} = tensor) do
    validate_type!(tensor, {:f, 32})

    case Nx.shape(tensor) do
      {^dim} ->
        data = Nx.to_binary(tensor)
        ExFaiss.NIF.add_to_index(ref, 1, data)

      {n, ^dim} ->
        data = Nx.to_binary(tensor)
        ExFaiss.NIF.add_to_index(ref, n, data)

      shape ->
        invalid_shape_error!(dim, shape)
    end

    index
  end

  @doc """
  Adds the given tensors and IDs to the given index.
  """
  def add_with_ids(%Index{dim: dim, ref: ref} = index, %Nx.Tensor{} = tensor, %Nx.Tensor{} = ids) do
    validate_type!(tensor, {:f, 32})
    validate_type!(ids, {:s, 64})

    case {Nx.shape(tensor), Nx.shape(ids)} do
      {{^dim}, {1}} ->
        data = Nx.to_binary(tensor)
        xids = Nx.to_binary(ids)
        ExFaiss.NIF.add_with_ids_to_index(ref, 1, data, xids)

      {{n, ^dim}, {n}} ->
        data = Nx.to_binary(tensor)
        xids = Nx.to_binary(ids)
        ExFaiss.NIF.add_with_ids_to_index(ref, n, data, xids)

      {tensor_shape, ids_shape} ->
        raise ArgumentError,
              "invalid shape for index with dim #{inspect(dim)}," <>
                " tensor shape must be rank-1 or rank-2 with trailing" <>
                " dimension equal to dimension of the index, while ids" <>
                " shape must be rank-1 with dimension equal to leading" <>
                " dimension of data, or 1 if data is rank-1, got shapes" <>
                " ids: #{inspect(ids_shape)}, embeddings: #{inspect(tensor_shape)}"
    end

    index
  end

  @doc """
  Searches the given index for the top `k` matches
  close to the given query vector.

  The result is a map with keys `:labels` and `:distances`
  which represent the index ID and pairwise distances from
  the query vector for each result vector.
  """
  def search(%Index{dim: dim, ref: index}, %Nx.Tensor{} = tensor, k)
      when is_integer(k) and k > 0 do
    validate_type!(tensor, {:f, 32})

    case Nx.shape(tensor) do
      {^dim} ->
        data = Nx.to_binary(tensor)
        {distances, labels} = ExFaiss.NIF.search_index(index, 1, data, k) |> unwrap!()

        %{
          distances: distances |> Nx.from_binary(:f32) |> Nx.reshape({1, k}),
          labels: labels |> Nx.from_binary(:s64) |> Nx.reshape({1, k})
        }

      {n, ^dim} ->
        data = Nx.to_binary(tensor)
        {distances, labels} = ExFaiss.NIF.search_index(index, n, data, k) |> unwrap!()

        %{
          distances: distances |> Nx.from_binary(:f32) |> Nx.reshape({n, k}),
          labels: labels |> Nx.from_binary(:s64) |> Nx.reshape({n, k})
        }

      shape ->
        invalid_shape_error!(dim, shape)
    end
  end

  @doc """
  Trains an index on a representative set of vectors.
  """
  def train(%Index{dim: dim, ref: ref} = index, %Nx.Tensor{} = tensor) do
    validate_type!(tensor, {:f, 32})

    case Nx.shape(tensor) do
      {n, ^dim} ->
        data = Nx.to_binary(tensor)
        ExFaiss.NIF.train_index(ref, n, data)

      shape ->
        invalid_shape_error!(dim, shape)
    end

    index
  end

  @doc """
  Creates a copy of the given index.
  """
  def clone(%Index{dim: dim, ref: index}) do
    ref = ExFaiss.NIF.clone_index(index) |> unwrap!()
    %Index{dim: dim, ref: ref}
  end

  @doc """
  Reconstructs stored vectors at the given indices.
  """
  def reconstruct(%Index{dim: dim, ref: index}, %Nx.Tensor{} = keys) do
    n =
      case Nx.shape(keys) do
        {n} ->
          n

        {} ->
          1
      end

    keys_data = Nx.to_binary(keys)

    index
    |> ExFaiss.NIF.reconstruct_batch_from_index(n, keys_data)
    |> unwrap!()
    |> Nx.from_binary(:f32)
    |> Nx.reshape({n, dim})
  end

  @doc """
  Computes residuals after indexing.
  """
  def compute_residuals(%Index{dim: dim, ref: index}, %Nx.Tensor{} = xs, %Nx.Tensor{} = keys) do
    n =
      case {Nx.shape(xs), Nx.shape(keys)} do
        {{^dim}, {1}} ->
          1

        {{n, ^dim}, {n}} ->
          n

        {tensor_shape, ids_shape} ->
          raise ArgumentError,
                "invalid shape for index with dim #{inspect(dim)}," <>
                  " tensor shape must be rank-1 or rank-2 with trailing" <>
                  " dimension equal to dimension of the index, while ids" <>
                  " shape must be rank-1 with dimension equal to leading" <>
                  " dimension of data, or 1 if data is rank-1, got shapes" <>
                  " ids: #{inspect(ids_shape)}, embeddings: #{inspect(tensor_shape)}"
      end

    xs_data = Nx.to_binary(xs)
    keys_data = Nx.to_binary(keys)

    index
    |> ExFaiss.NIF.compute_residuals_from_index(n, xs_data, keys_data)
    |> unwrap!()
    |> Nx.from_binary(:f32)
    |> Nx.reshape({n, dim})
  end

  @doc """
  Writes an index to a file.
  """
  def to_file(%Index{ref: index}, fname) do
    :ok = ExFaiss.NIF.write_index(index, fname)
  end

  @doc """
  Reads an index from a file.
  """
  def from_file(fname, io_flags) do
    ref = ExFaiss.NIF.read_index(fname, io_flags) |> unwrap!()
    dim = ExFaiss.NIF.get_index_dim(ref) |> unwrap!()
    %Index{dim: dim, ref: ref}
  end

  @doc """
  Gets the number of vectors in the index.
  """
  def get_num_vectors(%Index{ref: index}) do
    ExFaiss.NIF.get_index_n_vectors(index) |> unwrap!()
  end

  defp invalid_shape_error!(dim, shape) do
    raise ArgumentError,
          "invalid shape for index with dim #{inspect(dim)}," <>
            " tensor shape must be rank-1 or rank-2 with trailing" <>
            " dimension equal to dimension of the index, got shape" <>
            " #{inspect(shape)}"
  end

  defp metric_type_to_int(:inner_product), do: 0
  defp metric_type_to_int(:l2), do: 1
  defp metric_type_to_int(:l1), do: 2
  defp metric_type_to_int(:linf), do: 3
  defp metric_type_to_int(:lp), do: 4
  defp metric_type_to_int(:canberra), do: 20
  defp metric_type_to_int(:braycurtis), do: 21
  defp metric_type_to_int(:jensenshannon), do: 22
  defp metric_type_to_int(invalid), do: raise(ArgumentError, "invalid metric #{inspect(invalid)}")
end
