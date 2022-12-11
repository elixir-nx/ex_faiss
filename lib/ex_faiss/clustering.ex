defmodule ExFaiss.Clustering do
  @moduledoc """
  Wraps references to Faiss clustering.
  """
  alias __MODULE__
  alias ExFaiss.Index
  import ExFaiss.Shared
  require Logger

  defstruct [:ref, :k, :index, :trained?]

  @doc """
  Creates a new Faiss clustering object.
  """
  def new(d, k, _opts \\ []) do
    # TODO: Handle options
    # TODO: Create correct index
    cluster = ExFaiss.NIF.new_clustering(d, k) |> unwrap!()
    index = Index.new(d, "Flat")
    %Clustering{ref: cluster, index: index, k: k}
  end

  @doc """
  Trains a Faiss clustering object.
  """
  def train(
        %Clustering{ref: clustering, index: %Index{dim: dim, ref: index}} = cluster,
        %Nx.Tensor{} = tensor
      ) do
    validate_type!(tensor, {:f, 32})

    case Nx.shape(tensor) do
      {^dim} ->
        # TODO: Warn?
        data = Nx.to_binary(tensor)
        ExFaiss.NIF.train_clustering(clustering, 1, data, index)

      {n, ^dim} ->
        data = Nx.to_binary(tensor)
        ExFaiss.NIF.train_clustering(clustering, n, data, index)

      shape ->
        raise ArgumentError,
              "invalid shape for index with dim #{inspect(dim)}," <>
                " tensor shape must be rank-1 or rank-2 with trailing" <>
                " dimension equal to dimension of the index, got shape" <>
                " #{inspect(shape)}"
    end

    %{cluster | trained?: true}
  end

  @doc """
  Returns cluster assignment for given embedding.
  """
  def get_cluster_assignment(
        %Clustering{trained?: true, index: %Index{} = index},
        %Nx.Tensor{} = tensor
      ) do
    Index.search(index, tensor, 1)
  end

  def get_cluster_assignment(_, _) do
    raise ArgumentError, "cannot get cluster assignments for un-trained clustering"
  end

  @doc """
  Returns clustering centroids of given clustering.
  """
  def get_centroids(%Clustering{trained?: true, ref: clustering, k: k, index: %Index{dim: d}}) do
    centroids_data = ExFaiss.NIF.get_clustering_centroids(clustering) |> unwrap!()

    centroids_data
    |> Nx.from_binary(:f32)
    |> Nx.reshape({k, d})
  end
end
