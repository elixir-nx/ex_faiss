defmodule ExFaiss.ClusteringTest do
  use ExUnit.Case

  alias ExFaiss.Clustering
  alias ExFaiss.Index

  describe "new" do
    test "creates a new clustering" do
      assert %Clustering{k: 10, ref: _, index: %Index{dim: 128}} = Clustering.new(128, 10)
    end
  end

  describe "train" do
    test "trains a cluster and adds to index" do
      trained =
        Clustering.new(128, 10)
        |> Clustering.train(Nx.random_uniform({100, 128}))

      assert %Clustering{k: 10, ref: _, index: %Index{dim: 128} = index} = trained
      assert Index.get_num_vectors(index) == 10
    end
  end
end
