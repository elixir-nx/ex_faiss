defmodule ExFaiss.IndexTest do
  use ExUnit.Case

  alias ExFaiss.Index

  describe "new" do
    test "creates indices from descriptions" do
      assert %Index{} =
               ExFaiss.Index.new(
                 512,
                 "OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)"
               )

      assert %Index{} = ExFaiss.Index.new(128, "PCA80,Flat")
      assert %Index{} = ExFaiss.Index.new(128, "OPQ16_64,IMI2x8,PQ8+16")
      assert %Index{} = ExFaiss.Index.new(512, "Flat", metric: :jensenshannon)
    end

    @tag :cuda
    test "creates gpu indices from descriptions" do
      # TODO: Unsupported clone?
      # assert %Index{device: {:cuda, 0}} =
      #          ExFaiss.Index.new(
      #            512,
      #            "OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)",
      #            device: {:cuda, 0}
      #          )

      assert %Index{device: {:cuda, 0}} = ExFaiss.Index.new(128, "PCA80,Flat", device: {:cuda, 0})

      assert %Index{device: {:cuda, 0}} =
               ExFaiss.Index.new(128, "OPQ16_64,IMI2x8,PQ8+16", device: {:cuda, 0})

      assert %Index{device: {:cuda, 0}} =
               ExFaiss.Index.new(512, "Flat", metric: :jensenshannon, device: {:cuda, 0})
    end
  end

  describe "clone" do
    test "creates clones of index" do
      assert %Index{ref: ref1} = index = ExFaiss.Index.new(512, "Flat")

      assert %Index{ref: ref2} = ExFaiss.Index.clone(index)

      assert ref1 != ref2
    end
  end

  describe "add" do
    test "adds valid tensors" do
      index = ExFaiss.Index.new(512, "Flat")

      assert %Index{} = ExFaiss.Index.add(index, Nx.random_uniform({512}))
      assert %Index{} = ExFaiss.Index.add(index, Nx.random_uniform({2, 512}))
    end

    @tag :cuda
    test "adds valid tensors to gpu index" do
      index = ExFaiss.Index.new(512, "Flat", device: {:cuda, 0})

      assert %Index{} = ExFaiss.Index.add(index, Nx.random_uniform({512}))
      assert %Index{} = ExFaiss.Index.add(index, Nx.random_uniform({2, 512}))
    end

    test "raises on invalid types" do
      index1 = ExFaiss.Index.new(128, "Flat")

      assert_raise ArgumentError, ~r/invalid type/, fn ->
        ExFaiss.Index.add(index1, Nx.random_uniform({2, 128}, type: :f64))
      end
    end

    test "raises on invalid shapes" do
      index1 = ExFaiss.Index.new(128, "Flat")

      assert_raise ArgumentError, ~r/invalid shape/, fn ->
        ExFaiss.Index.add(index1, Nx.random_uniform({2, 3, 4}))
      end

      assert_raise ArgumentError, ~r/invalid shape/, fn ->
        ExFaiss.Index.add(index1, Nx.random_uniform({2, 256}))
      end
    end
  end

  describe "search" do
    test "searches a simple flat index" do
      index =
        ExFaiss.Index.new(1, "Flat", metric: :l1)
        |> ExFaiss.Index.add(Nx.iota({64, 1}, type: :f32))

      assert %{distances: distances, labels: labels} =
               ExFaiss.Index.search(index, Nx.tensor([0.0]), 32)

      assert distances == Nx.iota({1, 32}, type: :f32)
      assert labels == Nx.iota({1, 32})
    end

    @tag :cuda
    test "searches a simple flat gpu index" do
      index =
        ExFaiss.Index.new(1, "Flat", metric: :l1, device: {:cuda, 0})
        |> ExFaiss.Index.add(Nx.iota({64, 1}, type: :f32))

      assert %{distances: distances, labels: labels} =
               ExFaiss.Index.search(index, Nx.tensor([0.0]), 32)

      assert distances == Nx.iota({1, 32}, type: :f32)
      assert labels == Nx.iota({1, 32})
    end
  end

  describe "train" do
    test "trains an index" do
      index =
        ExFaiss.Index.new(10, "HNSW,Flat")
        |> ExFaiss.Index.train(Nx.random_uniform({100, 10}))
        |> ExFaiss.Index.add(Nx.random_uniform({100, 10}))

      assert %Index{} = index
    end

    @tag :cuda
    test "trains an index on gpu" do
      index =
        ExFaiss.Index.new(10, "HNSW,Flat", device: {:cuda, 0})
        |> ExFaiss.Index.train(Nx.random_uniform({100, 10}))
        |> ExFaiss.Index.add(Nx.random_uniform({100, 10}))

      assert %Index{} = index
    end
  end

  describe "reconstruct" do
    test "reconstructs vectors from keys" do
      data = Nx.random_uniform({1, 128})

      result =
        ExFaiss.Index.new(128, "Flat")
        |> ExFaiss.Index.add(data)
        |> ExFaiss.Index.reconstruct(Nx.tensor([0]))

      assert result == data
    end
  end

  describe "compute_residuals" do
    test "computes residuals from data and keys" do
      data = Nx.broadcast(0.0, {1, 128})

      result =
        ExFaiss.Index.new(128, "Flat")
        |> ExFaiss.Index.add(data)
        |> ExFaiss.Index.compute_residuals(data, Nx.tensor([0]))

      assert result == data
    end
  end

  describe "memory" do
    @tag :slow
    test "does not leak" do
      for _ <- 1..100 do
        ExFaiss.Index.new(128, "Flat")
        |> ExFaiss.Index.add(Nx.random_uniform({500, 128}))

        :erlang.garbage_collect()
      end
    end

    @tag :slow
    test "does not leak on gpu" do
      for _ <- 1..100 do
        ExFaiss.Index.new(128, "Flat", device: {:cuda, 0})
        |> ExFaiss.Index.add(Nx.random_uniform({500, 128}))

        :erlang.garbage_collect()
      end
    end
  end

  describe "multi-device" do
    @describetag :multi_device

    test "creates an on non-default device" do
      %Index{device: {:cuda, 1}} = ExFaiss.Index.new(128, "Flat", device: {:cuda, 1})
    end
  end
end
