# ExFaiss

Elixir front-end for [Facebook AI Similarity Search (Faiss)](https://github.com/facebookresearch/faiss).

ExFaiss is a low-level wrapper around Faiss which allows you to create and manage Faiss indices and clusterings. Faiss enables efficient search and clustering of dense vectors and has the potential to scale to millions, billions, and even trillions of vectors. ExFaiss works directly with [Nx](https://github.com/elixir-nx/nx) tensors, so you can seamlessly integrate ExFaiss into your existing Elixir ML workflows.

## Installation

Add `ex_faiss` to your dependencies:

```elixir
def deps do
  [
    {:ex_faiss, github: "elixir-nx/ex_faiss"}
  ]
end
```

ExFaiss will download, build, and cache Faiss on the first compilation. You must have CMake installed in order to build Faiss.

### GPU Installation

If you have an NVIDIA GPU with CUDA installed, you can enable the GPU build by setting the environment variable `USE_CUDA=true`. Note that if you have already built Faiss without GPU support, you will need to delete the cached build before continuuing. You can clean the existing installation by running `make clean`.

## Working with Indices

You can create indices which follow the syntax of Faiss' [Index Factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory). Indices require you to also specify a dimensionality of the vectors you plan to store:

```elixir
index = ExFaiss.Index.new(128, "Flat")
```

You can optionally place an index on a GPU by specifying the `:device` option:

```elixir
index = ExFaiss.Index.new(128, "Flat", device: :cuda)
```

Finally, you can add one or more tensors to the index at a time:

```elixir
index = ExFaiss.Index.add(index, Nx.random_uniform({32, 128}))
```

And then search the index for similar vectors:

```elixir
result = ExFaiss.Index.search(index, Nx.random_uniform({128}), 5)
```

Returns:

```
%{
  distances: #Nx.Tensor<
    f32[1][5]
    [
      [18.473186492919922, 18.697336196899414, 19.020721435546875, 19.091503143310547, 19.53148078918457]
    ]
  >,
  labels: #Nx.Tensor<
    s64[1][5]
    [
      [25, 0, 2, 9, 13]
    ]
  >
}
```

## License

```
Copyright (c) 2022 The Machine Learning Working Group of the Erlang Ecosystem Foundation

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
```
