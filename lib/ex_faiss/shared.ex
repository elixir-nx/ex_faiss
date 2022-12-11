defmodule ExFaiss.Shared do
  @moduledoc false

  def validate_type!(tensor, type) do
    unless Nx.type(tensor) == type do
      raise ArgumentError,
            "invalid type #{inspect(Nx.type(tensor))}, vector type" <>
              " must be #{inspect(type)}"
    end
  end

  def unwrap!({:ok, val}), do: val
  def unwrap!({:error, reason}), do: raise(reason)
end
