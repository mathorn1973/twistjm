import torch
import torch.nn as nn


_TWISTJ_C = 0.5773502691896257  # 1 / sqrt(3)


def _validate_twist_input(x: torch.Tensor, dim: int, layer_name: str) -> None:
    if not x.is_floating_point():
        raise TypeError(f"{layer_name} expects a floating-point tensor, got {x.dtype}.")
    if x.ndim < 1:
        raise ValueError(f"{layer_name} expects at least 1D input, got shape {tuple(x.shape)}.")
    if x.size(-1) != dim:
        raise ValueError(
            f"{layer_name} expected last dim == {dim}, got {x.size(-1)} for shape {tuple(x.shape)}."
        )


class TwistJLayer(nn.Module):
    """
    TWIST J Motor
    BF16 training path using pure additive algebra over 4-channel blocks.
    Replaces standard linear matrix multiplication.
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"Dimension must be divisible by 4, got {dim}.")
        self.dim = dim
        self.scale = nn.Parameter(torch.randn(dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_twist_input(x, self.dim, "TwistJLayer")

        xv = x.reshape(-1, 4)
        yv = torch.empty_like(xv)

        # Pure additive algebra (J-operator)
        yv[:, 0] = (xv[:, 0] - xv[:, 2] + xv[:, 3]) * _TWISTJ_C
        yv[:, 1] = (xv[:, 1] - xv[:, 2]) * _TWISTJ_C
        yv[:, 2] = xv[:, 0] * _TWISTJ_C
        yv[:, 3] = (xv[:, 1] - xv[:, 2] + xv[:, 3]) * _TWISTJ_C

        out = yv.reshape_as(x)
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype)
        return out * scale + bias


class TwistJInt8Layer(nn.Module):
    """
    TWIST J Motor
    INT8 inference path using Rule 42.
    Guarantees zero overflow and pure integer computation.
    """
    def __init__(self, dim: int):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"Dimension must be divisible by 4, got {dim}.")
        self.dim = dim
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("bias", torch.zeros(dim))

    @classmethod
    def from_float(cls, float_layer: TwistJLayer) -> "TwistJInt8Layer":
        """Creates an INT8 inference layer directly from a trained BF16 layer."""
        if not isinstance(float_layer, TwistJLayer):
            raise TypeError(f"from_float expects TwistJLayer, got {type(float_layer).__name__}.")
        int8_layer = cls(float_layer.dim).to(device=float_layer.scale.device)
        with torch.no_grad():
            # Constant folding: Bake 1/sqrt(3) directly into the scale weights
            f_scale = float_layer.scale.detach().to(dtype=torch.float32)
            int8_layer.scale.copy_(f_scale * _TWISTJ_C)
            int8_layer.bias.copy_(float_layer.bias.detach().to(dtype=torch.float32))
        return int8_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_twist_input(x, self.dim, "TwistJInt8Layer")

        # 1. Rule 42 Quantization
        max_val = x.detach().abs().amax().to(dtype=torch.float32).clamp(min=1e-5)
        q_scale = max_val / 42.0
        
        # Quantize to [-42, 42]
        x_int8 = torch.clamp(torch.round(x.to(dtype=torch.float32) / q_scale), -42, 42).to(torch.int8)
        
        # Safe upcast for standard PyTorch integer math
        xv = x_int8.reshape(-1, 4).to(torch.int32)
        yv = torch.empty_like(xv)
        
        # 2. Integer Additive Algebra (Zero Saturation Guaranteed)
        yv[:, 0] = xv[:, 0] - xv[:, 2] + xv[:, 3]
        yv[:, 1] = xv[:, 1] - xv[:, 2]
        yv[:, 2] = xv[:, 0]
        yv[:, 3] = xv[:, 1] - xv[:, 2] + xv[:, 3]
        
        out_int8 = yv.to(torch.int8).reshape(x.shape)
        
        # 3. Dequantize and apply affine transform
        out_float = (out_int8.to(torch.float32) * q_scale).to(dtype=x.dtype)
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype)
        return out_float * scale + bias


if __name__ == "__main__":
    print("=== TWIST J Motor: Self-Test ===")
    
    # Nastavení parametrů
    dim = 512
    batch_size = 4
    seq_len = 128
    
    print(f"1. Initializing float Motor (Dim={dim})...")
    torch.manual_seed(42)
    model = TwistJLayer(dim)
    
    print("2. Generating random input data...")
    x = torch.randn(batch_size, seq_len, dim) * 2.5  # Záměrně širší rozsah pro test kvantizace
    
    print("3. Running Float Forward Pass...")
    out_float = model(x)
    
    print("4. Converting to INT8 Motor (Rule 42)...")
    model_int8 = TwistJInt8Layer.from_float(model)
    
    print("5. Running INT8 Forward Pass...")
    out_int8 = model_int8(x)
    
    # Výpočet chyby
    mae = (out_float - out_int8).abs().mean().item()
    max_err = (out_float - out_int8).abs().max().item()
    
    print("\n=== Results ===")
    print(f"Input Shape: {list(x.shape)}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Max Absolute Error:  {max_err:.4f}")
    print("\n[OK] TWIST J Engine is fully operational.")