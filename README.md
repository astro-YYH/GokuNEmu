# GokuNEmu: A Neural Network Emulator Based on the Goku Simulation Suite

**GokuNEmu** is a neural network (NN) emulator for the nonlinear matter power spectrum, trained on simulations from the **Goku** suite using the [T2N-MusE](https://github.com/astro-YYH/T2N-MusE) emulation technique.

---

## Installation

We recommend installing GokuNEmu via `pip`:

```bash
pip install gokunemu
```

> **Note for Intel Mac users:**  
> You may need to install `pytorch` via `conda` before installing GokuNEmu due to potential compatibility issues with `pip` wheels:
> ```bash
> conda install -c conda-forge pytorch
> ```

---

## Usage

Example notebooks are provided in the `examples/` directory:

- `example.ipynb`: Demonstrates how to use GokuNEmu for predicting the nonlinear matter power spectrum.
- `speed_benchmark.ipynb`: Benchmarks the runtime performance.

---

## Citation

If you use **GokuNEmu**, please cite:

- The main GokuNEmu paper: https://arxiv.org/abs/2507.07177  
- The Goku simulations: https://arxiv.org/abs/2501.06296  
- The T2N-MusE emulation method: https://arxiv.org/abs/2507.07184

---

## License

This project is licensed under the **MIT License**.