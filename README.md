# PiRD: Physics-informed Residual Diffusion for Flow Field Reconstruction

**Authors**: Siming Shan, Pengkai Wang, Song Chen, Jiaxu Liu  
**Proudly Advised by**: Professor Shengze Cai

---

## Abstract

The paper presents a novel approach called Physics-informed Residual Diffusion (PiRD) for reconstructing high-fidelity (HF) flow fields from low-fidelity (LF) observations. The key highlights are:

- **Markov Chain Transition**: PiRD builds a Markov chain between LF and HF flow fields, facilitating the transition from any LF distribution to the HF distribution, unlike CNN-based methods that are limited to specific LF data patterns.
- **Physics-Informed Constraints**: PiRD integrates physics-informed constraints into the objective function, ensuring the reconstructed flow fields comply with the underlying physical laws (e.g., vorticity transport equation) at every step of the Markov chain.

---

## Visualizations

### Low-Fidelity Flow Fields
| ![low_4x](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/0753435b-1c4c-4c01-b7f0-45ab1edd9a18) | ![low_8x](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/fc82c7cb-2415-45d8-9168-b0f8b88a86d4) | ![low_5p](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/0ad80509-c905-43a4-89f6-c8b3c7ea7362) | ![low_1p](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/1a00ee2d-4d14-43c9-b983-09a88bf764f1) |
|---|---|---|---|
| LF Flow Field 4x | LF Flow Field 8x | LF Flow Field 5% | LF Flow Field 1% |

### High-Fidelity Flow Fields
| ![diffusion_4x](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/795d55c1-0979-4bb3-a636-46dc901ae0a3) | ![diffusion_8x](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/1f8c44cd-65ad-4a27-a7f3-122bae0e6678) | ![diffusion_5p](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/1eeec1a3-bfbd-4889-8fdc-33d601a2e0fe) | ![diffusion_1p](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/baa1690b-4b46-4962-871b-fdb8d081bdc9) |
|---|---|---|---|
| HF Flow Field 4x | HF Flow Field 8x | HF Flow Field 5% | HF Flow Field 1% |

---

For more details, please visit the [GitHub Repository](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction).
