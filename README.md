# PiRD: Physics-informed Residual Diffusion for Flow Field Reconstruction

**Authors**: Siming Shan, Pengkai Wang, Song Chen, Jiaxu Liu  
**Advised by**: Professor Shengze Cai

---

## Abstract

The paper presents a novel approach called Physics-informed Residual Diffusion (PiRD) for reconstructing high-fidelity (HF) flow fields from low-fidelity (LF) observations. The key highlights are:

- **Markov Chain Transition**: PiRD builds a Markov chain between LF and HF flow fields, facilitating the transition from any LF distribution to the HF distribution, unlike CNN-based methods that are limited to specific LF data patterns.
- **Physics-Informed Constraints**: PiRD integrates physics-informed constraints into the objective function, ensuring the reconstructed flow fields comply with the underlying physical laws (e.g., vorticity transport equation) at every step of the Markov chain.

---

## Visualizations

### Low-Fidelity Flow Fields
<div align="center">

| ![low_4x](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/0753435b-1c4c-4c01-b7f0-45ab1edd9a18) | ![low_8x](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/fc82c7cb-2415-45d8-9168-b0f8b88a86d4) | ![low_5p](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/0ad80509-c905-43a4-89f6-c8b3c7ea7362) | ![low_1p](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/1a00ee2d-4d14-43c9-b983-09a88bf764f1) |
|---|---|---|---|
| LF Flow Field 4x | LF Flow Field 8x | LF Flow Field 5% | LF Flow Field 1% |

</div>

### High-Fidelity Flow Fields
<div align="center">

| ![diffusion_4x](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/795d55c1-0979-4bb3-a636-46dc901ae0a3) | ![diffusion_8x](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/1f8c44cd-65ad-4a27-a7f3-122bae0e6678) | ![diffusion_5p](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/1eeec1a3-bfbd-4889-8fdc-33d601a2e0fe) | ![diffusion_1p](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/baa1690b-4b46-4962-871b-fdb8d081bdc9) |
|---|---|---|---|
| HF Flow Field 4x | HF Flow Field 8x | HF Flow Field 5% | HF Flow Field 1% |

</div>

### Flow Fields with Gaussian Noise Densities

<div align="center">
<img src="https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/4e34c73f-44bf-4371-a868-de6ac43e3670" alt="trend_noise_density_smooth">
</div>

The following table shows the flow fields with different densities of Gaussian noise added (0.2, 0.6, 1), and the corresponding reconstructed flow fields using PiRD.

<div align="center">

| Gaussian Noise 0.2 | Gaussian Noise 0.6 | Gaussian Noise 1.0 |
|---|---|---|
| ![gt0 2](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/28eff09f-26c5-44b5-b9e5-4e0b2d5b9163) | ![gt0 6](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/3a4289cc-2c89-4972-8c2c-3ccc9fda1670) | ![gt1](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/ef2b8042-f4e8-4e69-8782-fc36a629daaa) |
| ![diffusin_0 2](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/d3c36b6c-c1c6-4efc-997d-706436ec75b5) | ![diffusin_0 6](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/c140a2a1-38a8-4a55-a7bd-f003f0d1749b) | ![diffusin_1](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction/assets/81949413/2bfdee78-490e-41ce-986e-a770e8c72402) |

</div>


---

For more details, please visit the [GitHub Repository](https://github.com/SimingShan/PiRD-Physics-informed-Residual-Diffusion-for-Flow-Field-Reconstruction).
