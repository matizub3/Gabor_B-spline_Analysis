# Optimizing B-Spline-Based Gabor Systems for Signal Analysis

A research-driven Python implementation for learning compact, high-fidelity function approximations using **B-spline-based Gabor systems**.

This repository investigates how well a small, learned collection of **Gabor atoms** can approximate signals with both oscillatory and localized structure, while explicitly studying the trade-off between **approximation accuracy** and **computational complexity**. The project is grounded in ideas from **frame theory**, **time-frequency analysis**, and **approximation theory**.

---

## Overview
Gabor systems are among the most powerful tools in signal analysis because they provide simultaneous localization in time and frequency. In this project, that framework is paired with a **B-spline window**, producing a flexible and computationally efficient family of basis functions that can be optimized directly for a target signal.

Rather than relying on a fixed dictionary alone, this code learns a sparse approximation of a target function by optimizing:
* **Amplitude coefficients**
* **Translation parameters**
* **Frequency modulation parameters**

The result is a compact, interpretable representation of structured signals using a small number of learned Gabor B-spline atoms.

---

## Mathematical Formulation
The repository studies approximations of a target function $f(x)$ using linear combinations of parameterized Gabor functions of the form:

$$
\phi(x; g, a, b) = g(x - a) \cdot \exp(2\pi i b x)
$$

**Where:**
* $g$ is the B-spline window
* $a$ is a translation parameter
* $b$ is a modulation parameter

The approximation is obtained by solving:

$$
\min_{\alpha_j, a_j, b_j} \left\| f(x) - \sum_{j=1}^{k} \alpha_j \phi(x; g, a_j, b_j) \right\|^2
$$

This optimization problem is the core of the project: learning a compact set of B-spline-based Gabor atoms that best reconstruct a target signal while keeping the representation interpretable and efficient.

---

This combination is powerful because it merges two important strengths:

* **B-splines:** Provide compact support, computational simplicity, and local control. They are particularly well-suited for localized basis construction compared to classical Gaussian windows in certain computational settings.
* **Gabor systems:** Provide excellent time-frequency localization and expressive oscillatory structure, making them highly relevant to complex signal analysis.

Together, they create a basis family that is both mathematically meaningful and practically useful for adaptive approximation tasks, with broader relevance to stable frame constructions and efficient function representation.

---

## Code Functionality

This repository provides a compact experimental framework for:
* Defining a **first-order B-spline window**
* Generating **complex-valued Gabor atoms**
* Learning a sum of Gabor atoms through numerical optimization
* Running **multiple optimization restarts** to improve robustness
* Selecting the best approximation found
* Visualizing both the learned approximation and the learned basis functions

The implementation focuses on adaptive basis learning, where the coefficients, translations, and modulations are all optimized jointly for a fixed number of atoms $k$.

---

## Target Function

The main target function explored in the project combines:
* A sine component
* A cosine component
* A sharply localized Gaussian component

In our poster, this target is written as:

$$
f_1(x) = x(x - 1)\sin(13x) + (1 - x)\cos(23x) + e^{-1000(x-.5)^2}
$$

This design makes the approximation task especially interesting because the model must capture both broad oscillatory behavior and a highly localized spike at the same time. The project also explores related target variants with different Gaussian scales or with only oscillatory or Gaussian structure.

---

## Results

The poster presents:
* A visualization of the learned **translation parameters** $a_j$ and **modulation parameters** $b_j$ as points in time-frequency space for different values of $k$
* Loss comparisons across several target functions with different structural properties
* Approximation plots for increasing values of $k$, specifically **$k = 4, 6, 7, 8, 10, 15$**, showing progressively stronger reconstructions as more atoms are used

The experiments also compare performance across four target-function settings:
* $f_1(x)$: oscillatory + sharply localized Gaussian
* $f_2(x)$: oscillatory + broader Gaussian
* $f_3(x)$: oscillatory only
* $f_4(x)$: Gaussian only

These results highlight that the learned Gabor B-spline representation is flexible enough to model both localized and oscillatory behavior, while also revealing how parameter distributions adapt to the structure of the signal being approximated.

---

## Discussion

A major takeaway from the project is that choosing the number of basis functions $k$ requires balancing two competing goals:
* **Higher accuracy**, since larger $k$ gives the model more expressive power
* **Lower computational cost**, since larger $k$ introduces more parameters and more challenging optimization

Our poster also points toward future work involving **dual frame comparisons**, with the goal of better understanding redundancy, stability, and signal recovery in learned Gabor systems.

---

## Installation & Usage

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install numpy scipy matplotlib
python main.py

```

## Future Work

There are several promising directions for extending this project:

- **Dual frame comparisons:** Compare learned B-spline Gabor approximations with dual-frame-based reconstruction methods to better understand stability, redundancy, and recovery performance.
- **Fixed-lattice formulations:** Explore structured lattice-based Gabor systems with fixed translation and modulation spacing, which may connect more directly to classical frame constructions.
- **Higher-order B-splines:** Investigate whether smoother or higher-degree B-spline windows improve approximation quality or produce more stable learned representations.
- **Complexity benchmarking:** Systematically measure runtime, optimization difficulty, and approximation error as the number of learned atoms $k$ increases.
- **Alternative target functions:** Extend the experiments to noisier, more irregular, or application-specific signals to test generalization beyond the current synthetic targets.
- **Constrained and regularized optimization:** Introduce constraints or sparsity-promoting penalties to improve interpretability and robustness of the learned parameter sets.

These directions would further connect the computational experiments in this repository to broader questions in time-frequency analysis and approximation theory.

---

## Acknowledgments

This work is based on *Optimizing B-Spline Based Gabor Systems for Signal Analysis* by **Mateusz Zubrzycki** and **Kelvin Guobadia** at **Tufts University**.

This work was partially supported by the **National Science Foundation** under grant number **DMS 2309652**.

---

## References

- D. Han, K. Kornelson, D. Larson, and E. Weber, *Frames for Undergraduates*. American Mathematical Society, 2007, pp. 88–104.
- K. Gröchenig, *Foundations of Time-Frequency Analysis*. Springer Science+Business Media, 2001, pp. 83–93.
