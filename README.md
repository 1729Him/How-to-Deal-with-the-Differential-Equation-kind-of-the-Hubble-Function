How to Constrain the Parameters of the Hubble Function Given in the Form of Differential Equations

In this repository, we explore how to deal with Hubble functions that are expressed as a system of differential equations. As an example, we will consider the linear cosmological model, which was studied in the review paper: [Semi-Symmetric Metric Gravity: A Brief Overview](https://doi.org/10.48550/arXiv.2411.03060). In this paper, Prof. Dr. Tiberiu Harko and Mr. Lehel Csillag derives the form of the Hubble function in the context of a cosmological model, where the Hubble parameter is represented in its normalized form.

### System of Differential Equations

The Hubble parameter \$h(z)\$ and the density parameter \$\Omega(z)\$ evolve with redshift according to the following differential equations:

$$
\frac{dh(z)}{dz} = \frac{3h^2(z) - 3\sigma_0 (2h(z)\Omega(z) - \Omega^2(z))}{2(1 + z)h(z)}
$$

$$
\frac{d\Omega(z)}{dz} = \frac{-2(3\sigma_0 - 2)h(z)\Omega(z) - (1 - 3\sigma_0)\Omega^2(z)}{2(1 + z)h(z)}
$$

### Initial Conditions

These equations are solved with the initial conditions:

* $h(0) = 1$
* $\Omega(0) = \Omega_0$

### Scaling the Hubble Function

The above system of differential equations uses the normalized form of the Hubble parameter. To match this with physical quantities, we scale the function by $H_0$, the present-day Hubble constant, in order to properly model the evolution of the universe.

### Example 1: Using `emcee`

In this example, we use the [emcee](https://github.com/dfm/emcee) Python package to solve the system of equations and perform parameter estimation through Markov Chain Monte Carlo (MCMC) simulations.

### Example 2: Using `dynesty`

In this second example, we use the [dynesty](https://dynesty.readthedocs.io/en/v2.1.5/) Python package for Nested Sampling to solve the differential equations and explore the parameter space.

By using these two different approaches, we will demonstrate the power of MCMC and Nested Sampling for solving cosmological models and constraining cosmological parameters.

