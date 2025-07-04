# How-to-Deal-with-the-Differential-Equation-kind-of-the-Hubble-Function

In this Repository, We will going to Explore how we can deal with the Hubble Function which are in the from of the set of the Differential Eqautions

as an example, i will going to conisder the example the example which has been considered one of my Review Paper arXiv:2411.03060v1 "Semi-Symmetric Metric Gravity: A Brief Overview", I will going to conisder the  Linear cosmological model, in the paper, Prof.Dr.Tiberiu Harko has found the form of the Hubble function, where the Hubble paramter has been condiered in the form of the Normalaized form of the Huuble paramer one can the form given below



$\large{\frac{dh(z)}{dz} = \frac{3h^2(z) - 3\sigma_0 (2h(z)\Omega(z) - \Omega^2(z))}{2(1 + z)h(z)}}$

R\frac{d\Omega(z)}{dz} = \frac{-2(3\sigma_0 - 2)h(z)\Omega(z) - (1 - 3\sigma_0)\Omega^2(z)}{2(1 + z)h(z)}R

The system of equations has to be solved with initial conditions $h(0) = 1$, $\Omega(0)=\Omega_0$.

The goal is to conder the two example fisrt, i will take the emcee https://github.com/dfm/emcee and the other example is with the dynesty https://dynesty.readthedocs.io/en/v2.1.5/ , keeo in mind the Huuble function above in the form of the normalexd so one need to sacle i by $H_0$
