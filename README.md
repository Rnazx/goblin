# goblin: Galactic magnetic fields from OBservervationaL INputs
A generalised framework to estimate galactic magnetic fields using observational inputs.

### Dependencies
* Sympy
* NumPy
* Scipy
* Matplotlib

If these packages are not available in your Python interpreter, you can run
```
pip install -r requirements.txt
```
A description of the code framework is available in [this](framework.md) file.
## Instructions to run the code
There are different scripts involved in finding the magnetic fields and pitch angles. The outputs from these scripts are saved as a [pickle](https://docs.python.org/3/library/pickle.html) file. In order to run the relevant files, one can run the script.
```
./main.sh
```
This runs all the necessary scripts, from algebraically solving the expressions to finding the solutions numerically. The order in which to run the scripts is as follows:
### 1. Solving the expressions

* We solve for the magnetic fields and turbulence using [Sympy](https://www.sympy.org/en/index.html).
* The code for solving the expressions can be found in the [expressions](expressions) directory.
* The [turbulence_expressions.py](expressions/turbulence_expressions.py) script solves for the turbulence parameters in terms of the observables. Subsequently, the [magnetic_expressions.py](expressions/magnetic_expressions.py) script uses this solution to find the expressions for the magnetic fields and pitch angles.

### 2. Cleaning and formatting the data
* For each galaxy, the data for the different observables are compiled from different sources and used in the Python files in the [data](data) directory.
*  As this data is compiled from different sources, the radial range involved for each observable can be different. Hence, an interpolation method is implemented where the coarsest radial range is chosen, and the other observables are interpolated for this radial range. A depiction of this interpolation method is shown:
<p align="center">
<img src = "https://github.com/Rnazx/MSc.-Thesis/assets/42196798/edec171d-9f47-4877-b9ec-7e1c19892d9c" width ="500" height = "350" alt = "interpolation" />

<em align="center">The black dots represent the raw data. The green lines represent the set of points for which we want the data to be interpolated.</em>
</p>

**Note:** There are various switches in the scripts to choose the source from which the data is taken from.

* This interpolated data is then used in [zipped_data.py](zipped_data.py), where the values for the parameters and switches are read from the [parameters.in](parameters.in) and [switches.in](switches.in) input files respectively.
* The parameters and the interpolated data are then formatted into the desired format and saved as a pickle file.

### 3. Using the data in the solved expressions
<p align="center">
<img src = "https://github.com/Rnazx/goblin/assets/42196798/800f3048-82c3-46b5-8927-526da900423d" width ="825" height = "400" alt = "interpolation" />

<em align="center"> We solve for the turbulence and magnetic fields using the desired model. Five additional parameters are introduced. For the equations mentioned in this flowchart, kindly refer to the manuscript.</em>
</p>

* This pickle file is further used in the [get_magnetic_observables.py](get_magnetic_observables.py) script.
* In this script, we numerically solve for the magnetic observables and turbulence parameters using a bunch of functions from the [helper_functions.py](helper_functions.py) script.
* **Solving for h numerically:** The [_exp_analytical_data()_](helper_functions.py#L82) is used to substitute in the values of the observables and the physical constants in the solved expression for h. We then use Scipy's fsolve routine to find the root of the polynomial equation in h. 
* **Substituting for this solution in the other expressions:** This numerical solution is then used to obtain the turbulence and the magnetic field quantities using the [_datamaker()_](helper_functions.py#L94) function.
These solutions obtained are again stored in a pickle file, which is used in [plots.ipynb](plots.ipynb) to plot the desired outputs.

## Modifying the model
* To modify the model, you can change the expressions for the turbulence and the magnetic fields in the [expressions](/expressions) directory.
* If the model is changed, uncomment the corresponding lines in [main.sh](main.sh).



