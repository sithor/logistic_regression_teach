# Logistic Regression Teaching App

An interactive Shiny app that demonstrates **Maximum Likelihood Estimation (MLE)** for logistic regression.

## Features

This app helps you understand how logistic regression works by:

1. **Interactive Data Generation**: Create synthetic data with known parameters
2. **Manual Parameter Exploration**: Adjust β₀ and β₁ to see how they affect the model fit
3. **Maximum Likelihood Fitting**: Automatically find optimal parameters using MLE
4. **3D Likelihood Surface**: Visualize the log-likelihood function across parameter space
5. **Optimization Path**: See how the gradient ascent algorithm converges to the solution
6. **Theory Tab**: Learn the mathematical foundations of logistic regression

## Installation

### Install Required Packages

Run the installation script:

```r
source("install_packages.R")
```

Or manually install the required packages:

```r
install.packages(c("shiny", "ggplot2", "plotly"))
```

## Running the App

### Option 1: RStudio
Open `app.R` in RStudio and click the "Run App" button.

### Option 2: R Console
```r
shiny::runApp("app.R")
```

### Option 3: Command Line
```bash
R -e "shiny::runApp('app.R')"
```

## How to Use

1. **Generate Data**: 
   - Adjust the sample size and true parameter values
   - Click "Generate New Data" to create a new dataset

2. **Explore Manually**:
   - Use the manual parameter sliders to see how different β₀ and β₁ values affect the fit
   - Watch the log-likelihood change as you adjust parameters

3. **Fit the Model**:
   - Click "Fit Model (MLE)" to find the optimal parameters
   - The fitted curve (green dashed line) will appear on the plot

4. **Visualize the Surface**:
   - Go to the "Log-Likelihood Surface" tab to see the 3D landscape
   - The peak represents the MLE solution

5. **See the Optimization Path**:
   - Enable "Show Optimization Steps"
   - Click "Fit Model" again
   - View the optimization trajectory in the "Optimization Path" tab

## Understanding the Concepts

### Logistic Function
The probability of y=1 given x is:
```
P(y=1|x) = 1 / (1 + exp(-(β₀ + β₁x)))
```

### Log-Likelihood
For n observations:
```
ℓ(β₀, β₁) = Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

### Maximum Likelihood Estimation
MLE finds parameters that maximize the log-likelihood:
```
β̂₀, β̂₁ = argmax ℓ(β₀, β₁)
```

## Educational Goals

This app is designed to help students:
- Understand the logistic function and its S-shaped curve
- See how parameters affect model predictions
- Visualize the likelihood function
- Understand optimization algorithms
- Connect theory with practice

## Requirements

- R >= 3.6.0
- shiny
- ggplot2
- plotly

## License

MIT