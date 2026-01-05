library(shiny)
library(ggplot2)
library(plotly)

# Define UI
ui <- fluidPage(
  titlePanel("🧮 Logistic Regression: Maximum Likelihood Estimation"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("Data Generation"),
      sliderInput("n_samples", "Number of Samples:", 
                  min = 20, max = 200, value = 100, step = 10),
      sliderInput("true_beta0", "True Intercept (β₀):",
                  min = -3, max = 3, value = -1, step = 0.5),
      sliderInput("true_beta1", "True Slope (β₁):",
                  min = -3, max = 3, value = 2, step = 0.5),
      actionButton("generate_data", "Generate New Data", 
                   class = "btn-primary"),
      
      hr(),
      
      h4("Manual Parameter Control"),
      sliderInput("manual_beta0", "Intercept (β₀):",
                  min = -5, max = 5, value = 0, step = 0.1),
      sliderInput("manual_beta1", "Slope (β₁):",
                  min = -5, max = 5, value = 1, step = 0.1),
      
      hr(),
      
      actionButton("fit_model", "Fit Model (MLE)", 
                   class = "btn-success"),
      
      hr(),
      
      h4("Optimization Display"),
      checkboxInput("show_iterations", "Show Optimization Steps", FALSE),
      checkboxInput("show_residuals", "Show Residuals", FALSE),
      
      hr(),
      
      helpText("This app demonstrates how logistic regression uses maximum likelihood estimation (MLE) to find optimal parameters.")
    ),
    
    mainPanel(
      width = 9,
      tabsetPanel(
        tabPanel("Data & Fit",
                 fluidRow(
                   column(12,
                          div(style = "background-color: #f0f8ff; padding: 12px; border-radius: 5px; margin-bottom: 15px;",
                              h5(style = "color: #2c3e50; margin-top: 0;", "What is Likelihood?"),
                              p(style = "margin-bottom: 8px;", strong("Likelihood"), " is the joint probability of observing your data, given a particular model with specific parameter values."),
                              withMathJax(),
                              p(style = "margin-bottom: 8px;", "For logistic regression: \\(L(\\beta_0, \\beta_1 | \\text{data}) = \\prod_{i=1}^{n} P(y_i | x_i, \\beta_0, \\beta_1)\\)"),
                              p(style = "margin-bottom: 8px;", "Maximum Likelihood Estimation (MLE) finds the parameter values that make your observed data ", em("most probable"), ". We maximize the log-likelihood (sum of log probabilities) rather than the likelihood itself for numerical stability."),
                              p(style = "margin-bottom: 8px; padding: 8px; background-color: #e3f2fd; border-left: 3px solid #2196F3;", 
                                strong("Key:"), " Higher log-likelihood values = data are MORE likely given the model. The goal is to find parameters that maximize this value."),
                              hr(style = "margin: 10px 0;"),
                              h5(style = "color: #2c3e50;", "Why Deviance Instead of Residuals?"),
                              p(style = "margin-bottom: 8px;", "In ", strong("linear regression"), ", we minimize residuals: \\((y - \\hat{y})^2\\). But in ", strong("logistic regression"), ", \\(y\\) is 0 or 1 while \\(\\hat{y}\\) is a probability (0 to 1), so raw residuals aren't as meaningful."),
                              p(style = "margin-bottom: 8px;", strong("Deviance"), " = \\(-2 \\times \\text{log-likelihood}\\) measures model fit probabilistically. Lower deviance = better fit = higher likelihood of the data."),
                              p(style = "margin-bottom: 0;", em("Enable 'Show Residuals' to see prediction errors. The bar chart below shows how deviance (and individual residual contributions) improve when the model is fitted."))
                          )
                   )
                 ),
                 fluidRow(
                   column(6, plotlyOutput("data_plot", height = "400px")),
                   column(6, plotlyOutput("probability_plot", height = "400px"))
                 ),
                 fluidRow(
                   column(12, 
                          h4("Model Comparison: Deviance & Fit Quality"),
                          plotlyOutput("deviance_comparison", height = "300px")
                   )
                 ),
                 fluidRow(
                   column(12, 
                          h4("Current Parameters & Log-Likelihood"),
                          verbatimTextOutput("current_params")
                   )
                 )
        ),
        
        tabPanel("Log-Likelihood Surface",
                 fluidRow(
                   column(12,
                          div(style = "background-color: #fff9e6; padding: 12px; border-radius: 5px; margin-bottom: 15px;",
                              h5(style = "color: #2c3e50; margin-top: 0;", "The Likelihood Surface"),
                              withMathJax(),
                              p(style = "margin-bottom: 8px;", "The surface below shows how the ", strong("joint probability of the data"), " changes as we vary the model parameters \\(\\beta_0\\) and \\(\\beta_1\\)."),
                              p(style = "margin-bottom: 8px;", "Each point on this surface represents: 'If the true parameters were \\(\\beta_0\\) and \\(\\beta_1\\), how probable would our observed data be?'"),
                              p(style = "margin-bottom: 0;", strong("The peak"), " is where the data is most probable - these are the Maximum Likelihood Estimates (MLE). The optimization algorithms navigate this surface to find this peak.")
                          )
                   )
                 ),
                 plotlyOutput("likelihood_surface", height = "600px"),
                 fluidRow(
                   column(12,
                          h4("Understanding the Surface"),
                          p("This 3D surface shows the log-likelihood for different parameter values. 
                            The peak (maximum) represents the MLE estimates. The red point shows 
                            the current manual parameter values, and the green point shows the 
                            fitted MLE parameters.")
                   )
                 )
        ),
        
        tabPanel("Optimization Path",
                 plotlyOutput("optimization_path", height = "600px"),
                 fluidRow(
                   column(12,
                          h4("Gradient Ascent Visualization"),
                          verbatimTextOutput("optimization_details")
                   )
                 )
        ),
        
        tabPanel("Newton-Raphson",
                 fluidRow(
                   column(12,
                          h4("Newton-Raphson Iterative Fitting"),
                          p("This tab implements Newton-Raphson optimization from scratch and shows the geometric interpretation of each iteration."),
                          div(style = "background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0;",
                              h5(style = "color: #2c3e50;", "How Newton-Raphson Works (Simple Explanation)"),
                              tags$ol(
                                tags$li(strong("Start with a guess:"), " Begin with initial parameter values (β₀, β₁)"),
                                tags$li(strong("Fit a quadratic curve:"), " At your current position, approximate the log-likelihood surface with a parabola (quadratic function)"),
                                tags$li(strong("Jump to the peak:"), " Find where this parabola reaches its maximum - that's your next guess"),
                                tags$li(strong("Repeat:"), " Keep doing this until you stop improving significantly")
                              ),
                              p(style = "margin-top: 10px;", 
                                em("Why it works:"), " The parabola uses information about both the ", 
                                strong("slope (gradient)"), " and ", strong("curvature (Hessian)"), 
                                " of the log-likelihood surface, so it makes very accurate predictions about where the peak is. This is much faster than just following the gradient!"),
                              p(strong("In the plots below:"), " The left shows the true surface, the right shows the parabola approximation. Watch how the green arrow points to where the parabola peaks.")
                          ),
                          div(style = "background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0;",
                              h5(style = "color: #2c3e50;", "How Does Newton-Raphson Know Which Direction?"),
                              p(strong("Short answer:"), " Yes! The inverse Hessian (\\(H^{-1}\\)) determines both the direction AND the step size."),
                              withMathJax(),
                              tags$ul(
                                tags$li(strong("Gradient alone (\\(\\nabla \\ell\\)):"), " Points in the direction of steepest ascent - like climbing straight uphill"),
                                tags$li(strong("Inverse Hessian (\\(H^{-1}\\)):"), " Modifies this direction based on the curvature of the surface:"),
                                tags$ul(
                                  tags$li("In directions with high curvature (steep bowl), it takes smaller steps"),
                                  tags$li("In directions with low curvature (flat), it takes larger steps"),
                                  tags$li("It can even rotate the direction to account for correlation between parameters")
                                ),
                                tags$li(strong("Combined (\\(-H^{-1} \\nabla \\ell\\)):"), " Gives an 'intelligent' direction that jumps directly to where the quadratic approximation peaks")
                              ),
                              p(style = "margin-top: 10px; font-style: italic;", 
                                "Think of it this way: The gradient says 'go uphill', but the inverse Hessian says 'actually, based on the curvature, the peak is over THERE' - and points to a smarter direction!")
                          )
                   )
                 ),
                 fluidRow(
                   column(4,
                          sliderInput("nr_start_beta0", "Starting β₀:",
                                    min = -4, max = 4, value = 0, step = 0.5),
                          sliderInput("nr_start_beta1", "Starting β₁:",
                                    min = -4, max = 4, value = 0.5, step = 0.5),
                          sliderInput("nr_max_iter", "Max Iterations:",
                                    min = 1, max = 20, value = 10, step = 1),
                          actionButton("run_nr", "Run Newton-Raphson", 
                                     class = "btn-primary"),
                          hr(),
                          sliderInput("nr_show_iter", "Show Iteration:",
                                    min = 1, max = 10, value = 1, step = 1),
                          hr(),
                          verbatimTextOutput("nr_iteration_info")
                   ),
                   column(8,
                          plotOutput("nr_geometry_plot", height = "500px")
                   )
                 ),
                 fluidRow(
                   column(12,
                          h4("Convergence Path"),
                          plotlyOutput("nr_convergence_plot", height = "400px")
                   )
                 ),
                 fluidRow(
                   column(12,
                          h4("Mathematical Details: From Derivatives to Quadratic Approximation"),
                          div(style = "background-color: #fff9e6; padding: 15px; border-radius: 5px; margin: 10px 0;",
                              p(strong("How derivatives create the parabola:")),
                              p("Using a Taylor series expansion around the current point \\(\\beta^{(t)}\\), we approximate the log-likelihood:"),
                              withMathJax("$$\\ell(\\beta) \\approx \\ell(\\beta^{(t)}) + \\nabla \\ell(\\beta^{(t)})^T (\\beta - \\beta^{(t)}) + \\frac{1}{2}(\\beta - \\beta^{(t)})^T H(\\beta^{(t)}) (\\beta - \\beta^{(t)})$$"),
                              tags$ul(
                                tags$li(strong("Constant term:"), " \\(\\ell(\\beta^{(t)})\\) - the log-likelihood at the current point"),
                                tags$li(strong("Linear term:"), " \\(\\nabla \\ell(\\beta^{(t)})^T (\\beta - \\beta^{(t)})\\) - uses the ", em("gradient"), " (slope) to predict how ℓ changes"),
                                tags$li(strong("Quadratic term:"), " \\(\\frac{1}{2}(\\beta - \\beta^{(t)})^T H(\\beta^{(t)}) (\\beta - \\beta^{(t)})\\) - uses the ", em("Hessian"), " (curvature) to capture the bend")
                              ),
                              p(style = "margin-top: 10px;", "These three terms together form a ", strong("quadratic function"), 
                                " (parabola in 2D, paraboloid in higher dimensions). To find where this parabola peaks, we take its derivative and set to zero, giving us:"),
                              withMathJax("$$\\beta^{(t+1)} = \\beta^{(t)} - H^{-1}(\\beta^{(t)}) \\nabla \\ell(\\beta^{(t)})$$"),
                              p(strong("Key insight:"), " The Hessian tells us the curvature (how curved the surface is), which helps us figure out how big of a step to take in the gradient direction.")
                          ),
                          
                          div(style = "background-color: #ffebee; padding: 15px; border-radius: 5px; margin: 10px 0;",
                              h5(style = "color: #2c3e50; margin-top: 0;", "When Does Newton-Raphson Stop?"),
                              p(strong("The algorithm stops when one of these conditions is met:")),
                              withMathJax(),
                              tags$ol(
                                tags$li(strong("Convergence:"), " The change in parameter values between iterations is very small"),
                                tags$ul(
                                  tags$li("Specifically: \\(||\\beta^{(t+1)} - \\beta^{(t)}|| < \\text{tolerance}\\)"),
                                  tags$li("Default tolerance = 0.000001 (1e-6)"),
                                  tags$li("This means the parameters have stabilized and further iterations won't improve the solution")
                                ),
                                tags$li(strong("Maximum iterations reached:"), " Safety limit to prevent infinite loops"),
                                tags$ul(
                                  tags$li("You can set this in the 'Max Iterations' slider"),
                                  tags$li("If this limit is reached before convergence, the algorithm may not have found the optimal solution")
                                ),
                                tags$li(strong("Numerical problems:"), " The Hessian becomes singular (non-invertible)"),
                                tags$ul(
                                  tags$li("This can happen with extreme parameter values or poorly conditioned data"),
                                  tags$li("The algorithm cannot proceed and stops with an error message")
                                )
                              ),
                              p(style = "margin-top: 10px; font-style: italic;", 
                                "Watch the 'Gradient Magnitude' plot in the convergence section - it should approach zero as the algorithm finds the peak!")
                          ),
                          
                          div(style = "background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0;",
                              h5(style = "color: #2c3e50; margin-top: 0;", "Worked Example: Understanding the N-R Step"),
                              p(strong("How gradient direction and curvature determine the step:")),
                              withMathJax(),
                              p("Suppose at the current point \\(\\beta^{(t)}\\):"),
                              tags$ul(
                                tags$li(strong("Gradient is positive:"), " \\(\\nabla \\ell(\\beta^{(t)}) = +5\\)"),
                                tags$ul(
                                  tags$li("This means the log-likelihood is increasing as \\(\\beta\\) increases"),
                                  tags$li("We should move in the ", em("positive direction"), " to climb uphill")
                                )
                              ),
                              p(strong("But how far should we step? The Hessian determines this:")),
                              
                              tags$div(style = "margin-left: 20px;",
                                tags$p(strong("Scenario 1: High curvature"), " (steep bowl) → ", strong("Small step")),
                                tags$ul(
                                  tags$li("Hessian: \\(H = -10\\) (large negative value = high curvature)"),
                                  tags$li("Newton-Raphson step: \\(-H^{-1} \\nabla \\ell = -(-10)^{-1} \\times 5 = +0.5\\)"),
                                  tags$li(em("Small step of 0.5"), " because the surface curves sharply - the peak is nearby!")
                                ),
                                
                                tags$p(style = "margin-top: 10px;", strong("Scenario 2: Low curvature"), " (flat/gradual) → ", strong("Large step")),
                                tags$ul(
                                  tags$li("Hessian: \\(H = -2\\) (small negative value = low curvature)"),
                                  tags$li("Newton-Raphson step: \\(-H^{-1} \\nabla \\ell = -(-2)^{-1} \\times 5 = +2.5\\)"),
                                  tags$li(em("Large step of 2.5"), " because the surface is flat - the peak is far away!")
                                )
                              ),
                              
                              p(style = "margin-top: 10px; background-color: white; padding: 10px; border-left: 4px solid #4CAF50;",
                                strong("Key insight:"), " The step size is ", em("inversely proportional"), " to the curvature. ",
                                "Same gradient (+5) but high curvature → small step (0.5), low curvature → large step (2.5). ",
                                "This is why Newton-Raphson is smarter than just following the gradient!")
                          ),
                          
                          h4("Algorithm Explanation"),
                          p("Newton-Raphson (also called Fisher Scoring for logistic regression) uses second-order information (Hessian matrix) to find the maximum likelihood estimates."),
                          withMathJax("$$\\beta^{(t+1)} = \\beta^{(t)} - H^{-1}(\\beta^{(t)}) \\nabla \\ell(\\beta^{(t)})$$"),
                          p("where:"),
                          withMathJax(),
                          tags$ul(
                            tags$li("\\(\\nabla \\ell(\\beta)\\) is the gradient (first derivative) of the log-likelihood"),
                            tags$li("\\(H(\\beta)\\) is the Hessian matrix (second derivative)"),
                            tags$li("The algorithm uses a quadratic approximation at each step to find the next iterate")
                          ),
                          p("R's glm() function uses Iteratively Reweighted Least Squares (IRLS), which is equivalent to Newton-Raphson for logistic regression.")
                   )
                 )
        ),
        
        tabPanel("Theory",
                 h3("Logistic Regression and Maximum Likelihood Estimation"),
                 
                 h4("1. The Logistic Function"),
                 p("The probability of y=1 given x is:"),
                 withMathJax("$$P(y=1|x) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 x)}}$$"),
                 
                 h4("2. Log-Likelihood Function"),
                 p("For n observations, the log-likelihood is:"),
                 withMathJax("$$\\ell(\\beta_0, \\beta_1) = \\sum_{i=1}^{n} \\left[ y_i \\log(p_i) + (1-y_i) \\log(1-p_i) \\right]$$"),
                 withMathJax(),
                 p("where \\(p_i = P(y_i=1|x_i)\\)"),
                 
                 h4("3. Maximum Likelihood Estimation"),
                 p("MLE finds the parameters that maximize the log-likelihood:"),
                 withMathJax("$$\\hat{\\beta}_0, \\hat{\\beta}_1 = \\arg\\max_{\\beta_0, \\beta_1} \\ell(\\beta_0, \\beta_1)$$"),
                 
                 h4("4. Interpretation"),
                 withMathJax(),
                 tags$ul(
                   tags$li("\\(\\beta_0\\) (intercept): log-odds when x=0"),
                   tags$li("\\(\\beta_1\\) (slope): change in log-odds for unit increase in x"),
                   tags$li("\\(e^{\\beta_1}\\): odds ratio")
                 ),
                 
                 h4("How to Use This App"),
                 tags$ol(
                   tags$li("Generate data with known parameters using the left sidebar"),
                   tags$li("Manually adjust parameters to see how they affect the fit and log-likelihood"),
                   tags$li("Click 'Fit Model' to find the MLE estimates"),
                   tags$li("Explore the log-likelihood surface to understand the optimization landscape"),
                   tags$li("View the optimization path to see how the algorithm converges")
                 )
        )
      )
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactive values to store data and results
  values <- reactiveValues(
    data = NULL,
    fitted_beta0 = NULL,
    fitted_beta1 = NULL,
    opt_path = NULL,
    fitted = FALSE,
    nr_results = NULL
  )
  
  # Logistic function
  logistic <- function(x, beta0, beta1) {
    1 / (1 + exp(-(beta0 + beta1 * x)))
  }
  
  # Log-likelihood function
  log_likelihood <- function(beta, X, y) {
    beta0 <- beta[1]
    beta1 <- beta[2]
    p <- logistic(X, beta0, beta1)
    # Add small epsilon to avoid log(0)
    p <- pmax(pmin(p, 1 - 1e-15), 1e-15)
    sum(y * log(p) + (1 - y) * log(1 - p))
  }
  
  # Generate data
  observeEvent(input$generate_data, {
    set.seed(Sys.time())
    n <- input$n_samples
    X <- runif(n, -3, 3)
    p_true <- logistic(X, input$true_beta0, input$true_beta1)
    y <- rbinom(n, 1, p_true)
    
    values$data <- data.frame(X = X, y = y, p_true = p_true)
    values$fitted <- FALSE
    values$fitted_beta0 <- NULL
    values$fitted_beta1 <- NULL
    values$opt_path <- NULL
  })
  
  # Initialize data on startup
  observe({
    if (is.null(values$data)) {
      n <- input$n_samples
      X <- runif(n, -3, 3)
      p_true <- logistic(X, input$true_beta0, input$true_beta1)
      y <- rbinom(n, 1, p_true)
      values$data <- data.frame(X = X, y = y, p_true = p_true)
    }
  })
  
  # Fit model using MLE
  observeEvent(input$fit_model, {
    req(values$data)
    
    # Track optimization path
    opt_history <- list()
    iter <- 0
    
    # Custom objective function that tracks iterations
    obj_fun <- function(beta) {
      iter <<- iter + 1
      ll <- log_likelihood(beta, values$data$X, values$data$y)
      if (input$show_iterations) {
        opt_history[[iter]] <<- c(beta[1], beta[2], ll)
      }
      return(-ll)  # Negative because optim minimizes
    }
    
    # Fit using optim
    result <- optim(
      par = c(0, 0),
      fn = obj_fun,
      method = "BFGS",
      control = list(maxit = 100)
    )
    
    values$fitted_beta0 <- result$par[1]
    values$fitted_beta1 <- result$par[2]
    values$fitted <- TRUE
    
    if (input$show_iterations && length(opt_history) > 0) {
      values$opt_path <- do.call(rbind, opt_history)
      colnames(values$opt_path) <- c("beta0", "beta1", "log_lik")
    } else {
      values$opt_path <- NULL
    }
    
    # Update manual sliders to show fitted values
    updateSliderInput(session, "manual_beta0", value = round(result$par[1], 2))
    updateSliderInput(session, "manual_beta1", value = round(result$par[2], 2))
  })
  
  # Plot data with fitted curve
  output$data_plot <- renderPlotly({
    req(values$data)
    
    # Current parameters (manual or fitted)
    beta0 <- input$manual_beta0
    beta1 <- input$manual_beta1
    
    # Create prediction line
    x_seq <- seq(min(values$data$X), max(values$data$X), length.out = 100)
    p_manual <- logistic(x_seq, beta0, beta1)
    
    # Calculate predicted probabilities for each data point
    values$data$p_pred <- logistic(values$data$X, beta0, beta1)
    values$data$residual <- values$data$y - values$data$p_pred
    
    # Base plot
    p <- ggplot(values$data, aes(x = X, y = y)) +
      geom_line(data = data.frame(x = x_seq, p = p_manual),
                aes(x = x, y = p), color = "blue", linewidth = 1.2,
                linetype = "solid") +
      geom_point(aes(color = factor(y)), size = 3, alpha = 0.6) +
      scale_color_manual(values = c("0" = "#F8766D", "1" = "#00BA38"),
                         name = "Class") +
      labs(title = "Logistic Regression Fit",
           x = "X (Predictor)",
           y = "Y (Binary Outcome)",
           subtitle = paste0("Current: β₀=", round(beta0, 2), 
                           ", β₁=", round(beta1, 2))) +
      theme_minimal() +
      theme(legend.position = "right")
    
    # Add residuals if requested
    if (input$show_residuals) {
      p <- p + geom_segment(data = values$data,
                           aes(x = X, xend = X, 
                               y = y, yend = p_pred,
                               color = factor(y)),
                           alpha = 0.3, linewidth = 0.8,
                           linetype = "solid")
    }
    
    # Add fitted line if available
    if (values$fitted) {
      p_fitted <- logistic(x_seq, values$fitted_beta0, values$fitted_beta1)
      p <- p + geom_line(data = data.frame(x = x_seq, p = p_fitted),
                        aes(x = x, y = p), color = "green", linewidth = 1.2,
                        linetype = "dashed")
    }
    
    ggplotly(p) %>% layout(hovermode = "closest")
  })
  
  # Plot probabilities
  output$probability_plot <- renderPlotly({
    req(values$data)
    
    beta0 <- input$manual_beta0
    beta1 <- input$manual_beta1
    
    # Calculate predicted probabilities
    p_pred <- logistic(values$data$X, beta0, beta1)
    plot_data <- values$data
    plot_data$p_pred <- p_pred
    plot_data$residual <- plot_data$y - p_pred
    
    p <- ggplot(plot_data, aes(x = X)) +
      geom_line(aes(y = p_pred, color = "Predicted P(Y=1)"), linewidth = 1.2) +
      geom_point(aes(y = y, color = "Observed"), alpha = 0.5, size = 2) +
      scale_color_manual(values = c("Observed" = "black", 
                                    "Predicted P(Y=1)" = "blue"),
                        name = "") +
      labs(title = "Predicted Probabilities",
           x = "X (Predictor)",
           y = "Probability / Outcome") +
      theme_minimal() +
      theme(legend.position = "right")
    
    # Add residuals if requested
    if (input$show_residuals) {
      p <- p + geom_segment(aes(x = X, xend = X, 
                               y = y, yend = p_pred),
                           color = "red", alpha = 0.4, linewidth = 0.6)
    }
    
    ggplotly(p) %>% layout(hovermode = "closest")
  })
  
  # Display current parameters and log-likelihood
  output$current_params <- renderText({
    req(values$data)
    
    beta0 <- input$manual_beta0
    beta1 <- input$manual_beta1
    ll <- log_likelihood(c(beta0, beta1), values$data$X, values$data$y)
    
    result <- paste0(
      "Manual Parameters:\n",
      "  β₀ (Intercept) = ", round(beta0, 4), "\n",
      "  β₁ (Slope) = ", round(beta1, 4), "\n",
      "  Log-Likelihood = ", round(ll, 4), "\n",
      "  Deviance = ", round(-2 * ll, 4), "\n\n"
    )
    
    if (values$fitted) {
      ll_fitted <- log_likelihood(c(values$fitted_beta0, values$fitted_beta1), 
                                  values$data$X, values$data$y)
      deviance_reduction <- -2 * ll - (-2 * ll_fitted)
      result <- paste0(
        result,
        "MLE (Fitted) Parameters:\n",
        "  β₀ (Intercept) = ", round(values$fitted_beta0, 4), "\n",
        "  β₁ (Slope) = ", round(values$fitted_beta1, 4), "\n",
        "  Log-Likelihood = ", round(ll_fitted, 4), "\n",
        "  Deviance = ", round(-2 * ll_fitted, 4), "\n",
        "  Odds Ratio (exp(β₁)) = ", round(exp(values$fitted_beta1), 4), "\n\n",
        "Deviance Reduction: ", round(deviance_reduction, 4), 
        " (", round(100 * deviance_reduction / (-2 * ll), 2), "% improvement)\n\n"
      )
      
      result <- paste0(
        result,
        "True Parameters (used for data generation):\n",
        "  β₀ = ", input$true_beta0, "\n",
        "  β₁ = ", input$true_beta1
      )
    }
    
    result
  })
  
  # Deviance comparison plot
  output$deviance_comparison <- renderPlotly({
    req(values$data)
    
    beta0 <- input$manual_beta0
    beta1 <- input$manual_beta1
    
    # Calculate log-likelihood for manual model
    ll_manual <- log_likelihood(c(beta0, beta1), values$data$X, values$data$y)
    deviance_manual <- -2 * ll_manual
    
    # Calculate individual deviance contributions for manual
    p_manual <- logistic(values$data$X, beta0, beta1)
    p_manual <- pmax(pmin(p_manual, 1 - 1e-15), 1e-15)
    dev_contrib_manual <- -2 * (values$data$y * log(p_manual) + 
                                  (1 - values$data$y) * log(1 - p_manual))
    
    if (values$fitted) {
      # Calculate for fitted model
      ll_fitted <- log_likelihood(c(values$fitted_beta0, values$fitted_beta1), 
                                  values$data$X, values$data$y)
      deviance_fitted <- -2 * ll_fitted
      
      # Individual contributions for fitted
      p_fitted <- logistic(values$data$X, values$fitted_beta0, values$fitted_beta1)
      p_fitted <- pmax(pmin(p_fitted, 1 - 1e-15), 1e-15)
      dev_contrib_fitted <- -2 * (values$data$y * log(p_fitted) + 
                                    (1 - values$data$y) * log(1 - p_fitted))
      
      # Create comparison data
      comparison_data <- data.frame(
        Model = c("Manual Parameters", "MLE (Fitted)"),
        Deviance = c(deviance_manual, deviance_fitted),
        LogLikelihood = c(ll_manual, ll_fitted),
        Color = c("#FF6B6B", "#4ECDC4")
      )
      
      p <- plot_ly() %>%
        add_bars(
          data = comparison_data,
          x = ~Model,
          y = ~Deviance,
          marker = list(color = ~Color),
          text = ~paste0("Deviance: ", round(Deviance, 2), 
                        "<br>Log-Likelihood: ", round(LogLikelihood, 2)),
          hoverinfo = "text",
          name = "Deviance"
        ) %>%
        layout(
          title = "Model Fit Comparison (Lower Deviance = Better Fit)",
          yaxis = list(title = "Deviance (-2 × Log-Likelihood)"),
          xaxis = list(title = ""),
          showlegend = FALSE
        ) %>%
        add_annotations(
          x = 0.5,
          y = max(comparison_data$Deviance) * 0.95,
          text = paste0("Improvement: ", 
                       round(deviance_manual - deviance_fitted, 2),
                       " (", round(100 * (deviance_manual - deviance_fitted) / deviance_manual, 1), "%)"),
          showarrow = FALSE,
          font = list(size = 12, color = "#2E7D32")
        )
      
    } else {
      # Only manual model available
      comparison_data <- data.frame(
        Model = "Manual Parameters",
        Deviance = deviance_manual,
        LogLikelihood = ll_manual,
        Color = "#FF6B6B"
      )
      
      p <- plot_ly() %>%
        add_bars(
          data = comparison_data,
          x = ~Model,
          y = ~Deviance,
          marker = list(color = ~Color),
          text = ~paste0("Deviance: ", round(Deviance, 2), 
                        "<br>Log-Likelihood: ", round(LogLikelihood, 2)),
          hoverinfo = "text",
          name = "Deviance"
        ) %>%
        layout(
          title = "Current Model Fit (Click 'Fit Model' to see MLE comparison)",
          yaxis = list(title = "Deviance (-2 × Log-Likelihood)"),
          xaxis = list(title = ""),
          showlegend = FALSE
        )
    }
    
    p
  })
  
  # Plot log-likelihood surface
  output$likelihood_surface <- renderPlotly({
    req(values$data)
    
    # Create grid for likelihood surface
    beta0_seq <- seq(-4, 4, length.out = 50)
    beta1_seq <- seq(-4, 4, length.out = 50)
    grid <- expand.grid(beta0 = beta0_seq, beta1 = beta1_seq)
    
    grid$log_lik <- apply(grid, 1, function(row) {
      log_likelihood(c(row[1], row[2]), values$data$X, values$data$y)
    })
    
    # Reshape for surface plot
    ll_matrix <- matrix(grid$log_lik, nrow = length(beta0_seq), 
                       ncol = length(beta1_seq))
    
    # Create surface plot
    p <- plot_ly(x = beta1_seq, y = beta0_seq, z = ll_matrix) %>%
      add_surface(colorscale = "Viridis", name = "Log-Likelihood") %>%
      layout(
        title = "Log-Likelihood Surface",
        scene = list(
          xaxis = list(title = "β₁ (Slope)"),
          yaxis = list(title = "β₀ (Intercept)"),
          zaxis = list(title = "Log-Likelihood"),
          camera = list(eye = list(x = 1.5, y = 1.5, z = 1.3))
        )
      )
    
    # Add manual point
    ll_manual <- log_likelihood(c(input$manual_beta0, input$manual_beta1), 
                               values$data$X, values$data$y)
    p <- p %>% add_trace(
      x = input$manual_beta1,
      y = input$manual_beta0,
      z = ll_manual,
      type = "scatter3d",
      mode = "markers",
      marker = list(size = 8, color = "red"),
      name = "Manual Parameters"
    )
    
    # Add fitted point if available
    if (values$fitted) {
      ll_fitted <- log_likelihood(c(values$fitted_beta0, values$fitted_beta1), 
                                  values$data$X, values$data$y)
      p <- p %>% add_trace(
        x = values$fitted_beta1,
        y = values$fitted_beta0,
        z = ll_fitted,
        type = "scatter3d",
        mode = "markers",
        marker = list(size = 10, color = "green"),
        name = "MLE (Fitted)"
      )
    }
    
    p
  })
  
  # Plot optimization path
  output$optimization_path <- renderPlotly({
    if (is.null(values$opt_path)) {
      plot_ly() %>%
        layout(
          title = "Optimization Path (Enable 'Show Optimization Steps' and click 'Fit Model')",
          xaxis = list(title = "β₁ (Slope)"),
          yaxis = list(title = "β₀ (Intercept)")
        )
    } else {
      opt_df <- as.data.frame(values$opt_path)
      opt_df$iteration <- 1:nrow(opt_df)
      
      # Create contour plot of log-likelihood
      beta0_seq <- seq(min(opt_df$beta0) - 1, max(opt_df$beta0) + 1, length.out = 40)
      beta1_seq <- seq(min(opt_df$beta1) - 1, max(opt_df$beta1) + 1, length.out = 40)
      grid <- expand.grid(beta0 = beta0_seq, beta1 = beta1_seq)
      
      grid$log_lik <- apply(grid, 1, function(row) {
        log_likelihood(c(row[1], row[2]), values$data$X, values$data$y)
      })
      
      p <- plot_ly() %>%
        add_contour(
          x = beta1_seq,
          y = beta0_seq,
          z = matrix(grid$log_lik, nrow = length(beta0_seq)),
          colorscale = "Viridis",
          contours = list(showlabels = TRUE),
          name = "Log-Likelihood"
        ) %>%
        add_trace(
          data = opt_df,
          x = ~beta1,
          y = ~beta0,
          type = "scatter",
          mode = "lines+markers",
          marker = list(size = 8, color = ~iteration, 
                       colorscale = "Reds", showscale = TRUE),
          line = list(color = "red", width = 2),
          name = "Optimization Path",
          text = ~paste("Iteration:", iteration, 
                       "<br>β₀:", round(beta0, 3),
                       "<br>β₁:", round(beta1, 3),
                       "<br>Log-Lik:", round(log_lik, 2))
        ) %>%
        layout(
          title = "Gradient Ascent Optimization Path",
          xaxis = list(title = "β₁ (Slope)"),
          yaxis = list(title = "β₀ (Intercept)")
        )
      
      p
    }
  })
  
  # Display optimization details
  output$optimization_details <- renderText({
    if (is.null(values$opt_path)) {
      "Enable 'Show Optimization Steps' and click 'Fit Model' to see the optimization trajectory."
    } else {
      opt_df <- as.data.frame(values$opt_path)
      paste0(
        "Optimization converged in ", nrow(opt_df), " iterations\n\n",
        "Starting point:\n",
        "  β₀ = ", round(opt_df$beta0[1], 4), "\n",
        "  β₁ = ", round(opt_df$beta1[1], 4), "\n",
        "  Log-Likelihood = ", round(opt_df$log_lik[1], 4), "\n\n",
        "Final point:\n",
        "  β₀ = ", round(opt_df$beta0[nrow(opt_df)], 4), "\n",
        "  β₁ = ", round(opt_df$beta1[nrow(opt_df)], 4), "\n",
        "  Log-Likelihood = ", round(opt_df$log_lik[nrow(opt_df)], 4), "\n\n",
        "Improvement: ", round(opt_df$log_lik[nrow(opt_df)] - opt_df$log_lik[1], 4)
      )
    }
  })
  
  # Newton-Raphson Implementation
  newton_raphson_logistic <- function(X, y, beta_init, max_iter = 20, tol = 1e-6) {
    beta <- beta_init
    n <- length(y)
    history <- list()
    
    for (iter in 1:max_iter) {
      # Calculate probabilities with numerical stability
      eta <- beta[1] + beta[2] * X
      eta <- pmax(pmin(eta, 20), -20)  # Clip to avoid overflow
      p <- 1 / (1 + exp(-eta))
      
      # Gradient (score vector)
      grad <- c(
        sum(y - p),
        sum((y - p) * X)
      )
      
      # Hessian matrix (Fisher information matrix)
      W <- p * (1 - p)
      W <- pmax(W, 1e-10)  # Avoid zero weights
      
      H <- matrix(c(
        -sum(W),
        -sum(W * X),
        -sum(W * X),
        -sum(W * X^2)
      ), nrow = 2, byrow = TRUE)
      
      # Store iteration info
      ll <- log_likelihood(beta, X, y)
      history[[iter]] <- list(
        beta = beta,
        grad = grad,
        H = H,
        log_lik = ll,
        iter = iter
      )
      
      # Newton-Raphson update with error handling
      # Check if Hessian is invertible
      H_det <- det(H)
      if (abs(H_det) < 1e-10) {
        # Hessian is singular or nearly singular
        # Return what we have so far with an error flag
        history[[iter]]$error <- "Hessian is singular - try different starting values"
        break
      }
      
      tryCatch({
        H_inv <- solve(H)
        beta_new <- beta - H_inv %*% grad
        
        # Check for numerical issues
        if (any(is.na(beta_new)) || any(is.infinite(beta_new))) {
          history[[iter]]$error <- "Numerical instability - try different starting values"
          break
        }
        
        # Check convergence
        if (sqrt(sum((beta_new - beta)^2)) < tol) {
          beta <- beta_new
          ll_final <- log_likelihood(beta, X, y)
          history[[iter + 1]] <- list(
            beta = beta,
            grad = c(0, 0),
            H = H,
            log_lik = ll_final,
            iter = iter + 1
          )
          break
        }
        
        beta <- as.vector(beta_new)
      }, error = function(e) {
        history[[iter]]$error <<- paste("Matrix inversion failed:", e$message)
        return(history)
      })
      
      # Check if error occurred
      if (!is.null(history[[iter]]$error)) {
        break
      }
    }
    
    return(history)
  }
  
  # Run Newton-Raphson
  observeEvent(input$run_nr, {
    req(values$data)
    
    beta_init <- c(input$nr_start_beta0, input$nr_start_beta1)
    results <- newton_raphson_logistic(
      values$data$X, 
      values$data$y, 
      beta_init, 
      max_iter = input$nr_max_iter
    )
    
    values$nr_results <- results
    
    # Update slider max
    updateSliderInput(session, "nr_show_iter", 
                     max = length(results),
                     value = 1)
  })
  
  # Display Newton-Raphson iteration info
  output$nr_iteration_info <- renderText({
    if (is.null(values$nr_results)) {
      return("Click 'Run Newton-Raphson' to start optimization")
    }
    
    iter_idx <- min(input$nr_show_iter, length(values$nr_results))
    iter_data <- values$nr_results[[iter_idx]]
    
    # Check for error
    error_msg <- ""
    if (!is.null(iter_data$error)) {
      error_msg <- paste0("\n\n⚠️ ERROR: ", iter_data$error, 
                         "\n\nTry starting values closer to 0 or use a different dataset.")
    }
    
    paste0(
      "Iteration: ", iter_data$iter, " / ", length(values$nr_results), "\n\n",
      "Current Parameters:\n",
      "  β₀ = ", round(iter_data$beta[1], 4), "\n",
      "  β₁ = ", round(iter_data$beta[2], 4), "\n\n",
      "Log-Likelihood: ", round(iter_data$log_lik, 4), "\n\n",
      "Gradient:\n",
      "  ∂ℓ/∂β₀ = ", round(iter_data$grad[1], 4), "\n",
      "  ∂ℓ/∂β₁ = ", round(iter_data$grad[2], 4), "\n\n",
      "Gradient Magnitude: ", round(sqrt(sum(iter_data$grad^2)), 4), "\n\n",
      "Hessian:\n",
      "  [", round(iter_data$H[1,1], 2), "  ", round(iter_data$H[1,2], 2), "]\n",
      "  [", round(iter_data$H[2,1], 2), "  ", round(iter_data$H[2,2], 2), "]",
      error_msg
    )
  })
  
  # Geometric visualization of Newton-Raphson step
  output$nr_geometry_plot <- renderPlot({
    if (is.null(values$nr_results)) {
      return(NULL)
    }
    
    req(values$data)
    iter_idx <- min(input$nr_show_iter, length(values$nr_results))
    iter_data <- values$nr_results[[iter_idx]]
    
    # Check for error
    if (!is.null(iter_data$error)) {
      par(mar = c(2, 2, 3, 2))
      plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
           xlab = "", ylab = "", main = "Error Occurred", axes = FALSE)
      text(0.5, 0.5, iter_data$error, col = "red", cex = 1.2, font = 2)
      text(0.5, 0.3, "Try different starting values\n(closer to 0)", 
           col = "darkred", cex = 1)
      return()
    }
    
    # Create grid for contour plot
    beta0_current <- iter_data$beta[1]
    beta1_current <- iter_data$beta[2]
    
    # Create grid around current point
    beta0_seq <- seq(beta0_current - 3, beta0_current + 3, length.out = 60)
    beta1_seq <- seq(beta1_current - 3, beta1_current + 3, length.out = 60)
    grid <- expand.grid(beta0 = beta0_seq, beta1 = beta1_seq)
    
    grid$log_lik <- apply(grid, 1, function(row) {
      log_likelihood(c(row[1], row[2]), values$data$X, values$data$y)
    })
    
    # Calculate quadratic approximation
    # Q(β) ≈ ℓ(β₀) + ∇ℓ(β₀)ᵀ(β - β₀) + 0.5(β - β₀)ᵀH(β₀)(β - β₀)
    ll_current <- iter_data$log_lik
    grad <- iter_data$grad
    H <- iter_data$H
    
    grid$quadratic_approx <- apply(grid, 1, function(row) {
      beta_diff <- c(row[1] - beta0_current, row[2] - beta1_current)
      ll_current + sum(grad * beta_diff) + 
        0.5 * sum(beta_diff * (H %*% beta_diff))
    })
    
    # Plot
    par(mfrow = c(1, 2), mar = c(4, 4, 3, 2))
    
    # Left: True log-likelihood
    ll_matrix <- matrix(grid$log_lik, nrow = length(beta0_seq))
    contour(beta1_seq, beta0_seq, t(ll_matrix), 
            nlevels = 20,
            xlab = "β₁ (Slope)", 
            ylab = "β₀ (Intercept)",
            main = paste("True Log-Likelihood\nIteration", iter_data$iter),
            col = "blue")
    
    # Add current point
    points(beta1_current, beta0_current, pch = 19, col = "red", cex = 2)
    
    # Add gradient arrow (only if gradient is non-zero)
    if (sqrt(sum(grad^2)) > 1e-6) {
      grad_scale <- 0.3
      arrows(beta1_current, beta0_current,
             beta1_current + grad_scale * grad[2],
             beta0_current + grad_scale * grad[1],
             col = "red", lwd = 2, length = 0.15)
    }
    
    # Add next iteration if available
    if (iter_idx < length(values$nr_results)) {
      next_iter <- values$nr_results[[iter_idx + 1]]
      if (is.null(next_iter$error)) {
        points(next_iter$beta[2], next_iter$beta[1], 
               pch = 17, col = "green", cex = 2)
        arrows(beta1_current, beta0_current,
               next_iter$beta[2], next_iter$beta[1],
               col = "green", lwd = 2, lty = 2, length = 0.15)
        legend("topright", 
               c("Current", "Gradient", "Next Step"),
               col = c("red", "red", "green"),
               pch = c(19, NA, 17),
               lty = c(NA, 1, 2),
               lwd = c(NA, 2, 2),
               cex = 0.8)
      } else {
        legend("topright", 
               c("Current (Error)", "Gradient"),
               col = c("red", "red"),
               pch = c(19, NA),
               lty = c(NA, 1),
               lwd = c(NA, 2),
               cex = 0.8)
      }
    } else {
      legend("topright", 
             c("Current (Converged)", "Gradient"),
             col = c("red", "red"),
             pch = c(19, NA),
             lty = c(NA, 1),
             lwd = c(NA, 2),
             cex = 0.8)
    }
    
    # Right: Quadratic approximation
    quad_matrix <- matrix(grid$quadratic_approx, nrow = length(beta0_seq))
    contour(beta1_seq, beta0_seq, t(quad_matrix), 
            nlevels = 20,
            xlab = "β₁ (Slope)", 
            ylab = "β₀ (Intercept)",
            main = "Quadratic Approximation\n(Newton-Raphson)",
            col = "purple")
    
    points(beta1_current, beta0_current, pch = 19, col = "red", cex = 2)
    
    # Maximum of quadratic approximation
    if (iter_idx < length(values$nr_results)) {
      next_iter <- values$nr_results[[iter_idx + 1]]
      if (is.null(next_iter$error)) {
        points(next_iter$beta[2], next_iter$beta[1], 
               pch = 17, col = "green", cex = 2)
        arrows(beta1_current, beta0_current,
               next_iter$beta[2], next_iter$beta[1],
               col = "green", lwd = 2, lty = 2, length = 0.15)
        legend("topright", 
               c("Current", "Max of Approx"),
               col = c("red", "green"),
               pch = c(19, 17),
               cex = 0.8)
      }
    }
  })
  
  # Newton-Raphson convergence plot
  output$nr_convergence_plot <- renderPlotly({
    if (is.null(values$nr_results)) {
      return(plot_ly() %>% 
               layout(title = "Run Newton-Raphson to see convergence"))
    }
    
    # Extract data from results
    n_iter <- length(values$nr_results)
    beta0_path <- sapply(values$nr_results, function(x) x$beta[1])
    beta1_path <- sapply(values$nr_results, function(x) x$beta[2])
    ll_path <- sapply(values$nr_results, function(x) x$log_lik)
    grad_norm <- sapply(values$nr_results, function(x) sqrt(sum(x$grad^2)))
    
    # Create convergence plots
    fig <- subplot(
      plot_ly() %>%
        add_trace(x = 1:n_iter, y = beta0_path, 
                 type = "scatter", mode = "lines+markers",
                 name = "β₀", line = list(color = "blue")) %>%
        add_trace(x = 1:n_iter, y = beta1_path, 
                 type = "scatter", mode = "lines+markers",
                 name = "β₁", line = list(color = "red")) %>%
        layout(yaxis = list(title = "Parameter Value"),
               xaxis = list(title = "Iteration")),
      
      plot_ly() %>%
        add_trace(x = 1:n_iter, y = ll_path, 
                 type = "scatter", mode = "lines+markers",
                 name = "Log-Likelihood", 
                 line = list(color = "green")) %>%
        layout(yaxis = list(title = "Log-Likelihood"),
               xaxis = list(title = "Iteration")),
      
      plot_ly() %>%
        add_trace(x = 1:n_iter, y = grad_norm, 
                 type = "scatter", mode = "lines+markers",
                 name = "Gradient Norm", 
                 line = list(color = "purple")) %>%
        layout(yaxis = list(title = "Gradient Magnitude (log scale)", type = "log"),
               xaxis = list(title = "Iteration")),
      
      nrows = 1, shareX = TRUE
    ) %>%
      layout(title = "Newton-Raphson Convergence")
    
    fig
  })
}

# Run the application
shinyApp(ui = ui, server = server)
