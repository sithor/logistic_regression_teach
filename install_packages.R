# Install required packages for the Logistic Regression Shiny App

cat("Installing required R packages...\n")

# List of required packages
packages <- c("shiny", "ggplot2", "plotly")

# Install packages that are not already installed
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(paste("Installing", pkg, "...\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  } else {
    cat(paste(pkg, "is already installed.\n"))
  }
}

cat("\nAll required packages are installed!\n")
cat("You can now run the app with: shiny::runApp('app.R')\n")
