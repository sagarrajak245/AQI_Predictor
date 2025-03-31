# R Code for AQI Prediction Pipeline

# Load required packages
library(tidyverse)
library(lubridate)
library(xgboost)
library(caret)
library(recipes)
library(jsonlite)
library(zoo)  # Added for na.approx function

# 1. AQI Calculation Functions
calculate_aqi_subindex <- function(value, breakpoints) {
  if (is.na(value)) return(NA)  # Handle NA values
  
  for (bp in breakpoints) {
    low <- bp[1]; high <- bp[2]; sub_low <- bp[3]; sub_high <- bp[4]
    if (!is.na(value) && value >= low && value <= high) {
      return(((sub_high - sub_low) / (high - low)) * (value - low) + sub_low)
    }
  }
  return(NA)
}

aqi_breakpoints <- list(
  "PM2.5" = list(c(0, 30, 0, 50), c(31, 60, 51, 100), c(61, 90, 101, 200),
                 c(91, 120, 201, 300), c(121, 250, 301, 400), c(251, 500, 401, 500)),
  "PM10" = list(c(0, 50, 0, 50), c(51, 100, 51, 100), c(101, 250, 101, 200),
                c(251, 350, 201, 300), c(351, 430, 301, 400), c(431, 600, 401, 500)),
  "NO2" = list(c(0, 40, 0, 50), c(41, 80, 51, 100), c(81, 180, 101, 200),
               c(181, 280, 201, 300), c(281, 400, 301, 400), c(401, 1000, 401, 500))
)


# 2. Data Processing Functions
compute_daily_aqi <- function(df) {
  # Handle the date parsing errors with suppressWarnings
  df <- df %>%
    mutate(`From Date` = suppressWarnings(ymd_hms(`From Date`))) %>%
    mutate(Date = as_date(`From Date`))
  
  # Filter out rows with NA dates
  df <- df %>% filter(!is.na(Date))
  
  # Calculate AQI for each pollutant
  for (pollutant in names(aqi_breakpoints)) {
    if (pollutant %in% colnames(df)) {  # Check if pollutant column exists
      col_name <- paste0(pollutant, "_AQI")
      df[[col_name]] <- sapply(df[[pollutant]], function(x) 
        calculate_aqi_subindex(x, aqi_breakpoints[[pollutant]]))
    }
  }
  
  # Calculate daily AQI using dplyr 1.1.0+ syntax for across()
  daily_aqi <- df %>%
    group_by(Date) %>%
    summarise(across(ends_with("_AQI"), \(x) max(x, na.rm = TRUE))) %>%
    mutate(AQI = pmax(PM2.5_AQI, PM10_AQI, NO2_AQI, na.rm = TRUE))
  
  return(daily_aqi)
}

create_lag_features <- function(df, lags = c(1, 2, 3)) {
  for (lag in lags) {
    df[[paste0("AQI_Lag", lag)]] <- lag(df$AQI, lag)
  }
  # Handle rows with NAs due to lag
  return(df %>% filter(!is.na(AQI_Lag3)))  # Filter instead of na.omit()
}

# 3. Main Data Processing
process_data <- function(data_paths) {
  # Read datasets with error handling
  datasets <- list()
  tryCatch({
    datasets <- list(
      BandraKurlaComplex = read_csv(data_paths$bkc, show_col_types = FALSE),
      Kurla = read_csv(data_paths$kurla, show_col_types = FALSE),
      Colaba = read_csv(data_paths$colaba, show_col_types = FALSE)
    )
  }, error = function(e) {
    stop("Error reading data files: ", e$message)
  })
  
  # Compute daily AQI
  daily_datasets <- lapply(datasets, compute_daily_aqi)
  
  # Interpolate missing values
  daily_datasets <- lapply(daily_datasets, function(df) {
    df %>%
      mutate(across(where(is.numeric), \(x) {
        # Handle -Inf values that might appear from max() with all NAs
        x[is.infinite(x)] <- NA
        # Apply na.approx from the zoo package
        na.approx(x, na.rm = FALSE, rule = 2)
      })) %>%
      fill(where(is.numeric), .direction = "down")
  })
  
  # Combine datasets and create lag features
  full_data <- bind_rows(daily_datasets, .id = "Station") %>%
    arrange(Date) %>%
    create_lag_features()
  
  return(full_data)
}

# 4. Model Training
train_aqi_model <- function(data_paths) {
  full_data <- process_data(data_paths)
  
  # Check if data is available
  if (nrow(full_data) == 0) {
    stop("No data available after processing. Check input data files.")
  }
  
  # Prepare data
  X <- full_data %>% select(-Date, -AQI, -Station)
  y <- full_data$AQI
  
  # Split data
  set.seed(123)  # For reproducibility
  train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- y[-train_idx]
  
  # Preprocessing
  preprocessor <- preProcess(X_train, method = c("center", "scale", "medianImpute"))  # Added imputation
  X_train_scaled <- predict(preprocessor, X_train)
  X_test_scaled <- predict(preprocessor, X_test)
  
  # Train XGBoost model with error handling
  xgb_model <- tryCatch({
    xgboost(
      data = as.matrix(X_train_scaled),
      label = y_train,
      nrounds = 100,
      objective = "reg:squarederror",
      eta = 0.1,
      max_depth = 5,
      verbose = 0
    )
  }, error = function(e) {
    stop("Error training XGBoost model: ", e$message)
  })
  
  # Evaluate with error handling
  train_pred <- predict(xgb_model, as.matrix(X_train_scaled))
  test_pred <- predict(xgb_model, as.matrix(X_test_scaled))
  
  # Use safer R2 function to handle possible NAs
  custom_R2 <- function(pred, obs) {
    valid <- !is.na(pred) & !is.na(obs)
    if (sum(valid) < 2) return(NA)
    cor(pred[valid], obs[valid])^2
  }
  
  metrics <- list(
    train_r2 = custom_R2(train_pred, y_train),
    test_r2 = custom_R2(test_pred, y_test),
    mae = mean(abs(test_pred - y_test), na.rm = TRUE),
    rmse = sqrt(mean((test_pred - y_test)^2, na.rm = TRUE))
  )
  
  return(list(
    model = xgb_model,
    preprocessor = preprocessor,
    metrics = metrics,
    test_data = list(X_test = X_test, y_test = y_test),
    train_idx = train_idx,
    full_data = full_data
  ))
}

# 5. Save Model Components
save_model <- function(model_obj, path = "aqi_model") {
  saveRDS(model_obj$model, file = paste0(path, "_xgb.rds"))
  saveRDS(model_obj$preprocessor, file = paste0(path, "_preprocessor.rds"))
  write_json(model_obj$metrics, paste0(path, "_metrics.json"))
  
  # Also save metadata for troubleshooting
  metadata <- list(
    date_created = Sys.time(),
    variables = names(model_obj$test_data$X_test),
    n_test_samples = nrow(model_obj$test_data$X_test)
  )
  write_json(metadata, paste0(path, "_metadata.json"))
}

# 6. Prediction Function
predict_aqi <- function(model_obj, new_data) {
  new_data_scaled <- predict(model_obj$preprocessor, new_data)
  predict(model_obj$model, as.matrix(new_data_scaled))
}

#  Visualization Functions (4 different visualizations)

# 1. Time Series Plot of AQI by Station
plot_aqi_time_series <- function(data_paths) {
  # Read and process data
  datasets <- list(
    BandraKurlaComplex = read_csv(data_paths$bkc, show_col_types = FALSE),
    Kurla = read_csv(data_paths$kurla, show_col_types = FALSE),
    Colaba = read_csv(data_paths$colaba, show_col_types = FALSE)
  )
  
  # Process each dataset to get daily AQI
  daily_datasets <- lapply(datasets, compute_daily_aqi)
  
  # Combine datasets for plotting
  combined_data <- bind_rows(daily_datasets, .id = "Station")
  
  # Create the time series plot
  p <- ggplot(combined_data, aes(x = Date, y = AQI, color = Station)) +
    geom_line() +
    geom_smooth(method = "loess", span = 0.2, se = FALSE, linetype = "dashed") +
    labs(title = "AQI Time Series by Station",
         x = "Date",
         y = "Air Quality Index (AQI)") +
    theme_minimal() +
    theme(legend.position = "bottom") +
    scale_color_brewer(palette = "Set1")
  
  # Save the plot
  ggsave("aqi_time_series.png", p, width = 10, height = 6)
  
  return(p)
}

# 2. Pollutant Contribution to AQI
plot_pollutant_contribution <- function(data_paths) {
  # Get data for one station
  station_data <- read_csv(data_paths$bkc, show_col_types = FALSE)
  daily_data <- compute_daily_aqi(station_data)
  
  # Gather the data to long format for visualization
  long_data <- daily_data %>%
    select(Date, PM2.5_AQI, PM10_AQI, NO2_AQI) %>%
    pivot_longer(cols = c(PM2.5_AQI, PM10_AQI, NO2_AQI),
                 names_to = "Pollutant",
                 values_to = "AQI_Value") %>%
    mutate(Pollutant = gsub("_AQI", "", Pollutant))
  
  # Create the plot
  p <- ggplot(long_data, aes(x = Date, y = AQI_Value, fill = Pollutant)) +
    geom_area(alpha = 0.7, position = "identity") +
    labs(title = "Contribution of Pollutants to AQI Over Time",
         x = "Date",
         y = "AQI Value") +
    theme_minimal() +
    theme(legend.position = "bottom") +
    scale_fill_brewer(palette = "Set2")
  
  # Save the plot
  ggsave("pollutant_contribution.png", p, width = 10, height = 6)
  
  return(p)
}

# 3. Monthly AQI Boxplot
plot_monthly_aqi <- function(data_paths) {
  # Read and process data
  datasets <- list(
    BandraKurlaComplex = read_csv(data_paths$bkc, show_col_types = FALSE),
    Kurla = read_csv(data_paths$kurla, show_col_types = FALSE),
    Colaba = read_csv(data_paths$colaba, show_col_types = FALSE)
  )
  
  # Process each dataset to get daily AQI
  daily_datasets <- lapply(datasets, compute_daily_aqi)
  
  # Combine datasets for plotting
  combined_data <- bind_rows(daily_datasets, .id = "Station")
  
  # Add month information
  data_with_month <- combined_data %>%
    mutate(Month = month(Date, label = TRUE, abbr = TRUE),
           Year = year(Date))
  
  # Create the plot
  p <- ggplot(data_with_month, aes(x = Month, y = AQI, fill = Month)) +
    geom_boxplot() +
    facet_wrap(~Year, scales = "free_x") +
    labs(title = "Monthly AQI Distribution",
         x = "Month",
         y = "Air Quality Index (AQI)") +
    theme_minimal() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_brewer(palette = "Spectral")
  
  # Save the plot
  ggsave("monthly_aqi.png", p, width = 10, height = 6)
  
  return(p)
}

# 4. Prediction vs Actual Plot
plot_prediction_vs_actual <- function(model_obj) {
  # Get the full data and train index from model object
  full_data <- model_obj$full_data
  train_idx <- model_obj$train_idx
  
  # Create test data
  test_data <- full_data[-train_idx, ]
  
  # Prepare features for prediction
  X_test <- test_data %>% select(-Date, -AQI, -Station)
  
  # Get actual values
  actual_values <- test_data$AQI
  
  # Make predictions
  predicted_values <- predict_aqi(model_obj, X_test)
  
  # Create data frame for plotting
  plot_data <- data.frame(
    Date = test_data$Date,
    Actual = actual_values,
    Predicted = predicted_values
  )
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = Date)) +
    geom_line(aes(y = Actual, color = "Actual"), size = 1) +
    geom_line(aes(y = Predicted, color = "Predicted"), size = 1, linetype = "dashed") +
    labs(title = "AQI: Actual vs Predicted Values",
         x = "Date",
         y = "AQI Value",
         color = "Type") +
    theme_minimal() +
    theme(legend.position = "bottom") +
    scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))
  
  # Save the plot
  ggsave("prediction_vs_actual.png", p, width = 10, height = 6)
  
  return(p)
}

# Modified run_pipeline function with visualization integration
run_pipeline <- function(visualize = TRUE) {
  # Specify data paths (update these to your actual file paths)
  data_paths <- list(
    bkc = file.path(getwd(), "BandraKurlaComplexMumbaiIITM.csv"),
    kurla = file.path(getwd(), "KurlaMumbaiMPCB.csv"),
    colaba = file.path(getwd(), "ColabaMumbaiMPCB.csv")
  )
  
  # Check if files exist
  for (path in data_paths) {
    if (!file.exists(path)) {
      stop("File not found: ", path)
    }
  }
  
  # Generate pre-modeling visualizations if requested
  if (visualize) {
    cat("Creating initial data visualizations...\n")
    
    # Generate time series plot
    cat("  - Generating AQI time series plot...\n")
    time_series_plot <- plot_aqi_time_series(data_paths)
    
    # Generate pollutant contribution plot
    cat("  - Generating pollutant contribution plot...\n")
    pollutant_plot <- plot_pollutant_contribution(data_paths)
    
    # Generate monthly AQI boxplot
    cat("  - Generating monthly AQI boxplot...\n")
    monthly_plot <- plot_monthly_aqi(data_paths)
    
    cat("Initial visualizations completed and saved.\n")
  }
  
  # Check if model already exists
  if (!file.exists("aqi_model_xgb.rds")) {
    cat("Training new AQI prediction model...\n")
    model_obj <- train_aqi_model(data_paths)
    save_model(model_obj)
    cat("Model training complete.\n")
    cat("Model metrics:\n")
    print(model_obj$metrics)
    
    # Generate post-modeling visualization
    if (visualize) {
      cat("Creating model performance visualization...\n")
      prediction_plot <- plot_prediction_vs_actual(model_obj)
      cat("Model performance visualization completed and saved.\n")
    }
  } else {
    cat("Model already exists. Use load_model() to load it or delete the file to train a new one.\n")
    
    # If user wants visualizations but model already exists, load it and visualize
    if (visualize) {
      cat("Loading existing model for visualization...\n")
      model_obj <- load_model()
      
      # We need the full data and train_idx for the prediction plot
      # Since these aren't saved with the model, we'll need to process the data again
      full_data <- process_data(data_paths)
      
      # Create a simplified model object with the necessary components
      simplified_model_obj <- list(
        model = model_obj$model,
        preprocessor = model_obj$preprocessor,
        full_data = full_data,
        train_idx = createDataPartition(full_data$AQI, p = 0.8, list = FALSE)
      )
      
      cat("Creating model performance visualization...\n")
      prediction_plot <- tryCatch({
        plot_prediction_vs_actual(simplified_model_obj)
      }, error = function(e) {
        cat("Could not create prediction vs actual plot with existing model:", e$message, "\n")
        cat("Consider retraining the model with visualize=TRUE for complete visualizations.\n")
        NULL
      })
      
      if (!is.null(prediction_plot)) {
        cat("Model performance visualization completed and saved.\n")
      }
    }
  }
  
  cat("Pipeline execution completed.\n")
}

# Load saved model
load_model <- function(path = "aqi_model") {
  model <- readRDS(paste0(path, "_xgb.rds"))
  preprocessor <- readRDS(paste0(path, "_preprocessor.rds"))
  return(list(model = model, preprocessor = preprocessor))
}

# Example usage
run_pipeline(visualize = TRUE)