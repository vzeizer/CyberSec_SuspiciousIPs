# Cyber Security Project: Suspicious IPs and Machine Learning Modeling

## time...

![logo](images/1_timetaken.png)

## time...

![logo](images/2_timetaken_scstatus.png)

## time...

![logo](images/3_hist_scstatus.png)

## time...

![logo](images/4_conexoesPOST.png)

## time...

![logo](images/5_forcabruta.png)

## time...

![logo](images/6_maisacessados_webapp.png)

## Machine Learning Modeling

# Text and Numeric Preprocessing Pipelines Documentation
Version: 1.0
Last Updated: January 18, 2025

## Overview
This document outlines the design and implementation of a robust preprocessing system that handles both text and numeric features in machine learning pipelines. The system is built using scikit-learn's Pipeline and ColumnTransformer components, providing a scalable and maintainable solution for feature engineering.

## Architecture

### Core Components
1. Text Pipeline
2. Numeric Pipeline
3. Combined Preprocessor

## Detailed Component Specifications

### Text Pipeline

#### Purpose
Transforms multiple text columns into a unified TF-IDF representation while handling various edge cases and data quality issues.

#### Key Features
- Robust text concatenation with error handling
- N-gram support (unigrams and bigrams)
- Automated removal of high and low frequency terms
- Unicode accent stripping
- Missing value handling

#### Implementation Details
```python
text_pipeline = Pipeline([
    ('concatenate', FunctionTransformer(concatenate_text_columns)),
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    ))
])
```

#### Design Decisions
1. **Concatenation Strategy**:
   - Uses space-based joining instead of other delimiters
   - Implements safe type conversion to handle mixed data types
   - Strips whitespace to prevent token artifacts

2. **TF-IDF Parameters**:
   - min_df=2: Removes rare terms that appear in < 2 documents
   - max_df=0.95: Removes overused terms appearing in > 95% of documents
   - ngram_range=(1, 2): Captures phrase patterns while maintaining computational efficiency

### Numeric Pipeline

#### Purpose
Standardizes numeric features while handling missing values and outliers.

#### Key Features
- Median-based missing value imputation
- Robust Scaler normalization, which is proper for data with outliers
- Automatic handling of sparse data

#### Implementation Details
```python
numeric_pipeline = Pipeline([
    ('imputer', FunctionTransformer(handle_numeric_missing)),
    ('scaler', RobustScaler())
])
```

#### Design Decisions
1. **Imputation Strategy**:
   - Uses median instead of mean to handle skewed distributions
   - Performs imputation before scaling to prevent data leakage
   - Maintains column-wise statistics for consistent transformation

2. **Scaling Approach**:
   - Implements robust scaler normalization for compatibility with various ML algorithms
   - Handles outliers through Quantiles

## Combined Preprocessor

### Architecture Benefits
1. **Modularity**: Each pipeline can be modified independently
2. **Maintainability**: Clear separation of concerns between text and numeric processing
3. **Scalability**: Easy to add new features or modify existing transformations
4. **Reproducibility**: Consistent application of transformations across training and test sets

### Usage Guidelines

#### Training Phase
```python
preprocessor = create_preprocessor(text_columns, numeric_columns)
X_train_transformed = preprocessor.fit_transform(X_train)
```

#### Inference Phase
```python
X_test_transformed = preprocessor.transform(X_test)
```

### Best Practices
1. **Data Validation**:
   - Verify column names and types before processing
   - Check for unexpected missing value patterns
   - Monitor frequency distributions of text features

2. **Performance Optimization**:
   - Use sparse matrices for text features when possible
   - Monitor memory usage with large datasets
   - Consider batch processing for very large datasets

3. **Model Integration**:
   - Include preprocessor in the full model pipeline
   - Use cross-validation to prevent data leakage
   - Save fitted preprocessor with model artifacts

## Error Handling and Edge Cases

### Text Processing
- Handles NULL values gracefully
- Manages mixed data types through safe conversion
- Processes malformed Unicode characters
- Handles empty strings and whitespace

### Numeric Processing
- Deals with infinity values
- Manages string values in numeric columns
- Handles missing value patterns
- Processes outliers effectively

## Performance Considerations

### Memory Usage
- Text pipeline may create sparse matrices
- Numeric pipeline maintains dense arrays
- Combined preprocessor optimizes memory based on data characteristics

### Computational Efficiency
- Vectorized operations for numeric processing
- Efficient text concatenation using filtered joins
- Optimized TF-IDF computation with sparse matrices

## Testing and Validation

### Unit Tests
Create tests for:
- Edge cases in text concatenation
- Missing value handling in numeric pipeline
- Combined pipeline transformation consistency

### Integration Tests
Verify:
- Pipeline behavior with different data types
- Transformation consistency across datasets
- Memory usage with large datasets

## Future Improvements

### Planned Enhancements
1. Add support for custom tokenization
2. Implement feature selection capabilities
3. Add parallel processing for large datasets
4. Enhance outlier detection and handling

### Maintenance Notes
- Monitor scikit-learn version compatibility
- Update documentation with new features
- Maintain test coverage for critical components

## Conclusion
This preprocessing system provides a robust, maintainable, and efficient solution for handling both text and numeric features in machine learning pipelines. Its modular design allows for easy modifications and extensions while maintaining high performance and reliability.

![logo](images/7_data_ml.png)

![logo](images/8_ml_reglog.png)

![logo](images/9_xai_text.png)