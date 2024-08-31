## Week 1: Introduction and Data Exploration

**Big Picture**

* **Inputs and Outputs:** Understanding the nature of input features and output variables (single or multiple) is crucial for framing the problem.
* **Business Objective:** Defining the specific goal the model aims to achieve provides direction for model selection and evaluation. 
* **Algorithm Selection:** The choice of algorithm depends on the problem type:
    * **Supervised/Unsupervised/Reinforcement Learning**
    * **Regression/Classification**
    * **Continuous learning/periodic updates**
    * **Batch/Online learning**
* **Performance Measure:** Selecting an appropriate metric to evaluate model performance is essential for judging its effectiveness.
    * **Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE)
    * **Classification:** Precision, Recall, F1-Score, Accuracy
* **Current Solution:** Identifying existing solutions provides a baseline for comparison and insights into potential improvements.

**Get the Data**

* **Data Sources:** Data can be spread across various sources: tables, files, documents, local storage, or the web.
* **Data Structure:** Understanding the structure of the data (structured or unstructured) is crucial for preprocessing and analysis.
* **Data Understanding:** Analyzing data statistics and distributions is vital:
    * `info()`: Provides a summary of the data, including data types and missing values.
    * `describe()`: Calculates basic descriptive statistics (mean, standard deviation, quartiles, etc.).
    * **Histograms:** Visualize the distribution of examples by features/labels.
* **Data Splitting:**  Dividing data into training and test sets is essential for model validation and generalization:
    * `train_test_split()`: Performs random sampling to create training and test sets.
    * **Data Snooping Bias:**  Avoid peeking into the test set during model development to prevent biased estimations. 
    * `train_test_split()`: Can handle multiple datasets with identical row numbers, ensuring consistent indexing.
    * **Stratified Sampling:**  Divide the population into homogeneous groups (strata) to ensure representative sampling.
    * `StratifiedShuffleSplit.split()`: Facilitates stratified sampling for dividing data between train and test.

**Visualize the Data**

* **Feature Relationship:** Visualizations help understand the relationships between features and their impact on the output:
    * **Scatter plots:** Show the relationship between two features.
    * **Correlation Coefficient:** Measures linear correlation between features (-1: strong negative, +1: strong positive, 0: no correlation).
    * **Correlation Matrix:** Visualizes the correlation coefficient using a heatmap.
    * **Rank Correlation:** Use for analyzing non-linear relationships between features.

## Week 2: Preprocessing and Feature Engineering

**Preprocessing Data**

* **Common Data Issues:** Real-world data often contains:
    * Outliers
    * Missing values
    * Different scales
    * Non-numeric attributes
    * Non-amenable distributions
* **Identifying Missing Values:** Use `isna().sum()` to find columns with missing values.
* **Imputing Missing Values:**  
    * `SimpleImputer`: Fill missing values using strategies like mean, median, or most frequent.
    * `KNNImputer`: Impute missing values based on nearest neighbors using Euclidean distance.
    * **Dropping Missing Values:** Use `dropna()` or `drop()` to remove rows with missing values.
* **Handling Non-numeric Attributes:**
    * `OrdinalEncoder`: Convert categorical features to ordinal numbers.
    * `OneHotEncoder`: Create binary features for each category (one-hot encoding).
* **Feature Scaling:**  Bring features to a similar scale to improve model performance:
    * `MinMaxScaler`: Scales features to a specific range (e.g., [0, 1]).
    * `StandardScaler`: Standardizes features to have zero mean and unit variance.
* **Pipelines:** Streamline sequential operations using pipelines:
    * `Pipeline`:  Executes a chain of transformations in a defined order.
    * `ColumnTransformer`: Handle different data types within a pipeline by applying specific transformations to different columns.

**Feature Extraction**

* **DictVectorizer:** Converts data in dictionary format to a matrix, typically used for handling categorical features.
* **FeatureHasher:** Outputs a sparse matrix by hashing features to determine column indices, useful for high-speed and low-memory vectorization.
* **Image and Text Features:** Additional APIs within `sklearn.feature_extraction` can extract features from image and text data.

**Imputing Data**

* **SimpleImputer:** Fills missing values using specified strategies:
    * `mean`:  Replace missing values with the mean of the feature.
    * `median`: Replace missing values with the median of the feature.
    * `most_frequent`: Replace missing values with the most frequent value of the feature.
    * `constant`: Replace missing values with a user-defined constant.
* **KNNImputer:**  Finds nearest neighbors based on Euclidean distance and uses their mean to impute missing values. 
* **MissingIndicator:** Creates a binary feature indicating the presence of missing values.

**Feature Scaling**

* **StandardScaler:** Standardizes features using z-score normalization (zero mean and unit variance).
* **MinMaxScaler:**  Scales features to a specified range (e.g., [0, 1]).
* **MaxAbsScaler:** Scales features by dividing each feature by its maximum absolute value, bringing values within the range [-1, 1].

**Other Transformers**

* **FunctionTransformer:**  Applies a user-defined function to transform features.
* **PolynomialFeatures:**  Generates polynomial features (e.g., x1^2, x1*x2, x2^2).
* **KBinsDiscretizer:**  Divides a continuous variable into bins and applies one-hot or ordinal encoding.

## Week 3: Model Selection and Evaluation

**Select and Train ML Model**

* **Baseline Model:** Create a simple model (e.g., linear regression) to establish a baseline for performance.
* **Model Evaluation:** Assess performance on both training and test sets using:
    * `mean_squared_error()`: Calculates Mean Squared Error.
    * `mean_absolute_error()`: Calculates Mean Absolute Error.
    * `r2_score()`:  Calculates R-squared (coefficient of determination).
* **Cross-Validation:** Employ cross-validation for a more robust performance evaluation:
    * Create multiple validation sets from the training set.
    * Generate separate MSE values for each validation set.
    * Use mean/standard deviation of MSE values for overall evaluation. 
* **Model Selection:**  Experiment with different models and select promising candidates based on performance and complexity. 
* **Addressing Underfitting:**
    * Increase model capacity.
    * Reduce constraints/regularization.
* **Addressing Overfitting:**
    * Use more data.
    * Choose a simpler model. 
    * Apply more constraints/regularization. 
* **Hyperparameter Tuning:** 
    * `GridSearchCV`:  Exhaustively search all possible combinations of hyperparameters within a predefined grid. 
    * `RandomizedSearchCV`: Randomly sample hyperparameters for a more efficient search. 
* **Parameter Grid:** Define the hyperparameter space using `param_grid`. 
* **Parallel Processing:** Set `n_jobs=-1` to enable parallel processing for faster model building. 
* **Error Handling:**  Set `error_score` to handle model failures (e.g., set to 0 to continue even if one model fails). 
* **Best Model Selection:** Access the best model and hyperparameters using `best_params_` and `best_estimator_`.
* **Refitting:**  Retrain the best estimator on the full training set (using `refit=True`) to improve its performance. 
* **Feature Importance:**  Analyze `best_estimator_.feature_importances_` to identify important features and potentially drop less relevant ones. 
* **Specialized Cross-Validation:** Some models (e.g., Lasso, Ridge, ElasticNet) offer specialized cross-validation methods.
* **Confidence Interval:**  Calculate the 95% confidence interval for the evaluation metric using `stats.t.interval()`

**Present Your Solution**

* **Documentation:**  Thorough documentation is essential for reproducibility and understanding the model.
* **Visualizations:** Create clear visualizations to communicate insights and model performance.

**Launch, Monitor and Maintain Your System**

* **System Monitoring:**  Monitor the deployed model for:
    * System outages
    * Degradation of model performance
* **Manual Evaluation:** Periodically perform manual evaluations to assess model accuracy.
* **Data Quality Assessment:**  Regularly check data quality to ensure model performance remains stable. 

**Data Loading Mechanisms**

* **Dataset Loaders (`load_*`)**:  Load toy datasets bundled with scikit-learn.
* **Dataset Fetchers (`fetch_*`):**  Download and load datasets from the internet.
* **Dataset Generators (`make_*`)**: Generate controlled synthetic datasets.
* **Return Types:** Loaders and fetchers return a `Bunch` object (a dictionary with data and target keys), while generators return a tuple (data, target). 
* **Additional Keys:** Loaders and fetchers may contain extra keys:
    * `feature_names`
    * `target_names`
    * `DESCR` (dataset description)
    * `filename` 
* **`return_X_y=True`:**  For loaders and fetchers, setting this parameter to True returns a tuple (data, target) instead of a `Bunch` object.

**Common Dataset Generators**

* `make_regression()`: Generates regression datasets with single or multiple labels.
* `make_blobs()`: Creates a specified number of normally distributed clusters of points for multi-class or unsupervised learning.
* `make_classification()`:  Generates multi-class datasets based on normally distributed clusters.
* `make_multilabel_classification()`:  Generates multi-label datasets. 

## Week 4: Regularization and Polynomial Regression

**Polynomial Regression**

* `interaction_only=True`: Include only interaction features (e.g., x1*x2) in the polynomial transformation.

**Regularization**

* **Ridge Regularization:**
    * Achieved using a `Ridge` object or `SGDRegressor` with `penalty='l2'`.
    * Cross-validation: Use `RidgeCV` or `GridSearchCV` with `SGDRegressor` and `penalty='l2'`.
* **Lasso Regularization:**
    * Achieved using a `Lasso` object or `SGDRegressor` with `penalty='l1'`.
    * Cross-validation: Use `LassoCV` or `GridSearchCV` with `SGDRegressor` and `penalty='l1'`.
* **ElasticNet Regularization:**
    * Achieved using `SGDRegressor` with `penalty='elasticnet'`.
    * Control the combination of L1 and L2 using `l1_ratio`.

**Regularization Parameters**

* **`penalty`:** Specifies the type of regularization: `'l1'`, `'l2'`, or `'elasticnet'`.
* **`alpha`:** A non-negative float value representing the regularization strength.
* **`l1_ratio`:** For `'elasticnet'`, controls the balance between L1 and L2 regularization.

## Week 5: Classification Algorithms and Metrics

**Classification Algorithms**

* **Generic:**  
    * `SGDClassifier`:  Performs classification using Stochastic Gradient Descent (SGD).
* **Specialized:**
    * `LogisticRegression`: Performs logistic regression for binary or multinomial classification. 
    * `Perceptron`: Implements the perceptron algorithm for large-scale learning.
    * `RidgeClassifier`: A classifier variant of the `Ridge` regressor.
    * `LinearSVC`:  A faster implementation of linear support vector classification. 
    * `sklearn.neighbors` classifiers:  K-Nearest Neighbors algorithms.
    * `sklearn.naive_bayes` classifiers: Naive Bayes algorithms.

**Common Classifier Methods**

* `fit(X, y)`:  Train the model using feature matrix `X` and label vector `y`.
* `predict(X)`:  Predict class labels for samples in feature matrix `X`.
* `decision_function(X)`:  Predict confidence scores for samples in feature matrix `X`.
* `score(X, y)`: Calculate the mean accuracy on the given test data and labels. 

**RidgeClassifier**

* **Binary Classification:** Predicts the class based on the sign of the output.
* **Multi-class Classification:** Predicts the class based on the highest output value.
* **Regularization:** Controlled by the `alpha` parameter (higher values mean stronger regularization).
* **Solver Methods:**
    * `sparse_cg`: For large-scale data.
    * `sag`/`saga`: For a large number of features or samples.
    * `lsqr`:  Fastest solution.
    * `auto`: Automatically select the solver.
* **`fit_intercept=False`:** Use if data is already centered.
* **`RidgeClassifierCV`:** Implements `RidgeClassifier` with built-in cross-validation.

**Perceptron**

* **Implementation:**  Uses the same underlying implementation as `SGDClassifier`.
* **Large-Scale Learning:**  Suited for large-scale learning tasks.
* **Iterative Training:** Can be trained iteratively using the `partial_fit` method.
* **Warm Start:**  Set `warm_start=True` to initialize the classifier with weights from the previous run. 

**LogisticRegression**

* **Objective Function:** Minimizes the sum of regularization penalty and cross-entropy loss.
* **`C` Parameter:** Inverse of the regularization strength (higher `C` means weaker regularization).
* **Solver Methods:** 
    * `liblinear`:  Suitable for smaller datasets.
    * `sag`/`saga`: For larger datasets. 
    * `liblinear`, `lbfgs`, `newton-cg`: For unscaled datasets. 
    * `lbfgs`: Default solver.
* **Penalty:**  Defaults to `'l2'`, can be changed to `'l1'`, `'elasticnet'`, or `'none'`.

**SGDClassifier**

* **Optimization:** Uses gradient descent, updating weights with a decreasing learning schedule.
* **Hyperparameters:**
    * `loss`:  Specifies the loss function.
    * `penalty`:  Specifies the regularization type (default is `'elasticnet'`).
    * `learning_rate`: Controls the learning rate schedule (default is `'invscaling'`).
    * `early_stopping`:  Enables early stopping based on validation performance.
* **Loss Functions:**
    * `'log'`:  Equivalent to `LogisticRegression` with `solver='sgd'`.
    * `'perceptron'`:  Equivalent to `Perceptron()`.
    * `'hinge'`:  Equivalent to `LinearSVC()`. 
    * `'squared_error'`:  Least-squares classification.

**Multi-class Strategies**

* **One-vs-Rest (OVR):**  Fits one classifier per class.
* **One-vs-One (OVO):**  Fits one classifier per pair of classes. 
* **Multi-label Classification:** Both OVR and OVO support multi-label classification with an indicator matrix.

**Multi-class Implementation**

* **`sklearn.multiclass`:**  Provides tools for experimenting with different multi-class strategies. 

**Evaluating Classifier Performance**

* **Confusion Matrix:** 
    * `confusion_matrix()`:  Calculates the confusion matrix.
    * `ConfusionMatrixDisplay`:  Visualizes the confusion matrix.
* **Classification Report:**
    * `classification_report()`:  Generates a text report with key classification metrics.
* **Precision-Recall Curve:** 
    * `precision_recall_curve()`:  Calculates precision and recall values for different probability thresholds.
* **ROC Curve:**
    * `roc_curve()`:  Calculates the receiver operating characteristic (ROC) curve. 
* **Handling Binary Metrics in Multi-class Problems:**
    * Use the `average` parameter to average binary metrics across classes:
        * `'macro'`:  Calculate the unweighted mean of binary metrics.
        * `'weighted'`:  Calculate the weighted mean, where weights are based on class support.
        * `'micro'`:  Calculate global metrics by counting true positives, false negatives, and false positives globally. 
        * `'samples'`:  Calculate metrics for each instance and return their average. 

## Week 6:  Naive Bayes and Text Feature Extraction

**Naive Bayes Classifier**

* **Bayes' Theorem:**  Naive Bayes applies Bayes' theorem with the assumption of conditional independence between features. 
* **Implementation:** 
    * `GaussianNB`: For data with Gaussian distributions (no categorical features).
    * `BernoulliNB`: For data with multivariate Bernoulli distributions (binary features). 
    * `MultinomialNB`:  For data with multinomial distributions (e.g., text classification).
    * `CategoricalNB`:  For data with many categorical features.
    * `ComplementNB`:  Often outperforms `MultinomialNB` in text classification tasks with class imbalance.
* **Model Training:** Use the `fit(X, y)` method to train the model on the feature matrix `X` and label vector `y`.
* **Prediction:** Use the `predict(X)` method to predict class labels for new samples.

**Text Feature Extraction**

* **Classes:** -1 and 1 should be used for binary classification problems with Perceptron and `RidgeClassifier`. Use 0 and 1 for `DummyClassifier` and `SGDClassifier`.
* **`CountVectorizer`:**  Converts text into a vector of numerical values, representing word counts (bag-of-words model).
* **`TfidfVectorizer`:** Similar to `CountVectorizer` but also considers term frequency-inverse document frequency (TF-IDF), giving higher weights to rare words.

## Week 7: Softmax Regression and K-Nearest Neighbors

**Softmax Regression**

* **Multi-class Classification:**  Softmax regression is used for solving multi-class classification problems.
* **Softmax Function:**  Normalizes an input vector into a probability distribution.

**K-Nearest Neighbors (KNN)**

* **Instance-based Learning:** KNN is an instance-based learning algorithm that does not build a model explicitly but uses the training data directly.
* **Classification:**  Determined by a majority vote of the k-nearest neighbors of a given point.
* **Implementation:** 
    * `KNeighborsClassifier`: The most common KNN implementation.
    * `RadiusNeighborsClassifier`: Uses neighbors within a fixed radius.
* **Hyperparameters:**
    * `n_neighbors`: Number of neighbors to consider.
    * `weights`: How to weight the neighbors: `'uniform'` (equal weights) or `'distance'` (inverse distance weighting).
    * `algorithm`:  Algorithm for finding nearest neighbors: `'ball_tree'`, `'kd_tree'`, `'brute'`, or `'auto'`.
    * `leaf_size`:  Affects speed and memory usage for `'ball_tree'` and `'kd_tree'`. 
    * `metric`:  Distance metric (default is `'minkowski'`). 
    * `p`:  Power parameter for Minkowski distance (default is 2). 

**KNN Performance**

* **Evaluation:** Use metrics like accuracy score, misclassification error, R-squared, or Root Mean Squared Error (RMSE).
* **Number of Neighbors:** Accuracy generally increases with more neighbors.
* **Overfitting/Underfitting:** Low values of k tend to overfit, while high values of k tend to underfit.
* **Noise and Outliers:** KNN is sensitive to noise and outliers, especially for low values of k. 
* **Curse of Dimensionality:**  Performance degrades as the number of features increases (high dimensionality).

**Text-based ML with KNN**

* **`HashingVectorizer`:** Use for memory and resource constraints, but you lose access to tokens. 
* **`CountVectorizer`:**  Use if you need to access the actual tokens.

## Week 8:  Support Vector Machines

**Support Vector Machines (SVM)**

* **Hyperplane Construction:**  SVMs create a hyperplane or set of hyperplanes to separate data points into different classes.
* **Implementation:**
    * `SVC`: Based on libsvm, uses a radial basis function (RBF) kernel by default.
    * `NuSVC`:  Similar to `SVC` but uses a parameter `nu` instead of `C`.
    * `LinearSVC`: Based on liblinear, uses a linear kernel and scales better with large datasets.
* **`SVC` Parameters:**
    * `C`:  Regularization parameter (strength is inversely proportional to `C`). 
    * `kernel`:  Specifies the kernel type (default is `'rbf'`). 
    * `gamma`:  Kernel coefficient for `'rbf'`, `'poly'`, or `'sigmoid'` kernels (default is `'scale'`). 
    * `coef0`: Independent term in the kernel function for `'poly'` or `'sigmoid'` kernels. 
* **Support Vectors:**  Access details using:
    * `support_`: Indices of support vectors. 
    * `support_vectors_`:  The actual support vectors. 
    * `n_support_`:  Number of support vectors for each class.
* **`NuSVC` Parameters:**
    * `nu`: Controls the number of support vectors and margin errors.

**`LinearSVC` Parameters:**

* **Flexibility:** Offers more flexibility in penalties and loss functions.
* **Scalability:** Scales better for larger datasets.
* **`dual`:** Decides between primal and dual optimization (use `dual=False` when `n_samples > n_features`).
* **`fit_intercept`:** Calculates the intercept if data is not centered (default is `True`).

**SVM Advantages**

* **High Dimensionality:**  Effective in high-dimensional spaces.
* **Memory Efficiency:** Uses only a subset of training points (support vectors).
* **Versatility:** Allows for different kernel functions.

**SVM Disadvantages**

* **Probability Estimates:**  Does not directly provide probability estimates, requiring expensive cross-validation.
* **Overfitting:** Prone to overfitting if the number of features is significantly larger than the number of samples. 

## Week 9:  Decision Trees

**Decision Trees**

* **Non-parametric:** Do not make assumptions about the data distribution.
* **Supervised Learning:**  Can be used for classification and regression.
* **Rule-based Prediction:**  Predicts labels based on rules inferred from the training data.
* **Algorithm:**  Scikit-learn uses the CART (Classification and Regression Trees) algorithm.

**Decision Tree Parameters**

* **`splitter`:** Strategy for splitting nodes: `'best'` (best split) or `'random'` (random split). 
* **`max_depth`:**  Maximum depth of the tree (default is `None`, expanding until all leaves are pure or contain less than `min_samples_leaf` samples). 
* **`min_samples_split`:**  Minimum number of samples required to split an internal node. 
* **`min_samples_leaf`:**  Minimum number of samples required to be at a leaf node.
* **`criterion`:**  Measures the quality of a split:
    * **Classification:** `'gini'` (Gini impurity) or `'entropy'` (information gain).
    * **Regression:** `'squared_error'`, `'friedman_mse'`, `'absolute_error'`, or `'poisson'`. 

**Tree Visualization**

* **`sklearn.tree.plot_tree`:** Visualize the decision tree. 

**Avoiding Overfitting**

* **Pre-pruning:** Use hyperparameter search (`GridSearchCV`) to find the best set of parameters and limit tree depth.
* **Post-pruning:**  Grow a full tree and then prune it using `cost_complexity_pruning` with parameters like `max_depth` and `min_samples_split`.

**Practical Tips**

* **Dimensionality Reduction:**  Use PCA or Feature Selection to reduce the number of features and prevent overfitting. 
* **Tree Depth:**  Start with a small `max_depth` (e.g., 3) for initial visualization and gradually increase it.
* **Class Imbalance:**  Balance the dataset before training to prevent bias towards dominant classes. 
* **Sample Influence:** Use `min_samples_split` or `min_samples_leaf` to ensure that multiple samples influence each decision, preventing overfitting. 

## Week 10: Bagging and Boosting

**Ensemble Learning**

* **Bagging and Boosting:** Techniques that combine multiple weak learners to create a stronger model.
* **Bias-Variance Trade-off:** Bagging increases bias and reduces variance, while boosting reduces bias and may increase variance.

**Random Forest**

* **Default Base Estimators:**
    * `RandomForestClassifier`: Uses `DecisionTreeClassifier` by default.
    * `RandomForestRegressor`: Uses `DecisionTreeRegressor` by default.
* **Base Estimator Specification:** 
    * `BaggingClassifier` and `BaggingRegressor` allow the specification of any model as the base estimator. 
    * `RandomForestClassifier` and `RandomForestRegressor` do not allow base estimator specification, always using decision trees. 

**Bagging Parameters**

* `max_samples`:  Number of samples to draw for training base estimators:
    * **Float:**  Treated as a fraction of the dataset.
    * `None`: Use all samples.
* `voting`:  Determines how class labels are predicted in `VotingClassifier`:
    * `'hard'`:  Use predicted class labels based on argmax of the summed probabilities.
    * `'soft'`: Use predicted class labels based on majority rule voting. 

**Random Forest vs. Bagging**

* **Feature Subsets:** Random forests force trees to use a subset of features during construction, promoting diversity among trees.

**Boosting**

* **`subsample < 1.0`:** Creates Stochastic Gradient Boosting. 
* **Overfitting:** Gradient Boosting algorithms are robust to overfitting, so larger numbers of estimators often improve performance. 
* **Parallel vs. Sequential:** Bagging is parallel, while boosting is sequential. 

**AdaBoost**

* **`learning_rate`:** Controls the contribution of each estimator:
    * **Higher values:** Increase the contribution of individual estimators.
    * **Trade-off:**  Exists between `learning_rate` and `n_estimators`.

## Week 11:  Clustering

**Clustering Methods**

* **Connectivity-based Clustering (Hierarchical):**  Clusters based on the distance between points.
* **Centroid-based Clustering:** Clusters based on the distance to cluster centroids.
* **Density-based Clustering:**  Clusters based on data point density.

**K-Means**

* **Centroid-based:**  Aims to minimize the inertia (sum of squared distances to cluster centroids). 
* **Sensitivity:** Sensitive to outliers. 
* **Local Optima:** May converge to a local optimum.
* **Implementation:** 
    * `KMeans`: Performs K-Means clustering. 
    * `SpectralClustering`: Clusters based on the eigendecomposition of a similarity matrix.
    * `DBSCAN`:  Density-based spatial clustering of applications with noise. 
    * `AgglomerativeClustering`:  Hierarchical clustering that merges clusters iteratively. 

**Hierarchical Clustering**

* **Divisive Approach:** Top-down approach that starts with all points in one cluster and divides them iteratively.
* **Agglomerative Approach:** Bottom-up approach that starts with individual points as clusters and merges them iteratively.
* **Linkage Methods:** Determine the distance between clusters:
    * **Single Linkage:**  Distance between the nearest points in two clusters.
    * **Complete Linkage:** Distance between the farthest points in two clusters.
    * **Average Linkage:**  Average distance between all pairs of points in two clusters.

**Dendrogram**

* **Hierarchical Representation:** Visualizes the hierarchical relationship between clusters, with vertical lines representing the distance between clusters.

## Week 12:  Multilayer Perceptron

**Multilayer Perceptron (MLP)**

* **`MLPClassifier`:** For multi-class and multi-label classification.
    * Uses Softmax as the output function for multi-class classification.
* **`MLPRegressor`:** For multi-output regression.
* **`hidden_layer_sizes`:**  Specifies the number of hidden layers and neurons per layer using a tuple. 
* **Activation Functions:**
    * `'identity'`:  No activation function (`f(x) = x`).
    * `'logistic'`: Logistic sigmoid function.
    * `'tanh'`:  Hyperbolic tangent function. 
    * `'relu'`:  Rectified Linear Unit (ReLU) function (default).
* **Weight Optimization:**
    * `'lbfgs'`:  Optimizer that does not use mini-batches. 
    * `'sgd'`:  Stochastic gradient descent. 
    * `'adam'`:  Adam optimizer. 
    * `batch_size`:  Size of mini-batches (default is `'auto'`).

**Weight and Bias Inspection**

* **`coefs_`:**  A list of weight matrices, one for each layer (excluding the input layer).
* **`intercepts_`:** A list of bias vectors, one for each layer (excluding the input layer).

**MLP Parameters**

* **`learning_rate`:**  Learning rate schedule: `'constant'`, `'invscaling'`, or `'adaptive'`.
* **`learning_rate_init`:**  Initial learning rate (default is 0.001). 
* **`power_t`:** Exponent for inverse scaling learning rate (default is 0.5). 
* **`max_iter`:**  Maximum number of iterations (epochs, default is 500). 
* **`solver`-specific Parameters:**
    * `learning_rate` and `power_t` are used only for `solver='sgd'`.
    * `learning_rate_init` is used for `solver='sgd'` or `'adam'`. 
    * `shuffle` is used for shuffling samples when `solver='sgd'` or `'adam'`.
    * `momentum` is used for gradient descent update when `solver='sgd'`. 

**MLPRegressor**

* **Activation Function:** Uses no activation function in the output layer.
* **Loss Function:** Uses squared error as the loss function.
* **Parameters:**  The parameters of `MLPRegressor` are the same as `MLPClassifier`.

## End-to-End Machine Learning Project (Wine Quality Prediction)

**Project Objective:** Predict wine quality based on physicochemical characteristics to replace an expensive quality sensor.

**Steps in an ML Project**

1. **Look at the big picture:** 
    * Frame the problem: Define the input, output, and business objective.
    * Select a performance measure: Choose appropriate metrics for evaluation. 
    * Check assumptions:  Review assumptions with domain experts.
2. **Get the data:** Access, understand, and explore the data.
3. **Discover and visualize the data:** Gain insights into features and relationships. 
4. **Prepare the data for Machine Learning algorithms:** 
    * Clean the data: Handle missing values and outliers. 
    * Transform features:  Scale, encode, and engineer features. 
5. **Select a model and train it:** Choose a model based on initial exploration and train it on the preprocessed data.
6. **Fine-tune your model:**  Optimize hyperparameters using techniques like grid search or randomized search. 
7. **Present your solution:** Document, visualize, and communicate findings.
8. **Launch, monitor, and maintain your system:**  Deploy, monitor, and maintain the model in a production environment. 

**Key Considerations for the Wine Quality Prediction Project**

* **Data Source:** Wine quality data is downloaded from the UCI Machine Learning Repository.
* **Data Examination:** The data is examined using:
    * `head()`: To view the first few rows.
    * `info()`: To get a summary of the data (types, missing values, etc.).
    * `describe()`: To calculate descriptive statistics.
* **Feature Significance:** Understand the meaning and relevance of each feature.
* **Data Visualization:** Visualize data using histograms and scatter plots to gain insights into feature distributions and relationships.
* **Test Set Creation:** Split the data into training and test sets to prevent data snooping bias. 
* **Stratified Sampling:** Ensure that the test set is representative of the overall distribution by using stratified sampling based on wine quality. 
* **Data Exploration:**  Analyze the training set using scatter plots and correlation matrices to understand feature relationships.
* **Data Preprocessing:**  
    * Create a copy of the training set for exploration.
    * Handle missing values (if any) using imputation techniques. 
    * Scale features to bring them to a similar range.
    * Transform categorical features using encoding techniques (if any). 
* **Model Selection:** Experiment with different models (e.g., linear regression, decision tree regressor, random forest regressor).
* **Cross-Validation:** Employ cross-validation for robust performance evaluation.
* **Hyperparameter Tuning:** Use grid search or randomized search to optimize model hyperparameters.
* **Model Evaluation on Test Set:** Evaluate the final model's performance on the held-out test set. 
* **Model Interpretation:** Analyze feature importances to understand the model's decision-making process. 

# -*- coding: utf-8 -*-
"""Complete_ML_Course_Notebook.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HgeLq_QgFbyUY9zphvpybVOKu4znu35J

Import necessary libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""### **STEP 1 :** Data Loading"""

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

data = pd.read_csv(data_url,sep =';')

"""### **STEP 2 :** Exploring the Data"""

data.info()

features = data.columns[:-1].values
labels = [data.columns[-1]]

print('Features List: ',features)
print('Labels List: ',labels)

data.describe()

print(data['quality'].value_counts())

"""**Visualization**"""

sns.set()
data.quality.hist()
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.show()

sns.pairplot(data)

"""##### **TRAIN-TEST SPLIT**

Writing a function to split the data into training and test. Make sure to set the seed so that we get the same test set in the next run.
"""

def split_train_test(data, test_ratio):
    #set the random seed
    np.random.seed(42)

    #shuffle the dataset
    shuffled_indices = np.random.permutation(len(data))

    #calculate the size of the test set.
    test_set_size = int(len(data)*test_ratio)

    #split dataset to get trainning and test sets.
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(data, 0.2)

from sklearn.model_selection import train_test_split
train_set, testa_set = train_test_split(data, test_size=0.2, random_state=42)

"""##### **STRATIFIED - SHUFFLE SPLIT**"""

from re import split
from sklearn.model_selection import StratifiedShuffleSplit

split_data= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split_data.split(data, data["quality"]):
    strat_train_set= data.loc[train_index]
    strat_test_set=data.loc[test_index]

strat_dist= strat_test_set['quality'].value_counts() / len(strat_test_set)

overall_dist= data['quality'].value_counts() / len(data)

"""##### **DISTRIBUTION - COMPARISION**"""

dist_comparison = pd.DataFrame({'overall': overall_dist, 'stratified': strat_dist})

dist_comparison['diff(s-o'] = dist_comparison['stratified'] - dist_comparison['overall']

dist_comparison['diff(s-o)_pct'] = 100 * (dist_comparison['diff(s-o']/dist_comparison['overall'])
print(dist_comparison)

#let's contrast this with random sampling
random_dist = test_set['quality'].value_counts()/len(test_set)
random_dist

"""Compare the difference in distribution of stratified and uniform sampling: stratified sampling gives us test distribution closer to the overall distribution than the random sampling.

### **STEP 3 :** Data Visualization

* performed on training set in case of large data set.

* sample examples to form **exploration set**

* Enables to understand features and their relationship among themselves and with output labels.

* In our case we have a small training data and we use it all for data exploration. There is no need to create a separate exploration set.

* It's good idea to create a copy of the training set so that we can freely manipulate it without worrying about any manipulation in the original set.
"""

exploration_set= strat_train_set.copy()

"""#### 1. Scatter plot"""

sns.scatterplot(x='fixed acidity' , y='density' ,hue ='quality', data=exploration_set)
plt.show()

exploration_set.plot(kind='scatter', x='fixed acidity', y='density', alpha=0.5,c='quality',cmap=plt.get_cmap('jet'))
plt.show()

"""#### 2. Standard correlation coefficient between features.

* Ranges between -1 to +1
* Correlation = +1 means Strong positive correlation between features
* Correlation = -1 means Strong negative correlation between features
* Correlation = 0 means No linear correlation between features
* Visualizaiton with **heatmap** only captures linear relationship between features
* For non-linear relationship, we use **rank correlation**

"""

corr_matrix = exploration_set.corr()

"""Checking features that are correlated with the label,i.e quality in our case."""

corr_matrix['quality'].sort_values(ascending=False)

"""Notice that **quality** has strong positive correlation with **alcohol** content [0.48] and strong negative correlation with **volitile acidity** [-0.38]

Visualization of correlation matix using Heatmap :
"""

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, yticklabels=True,cbar=True,annot=True)

"""We can notice:
* The correlation coefficient on diagonal is +1.
* Darker colors represent negative correlations, while fainer colors denote positive correlations. For example :
    * citric acid and fixed acidity have strong positive correlation.
    * pH and fixed acidity have strong negative correlation.

Another option to visualize the relationship between the feature is with **scatter matrix**.

"""

from pandas.plotting import scatter_matrix
attribute_list = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides']

scatter_matrix(exploration_set[attribute_list], figsize=(12,8))
plt.show()

"""For convenience of visualization, we show it for a small number of attributes/features.

Similar analysis can be carried out with combined features-features that are derived from the original features.

> Notes of wisdom
1. Visualization and data exploration do not have to be absolutely thorough.

2. Objective is to get quick insight into features and its relatioship with other features and labels.

3. Exploration is an iterative process: Once we build model and obtain more insights, we can come back to this step.

### **STEP 4 :** Prepare data for ML algorithm

We often need to preprocess the data before using it for model building due to variety of reasons.

* Due to errors in data capture, data may contain outliers or missing values

* Different features may be at different scales.

* The current data distribution is not exactly amenable to learning.

Typical steps in data preprocessing are as follows :

1. Separate features and labels.

1. Handling missing values and outliers

1. Feature scaling to bring all features on the same scale.

1. Applying certain transformations like log, square root on the features.


It is a good practive to make a copy of the data and apply preprocessing on that copy.

This ensures that in case something goes wrong, we will at least have original copy of the data intact.

#### 1. Separation of features and labels
"""

# Copy all features leaving aside the label.

wine_features = strat_train_set.drop('quality', axis=1)
wine_labels = strat_train_set['quality'].copy()

"""#### 2. Data Cleaning

##### 2.A. Handling missing values

 First check if there are any missing values in feature set. One way to find that out is column-wise.
"""

# counts the number of Nan in each column of wine_features
wine_features.isna().sum()

"""In case, we have non-zero numbers in any columns, we have a problem of missing values

* These values are missing due to errors in recording or they do not exist.

* if they are not recorded:
    * use imputation technique to fill up the missing values

    * Drop  the rows containig missing values
    
    
* if they do exists, it is better to keep it as NaN.

Sklearn provides the following methods to drop rows conatining missing values:

        1. dropna()
        
        2. drop()

If provides *SimpleImputer* class for filling up missing values with say, median value.
"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

imputer.fit(wine_features)

"""In case, the features contains non-numeric attributes, they need to be dropped before calling the fit method on imputer object.

Let's check the statistics learnt by the imputer on the training set:
"""

imputer.statistics_

"""Note that these are median values for each feature. We can cross-check it by calculating median on the feature set:"""

wine_features.median()

"""Finally we use the trained imputer to transform the training set such that the missing values are replaced by the medians."""

transf_features = imputer.transform(wine_features)
transf_features.shape

"""This returns a Numpy array and we can convert it to the dataframe if needed:"""

wine_features_transf = pd.DataFrame(transf_features, columns=wine_features.columns)

wine_features_transf.head()

"""##### 2.B. Handling text and categorical attributes

**ORDINAL ENCODING :**

* Converts categories to numbers

* Call `fit_transform()` method on ordinal_encoder object to convert text to numbers.

* The list of categories can be obtained via `categories_` instance variable.

One issue with this representation is that the ML algorithm would assume that the two nearby values are closer than the distinct ones
"""

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

"""**ONE-HOT ENCODING :**

* Converts categorical variables to *binary* variables.

* In other words, we create one binary feature per category - the feature value is 1 when the category is present, else it is 0.

* One feature is 1 (hot) and the rest are 0 (cold).

* The new features are referred to as *dummy features*.
Scikt-Learn provides a `OneHotEncoder` class to convert categorical values into one-hot vectors.

"""

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()

"""* We need to call `fit_transform()` method on OneHotEncoder object.

* The output is a SciPy sparse matrix rather than NumPy array. This enables us to save space when we have a huge nuber of categories.

* In case we want to convert it to dense reprersentation, we can do with `toarray()` method.

* The list of categories can be obtained via `categories_` instance variable

* As we observed that when the number of categories are very large, the one-hot encoding would result in a very large number of features.

* This can be addressed with one of the following approaches:
  
  * Replace with categorical numberical features
  
  * Convert into low-dimensional learnable vectors called `embeddings`

#### 3.Feature Scaling
* Most ML algorithms do not perform well when input features are on very different scales.

* Scaling of target label is generally not required.

##### 3.A. Min-Max Scaling or Normalization
* Scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1.

* We subtract minimum value of a feature from the current value and divide it by the difference between minimum and the maximum value of that feature.

* Scikit-Lean provides `MinMaxScalar` transformer for this.

* One can specify hyperparameter `feature_range` to specify the range of the feature.

##### 3.B. Standardization
* Scaling technique where the values are centered around the mean with a unit standard deviation.

* We subtract mean value of each featurer from the current value and divide it by the standard deviation so that the resulting feature has a unit variance.

* While `normalization`bounds values between 0 and 1, `standardization` does not bound values to a specific range.

* Standardization is less affected by the outliers compared to the normalization.

* Scikit-Learn provides `StandardScalar` transformation for features standardization.

* Note that all these transformers are learnt on the traning data and then applied on the training and test data to transform them.

**Never learn these transformers on the full dataset**

#### Transformation Pipeline
Scikit-Learn provides a Pipeline class to line up transformations in an intended order.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

estimators = [('SimpleImputer', SimpleImputer()), ('StandardScaler', StandardScaler())]

pipe = Pipeline(steps=estimators)
transform_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('standardscaler', StandardScaler())])

wine_features_tr = transform_pipeline.fit_transform(wine_features)

"""Let's understand what is happening here:
* `Pipeline` has a sequence of transformations-missing value imputation followed by standardization.

* Each step is the sequence is define by **name,estimator** pair.

* Each name should be unique and should not contain __ (double underscore)

* The output of one step is passed on the next one in sequence until it reaches the last step.

* Here the pipeline first performs imputation of missing values and its result is passed for standardization.

* The pipeline exposes the same method as the final estimator.
    
    * Here StandardScalar is the last estimator and since it is a transformer, we call `fit_transform()` method on the `Pipleline` object.

#### Transforming Mixed Features

* The real world data has both categorical as well as numerical features and we need to apply different transformations to them.

* Scikit-Learn introduced `ColumnTransformer` class to handle this.
"""

from sklearn.compose import ColumnTransformer

"""* The `ColumnTransformer` applies each transformation to the appropriate columns and then concatenates the outputs along the columns.

* Note that all transformers must return the same number of rows.
* The numeric transformers return **dense** matrix while the categorical ones return **sparse** matrix.

* The ColumnTransformer automatically determines the type of the output based on the density of resulting matrix.

### **STEP 5 :** Select and Train ML model

It is a good practice to build a quick baseline model on the preprocessed data and get an idea about model performance.
"""

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(wine_features_tr, wine_labels)

"""* Now that we have a working model of a regression, let's evaluate performance of the model on training as well as test sets.

* For regression models, we use mean squared error as an evaluation measure.
"""

from sklearn.metrics import mean_squared_error

quality_pred = lin_reg.predict(wine_features_tr)
mean_squared_error(wine_labels, quality_pred)

"""Let's evaluate performance on the test set.

We need to first apply transformation on the test set and then apply the model prediction function.

"""

# copy all features leaving aside the label.
wine_features_test = strat_test_set.drop("quality", axis=1)

#copy the label list
wine_labels_test = strat_test_set['quality'].copy()

#apply transformations
wine_features_test_tr = transform_pipeline.fit_transform(wine_features_test)

#call predict function and calculate MSE.
quality_test_pred = lin_reg.predict(wine_features_test_tr)
mean_squared_error(wine_labels_test, quality_test_pred)

"""Let's visualize the error between the actual and predicted values."""

plt.scatter(wine_labels_test, quality_test_pred)
plt.plot(wine_labels_test, wine_labels_test, 'b-')
plt.xlabel('Actual quality')
plt.ylabel('Predicted quality')
plt.show()

"""Let's try another model: DecisionTreeRegressor"""

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(wine_features_tr, wine_labels)

quality_pred = tree_reg.predict(wine_features_tr)
print('Training Error :', mean_squared_error(wine_labels, quality_pred))

quality_test_pred = tree_reg.predict(wine_features_test_tr)
print('Test Error : ',mean_squared_error(wine_labels_test, quality_test_pred))

"""Note that the training error is 0, while the test error is 0.58. This is an example of an overfitted model."""

plt.scatter(wine_labels_test, quality_test_pred)
plt.plot(wine_labels_test, wine_labels_test, 'r-')
plt.xlabel('Actual quality')
plt.ylabel('Predicted quality')
plt.show()

"""##### **Cross-Validation (CV)**

* Cross validation provides a separate MSE for each validation set, which we can use to get a mean estimation of MSE as well as the standard deviation, which helps us to determine how precise is the estimate.

* The additional cost we pay in cross validation is additional training runs, which may be too expensive in certain cases.
"""

from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores : \n", scores)
    print()
    print("Mean : ", scores.mean())
    print("Standard deviation : ", scores.std())

"""**Linear Regression CV**"""

scores = cross_val_score(lin_reg, wine_features_tr,
                        wine_labels, scoring="neg_mean_squared_error", cv=10)

lin_reg_mse_scores = -scores
display_scores(lin_reg_mse_scores)

"""**DecisionTreeRegressor CV**"""

scores = cross_val_score(tree_reg, wine_features_tr,
                        wine_labels, scoring="neg_mean_squared_error", cv=10)

tree_mse_scores = -scores
display_scores(tree_mse_scores)

"""Upon comparision of scores of the two models, we can see that the LinearRegressor has better MSE and more precise estimation compared to DecisionTree.

**RandomForest CV**

* Random forest model builds multiple decision trees on randomly selected features and then average their predictions.

* Building a model on top of other model is called * ensemble learning *
Which is often used to improve performance of ML models.
"""

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(wine_features_tr, wine_labels)

scores = cross_val_score(rf_reg, wine_features_tr,
                         wine_labels, scoring="neg_mean_squared_error", cv=10)

rf_mse_scores = -scores
display_scores(rf_mse_scores)

quality_pred = rf_reg.predict(wine_features_tr)
print('Training Error :', mean_squared_error(wine_labels, quality_pred))

quality_test_pred = rf_reg.predict(wine_features_test_tr)
print('Test Error : ',mean_squared_error(wine_labels_test, quality_test_pred))

"""* Random forest looks more promising than the other two models.

* It's good practice to build a few such models quickly without tuning their hyperparameters and shortlist a few promising models among them.

* Also save the methods to the disk in Python `pickle` format.

### **STEP 6 :** FineTune the model

* Usually there are a number of hyperparameters in the model, which are set
manually.

* Tuning these hyperparameters lead to better accuracy of ML models.
* Finding the best combination of hyperparameters is a search problem in the
space of hyperparameters, which is huge.

#### **Grid Search**
"""

from sklearn.model_selection import GridSearchCV

"""* We need to specify a list of hyperparameters along with the range of values to try.

* It automatically evaluates all possible combinations of hyperparameter values using cross-validation.

* For example, there are number of hyperparameters in RandomForest regression
such as:
  * Number of estimators

  * Maximum number of features
"""

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

"""Here the parameter grid contains two combinations:
1. The first combination contains n_estimators with 3 values and max_features with 4 values.

2. The second combination has an additional bootstrap parameter, which is set to
False. Note that it was set to its default value, which is True, in the first grid.

Let's compute the total combinations evaluated here:
1. The first one results in 3 × 4 = 12 combinations.

2. The second one has 2 values of n_estimators and 3 values of max_features, thus resulting 2 × 3 = 6 in total of values.

The total number of combinations evaluated by the parameter grid 12 + 6 = 18.

Let's create an object of GridSearchCV:
"""

grid_search = GridSearchCV(rf_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)

"""In this case, we set cv=5 i.e. using 5 fold cross validation for training the model.

We need to train the model for 18 parameter combinations and each combination
would be trained 5 times as we are using cross-validation here.

The total model training runs = 18 × 5 = 90
"""

grid_search.fit(wine_features_tr, wine_labels)

"""The best parameter combination can be obtained as follows:"""

grid_search.best_params_

"""Let's find out the error at different parameter settings:"""

cv_res = grid_search.cv_results_

for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(-mean_score, params)

"""As you can notice the lowest MSE is obtained for the best parameter combination. Let's obtain the best estimator as follows :"""

grid_search.best_estimator_

"""**NOTE** : GridSearchCV is initialized with `refit=True` option, which retrians the best estimator on the full training set. This is likely to lead us to a better model as it is trained on a larger dataset.

#### **Randomized Search**

* When we have a large hyperparameter space, it is desirable to try RandomizedSearchCV.

* RandomizedSearchCV is a wrapper around the RandomizedSearchCV class in the scikit-learn library.

* It selects a random value for each hyperparameter at the start of each iteration and repeats the process for the given number of random combinations.

* It enables us to search hyperparameter space with appropriate budget control.
"""

from sklearn.model_selection import RandomizedSearchCV

"""#### **Analysis of best model and its errors**
Analysis of the model provides useful insights about features. Let's obtain the feature importance as learnt by the model.
"""

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

sorted(zip(feature_importances ,features) ,reverse=True)

"""* Based on this information, we may drop features that are not so important.

* It is also useful to analyze the errors in prediction and understand its causes and fix them.

#### **Evaluation on test set**
Now that we have a reasonable model, we evaluate its performance on the test set. The following steps are involved in the process :

1. Transform the best features
"""

# copy all features leaving aside the label
wine_features_test = strat_test_set.drop('quality', axis=1)

# copy the label list
wine_labels_test = strat_test_set['quality'].copy()

# apply transformations
wine_features_test_tr = transform_pipeline.fit_transform(wine_features_test)

"""2. Use the predict method with the trained model and the test set."""

quality_test_pred = grid_search.best_estimator_.predict(wine_features_test_tr)

"""3. Compare the predicted labels with the actual ones and report the evaluation metrics."""

print('Test Error : ',mean_squared_error(wine_labels_test, quality_test_pred))

"""4. It's a good idea to get 95% confidence interval of the evaluation metric. It can be obtained by the following code:"""

from scipy import stats

confidence = 0.95
squared_errors = (quality_test_pred - wine_labels_test) **2

stats.t.interval (confidence, len(squared_errors)-1, loc=squared_errors.mean(),scale=stats.sem(squared_errors))

"""### **STEP 7 :** Present your Solution

Once we have satisfactory model based on its performance on the test set, we reach the prelaunch phase.

Before launch :
  1. We need to present our solution that highlights learnings, assumptions and systems limitation.

  2. Document everything, create clear visualizations and present the model.
  
  3. In case, the model doesn't work better that the experts, it may still be a good idea to launch it and free up bandwidths of human experts.

### **STEP 8:** Launch, Monitor and Maintain your system
* Launch :
    * Plug in input sources &

    * Write test cases

* Monitoring :
    * System outages

    * Degradation of model performance

    * Sampling predictions for human evaluation

    * Regular assessment of data quality, which is critical for model performance.

* Maintenance :

    * Train model regularly every fixed interval with fresh data.

    * Production roll out of the model.

### **SUMMARY**
In this module, we studied steps involved in end-to-end machine learning project with an example of prediction of wine quality.

Import necessary libraries
"""

from IPython.display import display, Math, Latex

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

"""## Data Cleaning

### **1. Data Imputation**

* Many machine learning algorithms need full feature matrix and they may not work in presence of missing data.
* Data imputation identifies **missing values** in each feature of the dataset and **replaces** them with an **appropriate values** based on a **fixed strategy** such as :

    * **mean** or **median** or **mode** of that feature.
    
    * **use specified constant** value. Sklearn library provides `sklearn.impute.SimpleImputer` class for this purpose.
"""

from sklearn.impute import SimpleImputer

"""Some of its important parameters:
  * *missing_values:* could be `int`,`str`,`np.nan` or `None`. Default is `np.nan`.

  * *strategy*: string, default is 'mean'. One the following strategies can be used:

    * `mean`- missing values are  replaced using the **mean** along each column.

    * `median`-missing values are replaced using the **median** along each column.

    * `most_frequent`-missing values are replaced using the **most_frequent** along each column.

    * `constant` - missing values are replaced with value specified in `fill_value` argument.

    * `add_indicator` - a boolean parameter that when set to `True` returns **missing value indicators** in `indicator_` member value.

**NOTE :**
   * `mean` and `median` strategies can only be used with numeric data.

   * `most_frequent` and `constant` strategies can be used with strings or numberic data.

#### **Data imputation on real world dataset.**
  Let's perform data imputation on real world dataset. We will be using <https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleveland.data> for this purpose. We will load this dataset from csv file.
"""

cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

heart_data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header=None, names=cols)

heart_data.head()

"""The dataset has the following features :
* Age (in years)

* Sex (1 = male; 0 = female)

* cp - chest pain type

* trestbps - resting blood pressure (anything above 130-140 is typically cause for concern)

* fbs - fasting blood sugar (>120 mg/dl) (1 = true; 0 = false)

* restecg - resting electrocardiographic results
    * normal=0
    * 1 = having ST-T wave abnormality;

    * 2 = showing probable or definite left ventricular hypertropy by Estes' criteria

* thalch - maximum heart rate achieved

* exang - excercise induced angina
    * 1 = yes

    * 0 = no

* oldpeak - depression induced by excercise relative to rest

* slope - slope of the peak excercise ST segment
    * 1 = unsloping;

    * 2 = flat value;

    * 3 = downsloping

* ca - number of major vessels (0-3) colored by fluroscopy

* thal - (3 = normal; 6 =fixed defect; 7 = reversable defect)

* num - diagnosis of heart disease (angiographic disease status)
  * 0 < 50% diameter narrowing;
  
  * 1: . 50% diameter narrowing

#### STEP 1 : Check if dataset has missing values

* This can be checked via dataset description or by check number of `nan` or `np.null` in the dataframe. However such check can be performed only for numerical features.

* For non-numberical features, we can list their unique values and check if there are values like `?`.
"""

heart_data.info()

"""Let's check if there are any missing values in numerical columns-here we have checked it for all columns in the dataframe.

"""

heart_data.isnull().sum()

"""There are two non-numerical features : `ca` and `thal` so list their unique values:"""

print('Unique values in ca:', heart_data.ca.unique())
print('Unique values in thal:', heart_data.thal.unique())

"""Both of them contain `?`, which is a missing value. Let's count the number of missing values."""

print('Number of missing vlaue in ca:',
      heart_data.loc[heart_data.ca == '?', 'ca'].count())
print('Number of missing vlaue in thal:',
      heart_data.loc[heart_data.thal == '?', 'thal'].count())

"""#### STEP 2 : Replace ? with `NaN`"""

heart_data.replace('?' ,np.nan ,inplace=True)

"""#### STEP 3 : Fill the mising values with sklearn missing value imputation utilities.

Here we use `SimpleImputer` with `mean` strategy. We will try two variations :

a. **add_indicator=False** : Default choice that only imputes missing values.
"""

imputer =  SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(heart_data)
heart_data_imputed = imputer.transform(heart_data)

print(heart_data.shape)
print(heart_data_imputed.shape)

"""b. **add_indicator=True** : Adds additional column for each column containing missing values.

In our case, this adds two columns one for `ca` and other for `thal`. It indicates if the sample has a missing value.

Now the number of extra column added will be 1 per missing columns that contains the boolean value i.e True/False to indicate that earlier some values were missing. It is just like a pointer for missing value update.
"""

imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean', add_indicator=True)

imputer = imputer.fit(heart_data)
heart_data_imputed_with_indicator = imputer.transform(heart_data)

print(heart_data.shape)
print(heart_data_imputed_with_indicator.shape)

"""### **2.Feature Scaling**
Feature Scaling **transform feature values** such that **all the features are on the same scale**.

When we use feature matrix with all features on the same scale, it provides us certain advantages as listed below:

* __Enables Faster Convergence__ in iterative optimization algorithms like gradien descent and its variants.

* The performance of ML algorithms such as SVM, K-NN and K-means etc. that compute euclidean distance among input samples gets impacted if the features are not scaled.

`Tree` based Ml algorithms are not affected by feature-scaling. In other words, feature scaling is not required for `tree` based ML algorithms.

Feature scaling can be performed with the following methods:
* Standardization

* Normalization

* MaxAbsScaler

#### **Feature Scaling on real world dataset.**
Let's demonstrate feature scaling on a real world dataset. For this purpose we will be using https://archive.ics.uci.edu/ml/datasets/Abalone .

We will use different scaling utilities in the sklearn library.
"""

cols = ['sex', 'Length', 'Diameter', 'Height','Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

abalone_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',header=None, names=cols)

"""Abalone dataset has the following features :

* Sex -nominal (M, F, and I (infant))

* Length (mm - Longest shell measurement)

* Diameter (mm - perpendicular to lenght)

* Height (mm - with meat in shell)

* Whole weight (grams -whole abalone)

* Shucked weight (grams - whole abalone)

* Viscera weight (grams - gut weight (after bleeding))

* Shell weight (grams - after being dried)

* Rings (target - age in years)

#### **STEP 1 : Examine the dataset**
"""

abalone_data.info()

"""#### **STEP 1 [Optional] : Convert non-numerical attributes to numerical ones.**

In this dataset, `sex` is a non-numeric column in this dataset. Let's examine it and see if we can convert it to numeric representation.
"""

abalone_data.sex.unique()

#Assign numerical values to sex.
abalone_data = abalone_data.replace({"sex": {'M': 1, 'F': 2, 'I': 3}})
abalone_data.info()

"""#### **STEP 2 : Separate labels from features.**"""

y = abalone_data.pop('Rings')

print('The DataFrame object after deleting the column : \n')
abalone_data.info()

"""#### **STEP 3 : Examine feature scales**

##### 3A. **Statistical method**

Check the scales of different feature with `describe` method of dataframe.
"""

abalone_data.describe().T

"""**Note :**
* There are 4177 examples or rows in this dataset.
* The mean and standard deviation of features are quite different from one another.

##### 3B. **Visualization of feature distributions**

This method includes :
* Histogram

* Kernel density estimation (KDE) plot

* Boxplot

* Violin plot

1. **Feature Histogram**

We will have separate and combined histogram plots to check if the feature are indeed on different scales.
"""

plt.hist(np.array(abalone_data['Length']))
plt.show()

plt.hist(np.array(abalone_data['Shucked weight']))
plt.show()

for i in abalone_data.columns:
    plt.hist(np.array(abalone_data[i]))

"""Observe that the features have different distributions and scales.

2. **KDE plot**

* Alternatively, we can generate **Kernel Density Estimate** plot using Gaussian Kernels.

* In statistics, kernel density function (KDE) is a non-parametric way to estimate the probability density function (PDF) of a random variable.
This function uses Gaussian Kernels and includes automatic bandwidth determination.
"""

ax = abalone_data.plot.kde()

"""Observe that the features have different distributions and scales.

3. **Boxplot**

* A **box plot** (or box-and-whisker plot) shows the **distribution of quantitative** in a way that facilitates comparisions between variables or across levels of a categorical variables.

* The box shows the **quartiles** of the dataset while the **whiskers** extend to show the rest of the distribution, except for points that are determined to be 'outliers' using a method that is a function of the inter-quartile range.
"""

ax = sns.boxplot(data=abalone_data, orient='h', palette='Set2')

"""#### **STEP 4 : Scaling the features**

##### 4A. **Normalization**

The features are normalized such that their range lies between $[0,1] or [-1,1]$. There are two ways to achieve this :

* `MaxAbsScaler` transform features in range $[-1,1]$
* `MinMaxScaler` transforms features in range $[0,1]$

**a. MaxAbsScaler**

It transforms the original features vector $ \textbf x$ into new feature vector $\textbf x^{'} $ so that all values fall within range [-1,1] and the range of each feature is the same.

\begin{equation}
\textbf x^{'} = \frac{\textbf x}{\text {MaxAbsoluteValue}}
\end{equation}

where :

\begin{equation}
 \text {MaxAbsolutevalue}= \text {max}(\textbf x.max,|\textbf x.min|)
 \end{equation}
"""

x = np.array([4, 2, 5, -2, -100]).reshape(-1, 1)
print(x)

from sklearn.preprocessing import MaxAbsScaler
max_abs_scaler = MaxAbsScaler()

x_mas = max_abs_scaler.fit_transform(x)
print(x_mas)

"""**b. MinMaxScaler**

Normalization is a procedure in which the feature values are scaled such that they range between 0 and 1. This technique is also called **Min-Max Scaling**.

It is performed with the following formula:
\begin{equation}
\mathbf X_{new} = \frac{X_{old} - X_{min} }{\mathbf X_{max} - X_{min}}
\end{equation}

where :
* $X_{old}$ is the old value of a data point, which is rescaled to $ X_{new}$.

* $X_{min}$ is minimum value of feature $X$

* $X_{max}$, is maximum value of feature $X$.

Normalization can be achieved by `MinMaxScaler` from sklearn library.
"""

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

X_normalized = min_max_scaler.fit_transform(abalone_data)
X_normalized[:5]

"""Let's look at the mean and standard deviation (SD) of each feature:"""

X_normalized.mean(axis=0)

X_normalized.std(axis=0)

"""The means of SDs of different features are now comparable. We can confirm this again through visualization as before:"""

cols = ['sex', 'Length', 'Diameter', 'Height', 'Whole weight',
        'Shucked weight', 'Viscera weight', 'Shell weight']

X_normalized = pd.DataFrame(X_normalized, columns=cols)

sns.histplot(data=X_normalized)

sns.kdeplot(data=X_normalized)

"""##### 4B. **Standardization**

* Standardization is another feature scaling technique that results into (close to ) zero mean and unit standard deviation of a feature's values.

* Formula for standardization:
\begin{equation}
X_{new} = \frac{X_{old}-\mu}{\sigma}
\end{equation}

where, $\mu$  and $\sigma$ respectively are the mean and standard deviation of the feature values.

* Standardization can be achieved by `StandardScaler` from sklearn library.
"""

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X_standardized = ss.fit_transform(abalone_data)
X_standardized[:5]

X_standardized.mean(axis=0)

X_standardized.std(axis=0)

"""The means of different features are now comparable with SD = 1"""

# sns.histplot(data=X_standardized)

in_cols = cols[:len(cols)-1]
plt.figure(figsize=(12, 8))

data = pd.DataFrame(X_standardized, columns=cols)

for colname in abalone_data:
    plt.hist(data[colname].values, alpha=0.4)

plt.legend(in_cols, fontsize=18, loc='upper right', frameon=True)
plt.title('Distribution of features across samples after standardization')
plt.xlabel('Range', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()

data.plot.kde()
plt.show()

ax = sns.boxplot(data=data, orient='h', palette='Set2')

ax = sns.violinplot(data=data, orient='h', palette='Set2')

"""Import necessary libraries"""

from IPython.display import display, Math, Latex

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

"""## Feature Transformations

### **1.Polynomial Features**

* Generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree.

* For example, if an input sample is two dimensional and of the form  $[a,b]$ , the degree-2 polynomial features are  $[1,a,a^2,b,b^2 ,ab]$ .

* `sklearn.preprocessing.PolynomialFeatures` enables us to perform polynomial transformation of desired degree.

Let's demonstrate it with wine quality dataset :
"""

from sklearn.preprocessing import PolynomialFeatures

wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep=';')

wine_data_copy = wine_data.copy()
wine_data = wine_data.drop(['quality'] ,axis=1)

print('Number of features before transformation = ', wine_data.shape)

poly = PolynomialFeatures(degree=2)
wine_data_poly = poly.fit_transform(wine_data)
print('Number of features after transformation = ', wine_data_poly.shape)

"""Note that after transformation, we have 78 features. Let's list out these features:"""

poly.get_feature_names_out()

"""Observe that :
* Some features have ^2 suffix - these are degree-2 features of input features. For example, `sulphates^2` is the square of `sulphates` features.

* Some features are combination of names of the original feature names. For example, `total sulfur dioxide pH` is a combinationn of two features `total sulfur dioxide` and `pH`.

### **2.Discretization**

**Discretization** (otherwise known as **quantization or binning**) provides a way to partition continuous features into discrete values.


* Certain datasets with continuous features may benefit from discretization, because it can transform the datasets of continuous attributes to one with only nominal attributes.

* One-hot encoded discretized features can make a model more expressive, while maintaining interpretability.

* For instance, pre-processing with discretizer can introduce non-linearity to linear models.

`KBinsDiscretizer` discretizes features into `k-bins`.
"""

from sklearn.preprocessing import KBinsDiscretizer

wine_data = wine_data_copy.copy()

#transform the dataset with KBinDiscretizer
kbd = KBinsDiscretizer(n_bins=10, encode='onehot')

X = np.array(wine_data['chlorides']).reshape(-1, 1)
X_binned = kbd.fit_transform(X)

X_binned

X_binned.toarray()[:5]

"""### **3.Handling Categorical Features**

We need to convert the categorical features into numeric features. It includes :
1. Ordinal encoding

2. One hot encoding

3. Label encoding

4. MultiLabel Binarizer

5. Using dummy variables

[Iris dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) has the following features:

1. sepal length (in cm)

2. sepal width (in cm)

3. petal length (in cm)

4. petal width (in cm)

class : Iris Setosa, Iris Versicolour, Iris Virginica
"""

cols = ['sepal length', 'sepal width', 'petal width', 'label']

iris_data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=cols)

iris_data.head()

"""**1. Ordinal Encoding**

* Categorical features are those that contain categories or groups such as education level, state etc as their data.

* These are non-numerical features and need to be converted into appropriate from before they feeding them for training an ML model.

* Our intuitive way of handling them could be to assign them a numerical value.

* As an example, take state as a feature with 'Punjab', Rajasthan, and Haryana as the possible values. We might consider assigning number to these values as follows:

    Old feature | New feature
    ------------|-------------
    Punjab      |     1
    Rajasthan   |     2
    Haryana     |     3



However, this approach assigns some ordering to the labels, i.e. states, thus representing that Haryana is thrice Punjab and Rajasthan is twice Pubjab, these relationships do not exist in the data, thus providing wrong information to the ML model.

Let's demonstrate this concept with `Iris` dataset.
"""

from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder()

iris_labels = np.array(iris_data['label'])

iris_labels_transformed = ordinal.fit_transform(iris_labels.reshape(-1, 1))
print(np.unique(iris_labels_transformed))

print()
print('First 5 labels in ordinal encoded form are : \n',
      iris_labels_transformed[:5])

"""**2. One-hot Encoding**

* This approach consists of creating an addtional feature for each label present in categorical feature(i.e. the number of different states here) and putting a 1 or 0 for these new features depending on the categorical feature's value. That is,


Old feature  |   New feature_1 (punjab) | New feature_2 (Rajasthan) | New feature_3(Haryana)
--------------|---------------------------|---------------------------|------------------------
Punjab        |          1                |           0               |         0
Rajasthan     |          0                |           1               |         0
Haryana       |          0                |           1               |         0


* It may be implemented using `OneHotEncoder` class from sklearn.preprocessing module.

The `label` in the iris dataset is a categorical attribute.
"""

iris_data.label.unique()

"""There are three class labels. Let's convert them to one hot vectors."""

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()

print('Shape of y before encoding : ', iris_data.label.shape)


'''
Passing 1d arrays as data to onehotcoder is deprecated in version, hence reshape to (-1,1) to have two dimensions.

Input of onehotencoder fit_transform must not be 1-rank array
'''

iris_labels = one_hot_encoder.fit_transform(iris_data.label.values.reshape(-1, 1))

# y.reshape(-1,1) is a 450 x 1 sparse matrix of type <class numpy.float64>

# y is a 150 x 3 sparse matrix of type <class numpy.float64> with 150 stored
# elements in Coordinate format.

print('Shape of y after encoding : ', iris_labels.shape)

# since output is sparse use toarray() to expand it.
print()
print('First 5 labels in one-hot vector form are : \n',iris_labels.toarray()[:5])

"""**3. Label Encoding**

Another option is to use `LabelEncoder` for transforming categorical features into integer codes.
"""

from sklearn.preprocessing import LabelEncoder

iris_labels = np.array(iris_data['label'])

label = LabelEncoder()
label_integer = label.fit_transform(iris_labels)

print('Labels in integer form are : \n', label_integer)

"""**4. MultiLabel Binarizer**

* Encodes categorical features with value 0 to $ k-1$ where $k$ is number of classes.

* As the name suggests for case where output are multilabels there we use each unique label as column and assign 0 or 1 depending upon in the dataset that value is present or not.

Movie genres is best example to understand.
"""

movie_genres = [
    {'action', 'comedy'},
    {'comedy'},
    {'action', 'thriller'},
    {'science-fiction', 'action', 'thriller'}
]

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
mlb.fit_transform(movie_genres)

"""**5. Using Dummy variables**

Use `get_dummies` to create a one-hot encoding for each unique categorical value in the 'class' column
"""

iris_data_onehot = pd.get_dummies(
    iris_data, columns=['label'], prefix=['one_hot'])

iris_data_onehot.head()

"""### **4.Custom Transformers**

Enables conversion of an existing Python function into a transformer to assist in data cleaning or processing.

Useful when:
1. The dataset consists of *hetereogeneous data types* (e.g. raster images and text captions)

2. The dataset is stored in a `pandas.DataFrame` and different columns require *different processing pipelines.*

3. We need stateless transformations such as taking the log of frequencies, custom scaling, etc.

We can implement a transformer from an arbitary function with `Function Transformer`.
"""

from sklearn.preprocessing import FunctionTransformer

"""For example, let us build a tranformer that applies a log transformation to features.

For this demonstration, we will be using a  [wine quality dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) from UCI machine learning repository.

It has got the following attributes:


1. fixed acidity

2. volatile acidity

3. citric acid

4. residual sugar

5. chlorides

6. free sulfur dioxide

7. total sulfur dioxide

8. density

9. pH

10. sulphates

11. alcohol

12. quality (output: score between 0 and 10)
"""

wine_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

wine_data.describe().T

"""Let's use `np.log1p` which returns natural logarithm of(1 + the feature value).

"""

transformer = FunctionTransformer(np.log1p, validate=True)

wine_data_transformed = transformer.transform(np.array(wine_data))
pd.DataFrame(wine_data_transformed, columns=wine_data.columns).describe().T

"""Simple Examples :"""

transformer = FunctionTransformer(np.log1p)

X = np.array([[0, 9], [7, 8]])
transformer.transform(X)

transformer = FunctionTransformer(np.exp2)

X = np.array([[1,3], [2,4]])
transformer.transform(X)

"""### **5.Composite Transformers**

* It applies a set of transformers to columns of an array or `pandas.DataFrame`, concatenates the transformed outputs from different transformers into a single matrix.

**5.A. Apply Transformation to diverse features**

* It is useful for transforming heterogeneous data by applying different transformers to separate subsets of features.

* It combines different feature selection mechanism and transformation into a single transformer object.

* It is a list of tuples.

* In the tuple, first we mention the reference name, second the method and third the column on which we want to apply column transformer.
"""

X = [
    [20.0,'male'],
    [11.2,'female'],
    [15.6,'female'],
    [13.0,'male'],
    [18.6, 'male'],
    [16.4,'female']
]

X = np.array(X)
print(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler ,OneHotEncoder

col_trans = ColumnTransformer([
    ('scaler' ,MaxAbsScaler() ,[0]),
    ('pass' ,'passthrough' ,[0]) ,
    ('encoder' ,OneHotEncoder() ,[1])
])

col_trans.fit_transform(X)

"""**5.B. TransformedTargetRegressor**

Transforms the target variable `y` before fitting a regression model.

* The predicted values are mapped back to the original space via an inverse transform.

* It takes **regressor** and **transformer** as arguments to be applied to the target variable.
"""

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)

# select a subset of data
X, y = X[:2000, :], y[:2000]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# transformer to scale the data
transformer = MinMaxScaler()

# first regressor - based on the original labels.
regressor = LinearRegression()

# second regressor - based on transformed labels.
ttr = TransformedTargetRegressor(regressor=regressor, transformer=transformer)

regressor.fit(X_train, y_train)
print('R2 score of raw_label regression: {0:.4f}'.format(
    regressor.score(X_test, y_test)))


ttr.fit(X_train, y_train)
print('R2 score of transformed label regression: {0:.4f}'.format(
    ttr.score(X_test, y_test)))

"""Import necessary libraries"""

from IPython.display import display, Math, Latex

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

"""## Feature Selection

`sklearn.feature_selection` module has useful APIs to select features/reduce dimensionality, either to improve estimators' accuracy score or to boost their performance on very high-dimensional datasets.

Top reasons to use feature selection are:


* It enables the machine learning algorithm to train faster.

* It reduces the complexity of a model and makes it easier to interpret.

* It improves the accuracy of a model if the right subset is chosen.

* It reduces overfitting.

#### **1.FILTER-BASED METHODS**

##### 1.A. Variance Threshold

* This transformer helps to keep only high variance features by providing a certain threshold.

* Features with  variance greater or equal to threshold value are kept rest are removed.

* By default, it removes any feature with same value i.e. 0 variance.
"""

data = [{'age': 4, 'height': 96.0},
        {'age': 1, 'height': 73.9},
        {'age': 3,  'height': 88.9},
        {'age': 2, 'height': 81.6}
        ]

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

data_transformed = dv.fit_transform(data)
np.var(data_transformed, axis=0)

from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=5)

data_new = vt.fit_transform(data_transformed)
data_new

"""As you may observe from output of above cell, the transformer has removed the age feature because its variance is below the threshold.

##### 1.B. SelectKBest

It selects k-highest scoring features based on a function and removes the rest of the features.

Let's take an example of California Housing Dataset.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression


X_california, y_california = fetch_california_housing(return_X_y=True)
X, y = X_california[:2000], y_california[:2000]

"""Let's select 3 most important features, since it is a regression problem, we can use only `mutual_info_regression` of `f_regression` scoring functions only."""

# mutual_info_regression is scoring method for linear regression method

skb = SelectKBest(mutual_info_regression, k=3)
X_new = skb.fit_transform(X, y)

print(f'Shape of feature-matrix before feature selection : {X.shape}')
print(f'Shape of feature-matrix after feature selection : {X_new.shape}')

"""##### 1.C. SelectPercentile

* This is very similar to `SelectKBest` from previous section, the only difference is, it selects top `percentile` of all features and drops the rest of features.

* Similar to `SelecKBest`, it also uses a scoring function to decide the importance of features.

Let's use the california housing price dataset for this API.
"""

from sklearn.feature_selection import SelectPercentile

sp = SelectPercentile(mutual_info_regression, percentile=30)
X_new = sp.fit_transform(X, y)

print(f'Shape of feature-matrix before feature selection : {X.shape}')
print(f'Shape of feature-matrix after feature selection : {X_new.shape}')

"""As you can see from above output, the transformed data now only has top 30 percentile of features, i.e only 3 out of 8 features."""

skb.get_feature_names_out()

"""##### 1.D. GenericUnivariateSelect

* It applies  univariate feature selection with a certain strategy, which is passed to the API via `mode` parameter.

* The `mode` can take one of the following values :

    * `percentile` (top percentage)

    * `k_best` (top k)

    * `fpr` (false positive rate)

    * `fdr` (false discovery rate)

    * `fwe` (family wise error rate)

* If we want to accomplish the same objective as `SelectKBest`, we can use following code:
"""

from sklearn.feature_selection import GenericUnivariateSelect

gus = GenericUnivariateSelect(mutual_info_regression, mode='k_best', param = 3)
X_new = gus.fit_transform(X,y)

print(f'Shape of feature-matrix before feature selection : {X.shape}')
print(f'Shape of feature-matrix after feature selection : {X_new.shape}')

"""#### **2.WRAPPER-BASED METHODS**

##### 2.A. Recursive Feature Elimination (RFE)

* STEP 1 : Fits the model

* STEP 2 : Ranks the features, afterwards it removes one or more features (depending upn `step` parameter)

These two steps are repeated until desired number of features are selected.
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

estimator = LinearRegression()

selector = RFE(estimator, n_features_to_select=3, step=3)
selector = selector.fit(X, y)

# support_ attribute is a boolean array marking which features are selected
print(selector.support_)

# rank of each feature
# if it's value is '1', then it is selected
# features with rank 2 and onwards are ranked least.
print(f'Rank of each feature is : {selector.ranking_}')

X_new = selector.transform(X)

print(f'Shape of feature-matrix before feature selection : {X.shape}')
print(f'Shape of feature-matrix after feature selection : {X_new.shape}')

"""##### 2.B. SelectFromModel

* Selects desired number of important features (as specified with `max_features` parameter) above certain threshold of feature importance as obtained from the trained estimator.

* The feature importance is obtained via `coef_`, `feature_importance_` or an `importance_getter` callable from the trained estimator.

* The feature importance threshold can be specified either numerically or through string argument based on built-in heuristics such as `mean`, `median` and `float` multiples of these like `0.1*mean`.
"""

from sklearn.feature_selection import SelectFromModel

estimator = LinearRegression()
estimator.fit(X, y)

print(f'Coefficients of features :\n {estimator.coef_}')
print()
print(f'Intercept of features : {estimator.intercept_}')
print()
print(f'Indices of top {3} features : {np.argsort(estimator.coef_)[-3:]}')

t = np.argsort(np.abs(estimator.coef_))[-3:]

model = SelectFromModel(estimator, max_features=3, prefit=True)
X_new = model.transform(X)

print(f'Shape of feature-matrix before feature selection : {X.shape}')
print(f'Shape of feature-matrix after feature selection : {X_new.shape}')

"""##### 2.C. SequentialFeatureSelection

It performs feature selection by selecting or deselecting features one by one in a greedy manner.
"""

from sklearn.feature_selection import SequentialFeatureSelector

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# estimator = LinearRegression()
# sfs = SequentialFeatureSelector(estimator, n_features_to_select=3)
# 
# sfs.fit_transform(X, y)
# print(sfs.get_support())

"""The features corresponding to True in the output of sfs.get_support() are selected. In this case,features 1, 6 and 7 are selected."""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# estimator = LinearRegression()
# sfs = SequentialFeatureSelector(
#     estimator, n_features_to_select=3, direction='backward')
# 
# sfs.fit_transform(X, y)
# print(sfs.get_support())

"""A couple of observations:
* Both `forward` and `backward` selection methods select the same featurers.

* The `backward` selection method takes longer than `forward` selection method.

From above examples, we can observe that depending upon number of features, `SFS` can accomplish feature selection in different periods forwards and backwards.

Import necessary libraries
"""

from IPython.display import display, Math, Latex

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

"""## Feature Extraction

* Feature Extraction aims to reduce the number of features in a dataset by creating new features from the existing ones (and then discarding the original features).

* These new reduced set of features should then be able to summarize most of the information contained in the original set of features.

* In this way, a summarised version of the original features can be created from a combination of the original set.

### **1. DictVectorizer**

Many a times the data is present as a $\textbf {list of dictionary objects.}$

ML algorithms expect the data in **matrix form** with shape $(n,m)$ where $n$ is the number of samples and $m$ is the number of features.
 `Vectorizer` **converts** a list of dictionary objects to feature matrix.

Let's create a sample data for demo purpose containing `age` and `height` of children.
  Each record/sample is a dictionary with two keys `age` and `height` , and corresponding values.
"""

from sklearn.feature_extraction import DictVectorizer

measurements = [
    {'city': 'Chennai', 'temperature': 33.},
    {'city': 'Kolkata', 'temperature': 18.},
    {'city': 'Delhi', 'temperature': 12.}]

vec = DictVectorizer()
vec.fit_transform(measurements).toarray()

vec.get_feature_names_out()

"""### **2. PCA - Principal Component Analysis**

* PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that capture maximum amount of the variance.

* It helps in reducing dimensions of a dataset, thus computational cost of next steps e.g. training a model, cross validation etc.

Let's generate some artificial data to better understand PCA :
"""

rand = np.random.RandomState(1)
X = np.dot(rand.rand(2, 2), rand.randn(2, 200)).T

plt.figure()
plt.title('Data points', size=20)

# set x and y labels
plt.xlabel('$x_1$', size=15)
plt.ylabel('$x_2$', size=15, rotation=0)

# plot the data points
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.axis('equal')

"""Let us fit a `PCA` transformer on this data and compute its two principal components:"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)

"""Let's print the principle axes, they are two 2D vectors for this example.

The PCA object returns them in the form of a matrix, where each column returns them in the form of a matrix, where each column represents a principle component:
"""

print(f'The {pca.n_components_} principle axes are :\n', pca.components_)

"""Let's also look at the explained variance corresponding to each priciple axes."""

print('Explained variance by each component : ', pca.explained_variance_)

"""To better understand PCA, let's visualize these principle axes :

* There are two principle axes **C1 and C2**. They are orthogonal to each other. An additional vector **C3** is also mentioned for comparision.

* The lengths of **C1** and **C2** are taken as square root of respective explained variance. The length of the vector implies how important that vector is.
"""

# draw projections of data points on different vectors

projections = X@pca.components_
print(projections.shape)

c3 = X[2]

arbitary_projection = X@c3
print(arbitary_projection.shape)

plt.figure(figsize=(12,8))

plt.scatter(projections[:, 0], 1+np.zeros((200, 1)), alpha=0.3, color='r')
plt.scatter(projections[:, 1], -1+np.zeros((200, 1)), alpha=0.3, color='b')
plt.scatter(arbitary_projection, np.zeros((200,)), alpha=0.3, color='grey')

plt.legend(['$\mathbf{C_2}$', '$\mathbf{C_3}$'], prop={'size': 16})
plt.title("variance covered by different vectors", size=20)

plt.ylim([-1.5, 1.5])
plt.yticks([], [])
plt.axis('equal')
plt.grid(True)
plt.xlabel('$z$', size=20)
plt.show()

"""*Reducing Dimensions*

We can use PCA to reduce number of dimensions of a dataset. The components that are least important i.e. their explained variance is low, are removed and only those components that capture high(i.e. desired) amount of variance are kept.

Let's reduce the dimension of our data from 2 to 1. We can observe the transformed data has only 1 feature.
"""

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)

print('Original shape :', X.shape)
print('Transformed shape :',X_pca.shape)

"""To better understand what happened to our data, let's visualize our original data and the reduced data.

To do this, we will need to bring the transformed data into space or original data, which can be accomplished by `inverse_transform` method of `PCA` object.

"""

plt.figure()
plt.title('Data and candidate vectors', size=20)

# set x and y labels
plt.xlabel('$x_1$', size=20)
plt.ylabel('$x_2$', size=20, rotation=0)

# plot data points
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)

for length, principal_axis, axis_name, i_color in zip(pca.explained_variance_,
pca.components_,['$\mathbf{C_1}$', '$\mathbf{C_2}$'], ['r', 'b']):
    v = principal_axis * np.sqrt(length)
    v0, v1 = pca.mean_, pca.mean_ + v

    # draw principal axis
    plt.quiver(*v0, *(v1-v0), scale=0.33, scale_units='xy', color=i_color)

    # label the  principal axis
    plt.text(*(3.4*v1), axis_name, size=20)

# draw 3rd component
lengths = np.eye(2)
np.fill_diagonal(lengths, np.sqrt(pca.explained_variance_))

c3 = pca.mean_+[-0.5, 0.3]

plt.quiver(*pca.mean_,*(1.1*(c3-pca.mean_)), scale=1, scale_units='xy',
color='grey')

# label the  principal axis
plt.text(*(1.4*c3),'$\mathbf{C_3}$',size=20,color='grey')

plt.axis('equal')
plt.show()

"""From above chart it is clear that the new/transformed data points are now projected on  C1  vector.

Import necessary libraries
"""

from IPython.display import display, Math, Latex

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

"""### Handling Imbalanced Data

Imbalanced datasets are those where one class is very less represented than the other class. This kind of data results in less efficient ML algorithm.

There are two main approaches to handle imbalanced data:
* Undersampling
* Oversampling

[Image Source](https://miro.medium.com/max/1400/0*mOgypphrofDS9Z32.png)

![Image Source](https://miro.medium.com/max/1400/0*mOgypphrofDS9Z32.png)

We will demonstrate how to handle imbalance with the help of **wine quality dataset** that we have used earlier.
"""

wine_data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

wine_data.shape

wine_data.info()

wine_data.quality.value_counts(ascending=True)

# display the histograms of the target variable 'quality'

wine_data['quality'].hist(bins=50)
plt.xlabel('Quality')
plt.ylabel('Number of samples')
plt.show()

"""### **1. Undersampling**

Undersampling refers to sampling from the majority class in order to keep only a part of these data points.

It may be carried out using **RandomUnderSampler** class from **imblearn** library.
"""

from imblearn.under_sampling import RandomUnderSampler

#class
class_count_3, class_count_4, class_count_5, class_count_6, class_count_7, class_count_8 = wine_data['quality'].value_counts()

# separate class
class_3 = wine_data[wine_data['quality'] == 3]
class_4 = wine_data[wine_data['quality'] == 4]
class_5 = wine_data[wine_data['quality'] == 5]
class_3 = wine_data[wine_data['quality'] == 3]
class_6 = wine_data[wine_data['quality'] == 6]
class_7 = wine_data[wine_data['quality'] == 7]
class_8 = wine_data[wine_data['quality'] == 8]


# print the shape of the class
print('class 3:', class_3.shape)
print('class 4:', class_4.shape)
print('class 5:', class_5.shape)
print('class 6:', class_6.shape)
print('class 7:', class_7.shape)
print('class 8:', class_8.shape)

wine_data.plot.hist()
plt.show()

# It allows you to count the items in an iterable list.
from collections import Counter

X = wine_data.drop(['quality'],axis=1)
y = wine_data['quality']

undersampler = RandomUnderSampler(random_state =0)
X_rus, y_rus = undersampler.fit_resample(X,y)

print('Original dataset shape : ',y.shape)
print('Resampled dataset shape : ', y_rus.shape)

print()
print(Counter(y))
print(Counter(y_rus))

"""The class with the least number of samples is '3'.

Hence all the other class samples are reduced to the number of samples in the least class.

### **2. Oversampling**

Oversampling refers to replicating some points from the minority class in order to increase the cardinality of the minority class.

This might consist of either replicating or generating synthetic data for the minority class.

It may be carried out using RandomOverSampler class from imblearn library.
"""

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X, y)

print('Original dataset shape : ', y.shape)
print('Resampled dataset shape : ', y_ros.shape)

print()
print(Counter(y))
print(Counter(y_ros))

print('New random points generated with RandomOverSampler : ',X_ros.shape[0] - X.shape[0])

"""The class with the majority number of samples is '5'. Hence all the other class samples that are lesser than this class count are newly sampled to the number of samples in the majority class.

#### **Oversampling using SMOTE**

SMOTE (Synthetic Minority Oversampling Technique) is a popular technique for over sampling. It is available under **imblean** library.
"""

from imblearn.over_sampling import SMOTE

oversampler = SMOTE()
X_smote, y_smote = oversampler.fit_resample(X, y)

Counter(y_smote)

print('New random points generated with SMOTE : ', X_ros.shape[0] - X.shape[0])

"""Types of SMOTE:

* Borderline SMOTE

* Borderline-SMOTE SVM

* Adaptive Synthetic Sampling(ADASYN)

Import necessary libraries
"""

from IPython.display import display, Math, Latex

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

"""#### **Chaining Transformers**

* The preprocessing transformations are applied one after another on the input feature matrix.

* It is important to apply exactly same transformation on training, evaluation and test set in the same order.

* Failing to do so would lead to incorrect predictions from model due to distribution shift and hence incorrect performance evaluation.

* The `sklearn.pipeline` module provides utilities to build a composite estimator, as a chain of transformers and estimators.

## Pipeline

Sequentially apply a list of transformers and estimators.

* Intermediate steps of the pipeline must be 'transformer' i.e, they must implement `fit` and `transform` methods.

* The final estimator only needs to implement `fit`.

The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.

### **1.Creating Pipelines**

A pipeline can be created with `Pipeline()`.

It takes a list of ('estimatorsName',estimator(...)) tuples. The pipeline object exposes interface of the last step.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

estimators = [
    ('simpleImputer' , SimpleImputer()),
    ('standardScaler' , StandardScaler()),
]

pipe = Pipeline(steps=estimators)

"""The same pipeline can also be created via `make_pipeline()` helper function, which doesn't take names of the steps and assigns them generic names based on their steps."""

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(SimpleImputer(), StandardScaler())

"""### **2.Accessing Individual steps in a Pipeline**"""

from sklearn.decomposition import PCA
estimators = [
    ('simpleImputer', SimpleImputer()),
    ('pca', PCA()),
    ('regressor', LinearRegression())
]
pipe = Pipeline(steps=estimators)

"""Let's print number of steps in this pipeline:"""

print(len(pipe.steps))

"""Let's look at each of the steps:"""

print(pipe.steps)

"""The second estimator can be accessed in following 4 ways:"""

print(pipe.named_steps.regressor)

pipe.steps[1]

pipe['pca']

"""### **3.Accessing parameters of a step in pipeline**

Parameters of the estimators in the pipeline can be accessed using the __syntax, note there are two underscores.
"""

estimators = [
    ('simpleImputer', SimpleImputer()),
    ('pca', PCA()),
    ('regressor', LinearRegression())
]

pipe = Pipeline(steps=estimators)
pipe.set_params(pca__n_components=2)

"""In above example `n_components` of `PCA()` step is set after the pipeline is created.

### **4.GridSeachCV with Pipeline**

By using naming convention of nested parameters, grid search can be implemented.
"""

from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = dict(
    imputer=['passthrough',SimpleImputer(),KNNImputer()],
    clf=[SVC(), LogisticRegression()],
    clf_C=[0.1, 1, 10, 100])

grid_search = GridSearchCV(pipe, param_grid=param_grid)

"""* `c` is an inverse of regularization, lower its value stronger the regularization is.

* In the example above `clf_C` provides a set of values for grid search.

### **Caching Transformers**

Transforming data is a computationally expensive step.
* For grid search, transformers need not be applied for every parameter configuration.

* They can be applied only once, and the transformed data can be reused.

* This can be achived by setting `memory` parameter of `pipeline` object.
"""

import tempfile
tempDirPath = tempfile.TemporaryDirectory()

estimators = [
              ('simpleImputer', SimpleImputer()),
              ('pca', PCA(2)),
              ('regressor',LinearRegression())
]

pipe = Pipeline(steps = estimators ,memory = tempDirPath)

"""### **FeatureUnion**
Concatenates results of multiple transformer objects.

* Applies a list of transformer objects in parallel, and their outputs are concatenated side-by-side into a larger matrix.

* `FeatuerUnion` and `Pipeline` can be used to create complex transformers.

### **5.Visualizing Pipelines**
"""

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('selector', ColumnTransformer([(
        'select_first_4', 'passthrough', slice(0, 4))])),

    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

cat_pipeline = ColumnTransformer([
    ('label_binarizer', LabelBinarizer(), [4]),
])

full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                               ('cat_pipeline', cat_pipeline)
                                               ])

from sklearn import set_config
set_config(display='diagram')

#displays HTML representation in a jupyter context
full_pipeline

"""### **Linear regression with sklearn API**

The objective of this notebook is to demonstrate how to build a linear regression model with `sklearn`.

We will be using the following set up:

1. Dataset : California Housing

2. Regression API : `LinearRegression`

3. Training : `fit` (normal equation) and `cross_validate` (normal equation with cross  validation).

4. Evaluation : `score` (r2 Score) and `cross_val_score` with different scoring parameters.

We will study the model diagnosis with `LearningCurve` and learn how to examine the learned model or weight vector.

### Importing the libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import permutation_test_score

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyRegressor

np.random.seed(42)
plt.style.use('seaborn')

"""We will use `ShuffleSplit` cross validation with:

* 10 folds (n_splits) and

* set aside 20% examples as test examples (`test_size`)

"""

shuffle_split_cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

"""### STEP 1:  **Load the dataset**

The first step is to load the dataset. We have already discussed how to load California Housing dataset in the last demonstration.
"""

features, labels = fetch_california_housing(as_frame=True, return_X_y=True)

"""The feature matrix is loaded in `features` dataframes and the labels in `labels` dataframe.

Let's examine the shapes of these two dataframes.
"""

print('Shape of feature matrix : ', features.shape)
print('Shape of labels matrix : ', labels.shape)

"""As a sanity check, make sure that the number of rows in feature matrix and labels match."""

assert (features.shape[0]==labels.shape[0])

"""### STEP 2: **Data Exploration**

Data exploration has beein covered in week 4 notebook.

### STEP 3: **Preprocessing and model building**

#### 3A. Train-test split

The first step is to split the training data into test set. We do not access the test data till the end.

All data exploration and tuning is performed on the training set and by setting aside a small portion of training as a dev or validation set.

The following code snippet divides the data into training and test sets :
"""

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, random_state=42)

train_features.info()

"""Let's examine the shapes of training and tet sets :"""

print('Number of training samples : ', train_features.shape[0])
print('Number of test samples : ', test_features.shape[0])

"""It's time to perform another sanity check-here we check if the training feature matrix has the same umber of rows as the training label vector.

We perform the same check on the test set too.
"""

assert (train_features.shape[0] == train_labels.shape[0])
assert (test_features.shape[0] == test_labels.shape[0])

"""#### 3B. Pipeline : Preprocessing + Model Building

* As a first step, build linear regression models with default parameter setting of `LinearRegression` APIs.

* We will make use of `Pipeline` API for combining data preprocessing and model building.

* We will use `StandardScaler` feature scaling to bring all features on the same scale followed by a `LinearRegression` model.

The `Pipeline` object has two components:
1. `StandardScaler` as step1
2. `LinearRegression` as step2

After constructing the pipeline object, let's train it with set :
"""

lin_reg_pipeline = Pipeline([
    ('feature_scaling', StandardScaler()),
    ('lin_reg', LinearRegression())
])

lin_reg_pipeline.fit(train_features, train_labels)

"""Now that we have trained the model, let's check the learnt / estimated weight vectors (intercept_, coef_) :"""

print('Intercept (w_0) : ',lin_reg_pipeline[-1].intercept_)
print()
print('Weight vector (w_1,w_2....,w_m) : \n' ,lin_reg_pipeline[-1].coef_)

"""A few things to notice:

* We accessed the `LinearRegression` object as `lin_reg_pipeline[-1]` which is the last step in pipeline.

* The intercept can be obtained via `intercept_` memeber variable and

* The weight vector correspoinding to features via `coef_`.

### STEP 4: **Model Evaluation**

Let's use `score` method to obtain train and test errors with twin objectives.

* Estimation of model performance as provided by test error.

* Comparision of errors for model diagnostic purpose (underfit /overfit /just the right fit)
"""

#evaluate model performance on both train and test set.

train_score = lin_reg_pipeline.score(train_features, train_labels)
print('Model performance on train set :', train_score)

test_score = lin_reg_pipeline.score(test_features, test_labels)
print('Model performance on test set :', test_score)

"""* The `score` method returns `r2` score whose best value is 1.

* The `r2` scores on training and test are comparable but they are not that high.

* It points to underfitting issue in model training.

#### 4A. Cross validation sccore (`cross_val_score`)

* Since the `score` was computed on one fold that was selected as a test set, it may not be all that robust.

* In order to obtain robust estimate of the performance, we use `cross_val_score` that calculates `score` on different test folds through cross validation.
"""

lin_reg_score = cross_val_score(lin_reg_pipeline, train_features, train_labels,scoring='neg_mean_squared_error' , cv=shuffle_split_cv)

print('Model performance on cross validation set : \n', lin_reg_score)

print(
    f'Score of linear regression model on the test set : \n'f"{lin_reg_score.mean():.3f} +/- {lin_reg_score.std():.3f}")

"""Here we got the negative mean squred error as a score. We can convert that to error as follows:"""

lin_reg_mse = - lin_reg_score

print(
    f'MSE of linear regression model on the test set :\n' f'{lin_reg_mse.mean():.3f} +/- {lin_reg_mse.std():.3f}')

"""We can use other `scoring` parameters and obtain cross validated scores based on that parameter.

The following choices are available for `scoring`:

* expalined_variance

* max_error

* neg_mean_absolute_error

* neg_root_mean_squared_log_error

* neg_median_absolute_error

* neg_mean_absolute_percentage_error

* r2 score

#### 4B. Cross validation

We just calculated `cross_val_score` based on the cross validation.

* It however return only scores of each fold. What if we also need to access the models trained in each fold along with some other statistics like `train error` for that fold.

* `cross_validate` API enables us to obtain them.
"""

lin_reg_cv_results = cross_validate(lin_reg_pipeline ,train_features ,train_labels ,scoring='neg_mean_squared_error' ,return_train_score=True ,return_estimator=True ,cv=shuffle_split_cv)

"""The `lin_reg_cv_results` is a dictionary with the following contents :

* trained `estimators`

* time taken for fitting (`fit_time`) and scoring(`score_time`) the models in cross validation,

* training score (`train_score`) and

* test scores (`test_score`)

##### **Returns of cross_validate score**

* scoresdict of float arrays of shape (n_splits,)

* Array of scores of the estimator for each run of the cross validation.

* A dict of arrays containing the score/time arrays for each scorer is returned.

* The possible keys for this dict are:

1. **test_score**

    * The score array for test scores on each cv split.
    
    * `Suffix_score` in `test_score` changes to a specific metric like `test_r2` or `test_auc` if there are multiple scoring metrics in the scoring parameter.

2. **train_score**

    * The score array for train scores on each cv split.
    
    * `Suffix_score` in `train_score` changes to a specific metric like `train_r2` or `train_auc` if there are multiple scoring metrics in the scoring parameter.
    
    * This is available only if `return_train_score` parameter is `True`.

3. **fit_time**
    
    * The time for fitting the estimator on each cv split.
    
    * This is available only if `return_fit_time` parameter is `True`.

    * The time for fitting the estimator on the train set for each cv split.


4. **score_time**

    The time for scoring the estimator on the test set for each cv split. (Note time for scoring on the train set is not included even if return_train_score is set to True)

5. **estimator**

    * The estimator objects for each cv split.
    
    * This is available only if `return_estimator` parameter is set to `True`.

Let's print the contents of the dictionary for us to examine :
"""

lin_reg_cv_results

"""* There are 10 values in each dictionary key. That is because of `cv`=10 or 10-fold cross validation that we used.

* We compare training and test errors to access generalization performance of our model. However we have training and test scores in the `cv_results` dictionary.

* Multiply these scores by -1 and convert them to errors.
"""

train_error = -1 * lin_reg_cv_results['train_score']
test_error = -1 * lin_reg_cv_results['test_score']

print(f'Mean squared error of linear regression model on the train set:\n',
      f'{train_error.mean():.3f} +\- {train_error.std():.3f}')

print()
print(f'Mean squared error of linear regression model on the test set:\n',
      f'{test_error.mean():.3f} +\- {test_error.std():.3f}')

"""#### 4C. **Learning Curve** / Effect of training set size on ERROR

Let's understand how the training set size or #samples affect the error.

We can use `Learning_curve` API that calculates cross validation scores for different #samples as specified in argument `train_sizes`.
"""

#@ title [Plot learning curves]
def plot_learning_curve(train_sizes, train_scores, test_scores):
    train_score_mean = np.mean(-train_scores, axis=1)
    train_score_std = np.std(-train_scores, axis=1)

    test_score_mean = np.mean(-test_scores, axis=1)
    test_score_std = np.std(-test_scores, axis=1)

    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    plt.fill_between(train_sizes,
                    train_score_mean - train_score_std,
                    train_score_mean + train_score_std,
                    alpha=0.1,
                    color='r',)

    plt.fill_between(train_sizes,
                    test_score_mean + test_score_std,
                    test_score_mean - test_score_std,
                    alpha=0.1,
                    color='g')

    plt.plot(train_sizes, train_score_mean, "o-", color='r', lw=2)
    plt.plot(train_sizes, test_score_mean, "o-", color='g', lw=2)

    plt.xlabel("Training examples ")
    plt.ylabel("MSE")
    # plt.legend(loc="best")

    return plt.show()

(train_sizes, train_scores, test_scores, fit_times, score_times) = learning_curve(lin_reg_pipeline, train_features, train_labels, cv=shuffle_split_cv,scoring='neg_mean_squared_error', n_jobs=-1,
return_times=True, train_sizes=np.linspace(0.2, 1, 10))

plot_learning_curve(train_sizes, train_scores, test_scores)

"""Observing that :

* Both curves have reached a plateau; they are close and fairly high.

* Few instances in the training set means the model can fit them perfectly. But as more instances are added to the training set, it becomes impossible for the model to fit the training data perfectly.

* When the model is trained on very few training instances, it is not able of generalizing properly, which is why the validation error is initially quite high.

* Then as the model learns on more training examples, the training and validation error reduce slowly.

These learning curves are typical of **underfitting** model.

#### 4D. **Scalability Curve** / Effect of training set size on FIT TIME

We can also study how training scales as the function of number of training samples.
"""

#@ title [Plot Scalability curves]
def plot_scalability_curve(train_sizes, fit_times):
    train_score_mean = np.mean(-train_scores, axis=1)
    train_score_std = np.std(-train_scores, axis=1)

    test_score_mean = np.mean(-test_scores, axis=1)
    test_score_std = np.std(-test_scores, axis=1)

    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    plt.fill_between(train_sizes,
                    fit_times_mean - fit_times_std,
                    fit_times_mean + fit_times_std,
                    alpha=0.1,
                    color='g',)

    plt.plot(train_sizes, fit_times_mean, "o-", color='b', lw=2)

    plt.xlabel("Training examples ")
    plt.ylabel("fit time")
    # plt.legend(loc="best")

    return plt.show()

plot_scalability_curve(train_sizes,fit_times)

"""As the number of training examples grows, the time to fit also increases.

#### 4E. Model Examination

Let's examine the weight vectors and how much variability exists between them across different cross-validated models.
"""

feature_names = train_features.columns
feature_names

"""For this we will first construct a dataframe of weight vectors and then plot them with `boxplot`."""

coefs = [i[-1].coef_ for i in lin_reg_cv_results["estimator"]]
weights_df = pd.DataFrame(coefs, columns=feature_names)

color = {'whiskers':'black','medians':'green','caps':'blue'}
weights_df.plot.box(color=color, vert=False,figsize=(12,12))

plt.title('Linear regression coefficients')
plt.show()

"""There is not much variability in weights by different models. It can also be seen in the standard deviation of weights as seen in `std` row below"""

weights_df.describe().T

"""#### 4F. Model Selection

Let's select the model with the lowest cross validated test error as the best performance model.
"""

lin_reg_cv_results['estimator']

best_model_index = np.argmin(test_error)
selected_model = lin_reg_cv_results['estimator'][best_model_index]

"""Let's examine the model coefficients and intercepts :"""

print('Intercept (w_0) :',selected_model['lin_reg'].intercept_)
print()
print('Coefficients (w_1,w_2.....,w_m) : \n',selected_model['lin_reg'].coef_)

"""#### 4G. Model Performance

Towards this, let's first obtain the predictions for test points in cross validation.
"""

from sklearn.model_selection import cross_val_predict

cv_predictions = cross_val_predict(lin_reg_pipeline, train_features, train_labels)

mse_cv = mean_squared_error(train_labels, cv_predictions)

plt.scatter(train_labels, cv_predictions, color='blue')
plt.plot(train_labels, train_labels, 'r-')

plt.title(f'Mean squared error = {mse_cv:.2f}', size=18)
plt.xlabel('Actual Median House value', size=12)
plt.ylabel('Predicted Median House value', size=12)
plt.show()

"""* The model seems to be all over the place in its predictions for examples with label 5.

* There are some negative predictions. We can fix this by adding a constraints on the weights to be positive.


At this stage, we should perform error analysis and check where the predictions are going wrong.

We can revisit feature construction, preprocessing or model stages and make the necessary course corrections to get better performance.

### STEP 5 : **Predictions**

We can use the best performing model from cross validation for getting predictions on the test set.
"""

test_predictions_cv = selected_model.predict(test_features)
test_predictions_cv[:5]

test_predictions = lin_reg_pipeline.predict(test_features)
test_predictions[:5]

"""### STEP 6 : **Report Model Performance**

We report the model perfromance on the test set.
"""

score_cv = selected_model.score(test_features, test_labels)
score = lin_reg_pipeline.score(test_features, test_labels)

print('R2 score for the best model obtained via cross validation :', score_cv)
print('R2 score for model w/o cv :', score)

"""Alternatively we can use any other metric of interest and report performance based on that.

For example, the mean squared error is as follows:
"""

mse_cv = mean_squared_error(test_labels, test_predictions_cv)
mse = mean_squared_error(test_labels, test_predictions)

print('MSE for the best model obtained via cross validation :', mse_cv)
print('MSE for model w/o cv : ', mse)

"""#### Testing Model on other metrics"""

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2, random_state=0)

lin_reg_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('lin_reg', LinearRegression())
])

lin_reg_pipeline.fit(X_train, y_train)
test_score = lin_reg_pipeline.score(X_test, y_test)
test_score

y_pred = lin_reg_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
np.sqrt(mse)

explained_variance_score(y_test, y_pred)

max_error(y_test,y_pred)

mean_absolute_error(y_test, y_pred)

mean_squared_error(y_test, y_pred)

"""### **BASELINE MODELS**

Now, we will build a couple of baseline models using `DummyRegression` and `permutation_test_score`.

We will compare performance of our linear regression model with these two baselines.

We will use `ShuffleSplit` as a cross validation strategy
"""

shuffle_split_cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

"""Let's load the data and split it into training and test."""

features, labels = fetch_california_housing(as_frame=True, return_X_y=True)
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels,  random_state=42)

"""#### 1. Linear Regression Classifier


* Build linear regression model with feature scaling as part of a pipeline.

* Train the model with 10-fold cross validation via ShuffleSplit.

* Capture errors on different folds.
"""

lin_reg_pipeline = Pipeline([
    ('feature scaling', StandardScaler()),
    ('lin_reg', LinearRegression())
])

lin_reg_cv_results = cross_validate(lin_reg_pipeline, train_features,       train_labels, cv=shuffle_split_cv, scoring='neg_mean_absolute_error', n_jobs=-1)

lin_reg_error = pd.Series(-lin_reg_cv_results['test_score'],
                           name='Linear regressor error')

lin_reg_cv_results.keys()

"""#### 2. Dummy Regression Classifier"""

def dummy_regressor_baseline(strategy, constant_val=None, quantile_val=None):
    baseline_model_median = DummyRegressor(strategy=strategy,
                                            constant=constant_val,
                                            quantile=quantile_val)

    baseline_median_cv_results = cross_validate(
        baseline_model_median, train_features, train_labels, cv=shuffle_split_cv, n_jobs=-1,  scoring='neg_mean_absolute_error')

    return pd.Series(-baseline_median_cv_results['test_score'], name="Dummy regressor error")

baseline_median_cv_results_errors = dummy_regressor_baseline(strategy='median')

baseline_mean_cv_results_errors = dummy_regressor_baseline(strategy='mean')

baseline_constant_cv_results_errors = dummy_regressor_baseline(
    strategy='constant', constant_val=2)

baseline_quantile_cv_results_errors = dummy_regressor_baseline(
    strategy='quantile', quantile_val=0.55)

"""Let's compare performance of these Dummy Regressors:"""

dummy_error_df = pd.concat([baseline_median_cv_results_errors,
                            baseline_mean_cv_results_errors,
                            baseline_constant_cv_results_errors,
                            baseline_quantile_cv_results_errors], axis=1)

dummy_error_df.columns = ['Median CV', 'Mean CV', 'Constant CV', 'Quantile CV']
dummy_error_df

"""Plotting erros using `barplot`"""

dummy_error_df.plot.hist(bins=50, density=True, edgecolor='black')

plt.legend(bbox_to_anchor=(1.05,0.8), loc='upper left')
plt.xlabel('Mean absolute error(k$)', size=12)
plt.ylabel('Frequency', size=12)
plt.title('Distribution of the testing errors' ,size=16)
plt.show()

"""#### Permutation_test_score

It permutes the target to generate randomized data and computes the empirical p-value against the null hypothesis, that features and targets are independent.

Here we are interested in `permutation_score` returned by this API, which indicates score of the model on different permutations.
"""

score, permutation_score, pvalue = permutation_test_score(
    lin_reg_pipeline, train_features, train_labels, cv=shuffle_split_cv, scoring='neg_mean_absolute_error', n_jobs=-1, n_permutations=30)

permutation_errors = pd.Series(-permutation_score, name='Permuted error')

print('Permutation test score :\n', permutation_score)

"""#### Model Comparision"""

dummy_error_df.plot.hist(bins=50, density=True, edgecolor='black')

plt.legend(bbox_to_anchor=(1.05,0.8), loc='upper left')
plt.xlabel('Mean absolute error(k$)', size=12)
plt.ylabel('Frequency', size=12)
plt.title('Distribution of the testing errors' ,size=16)
plt.show()

errors_df = pd.concat([lin_reg_error, baseline_median_cv_results_errors,permutation_errors], axis=1)

errors_df.plot.hist(bins=50, density=True, edgecolor='black')

plt.legend(bbox_to_anchor=(1.05, 0.8), loc='upper left')
plt.xlabel('Mean absolute error(k$)', size=12)
plt.ylabel('Frequency', size=12)
plt.title('Distribution of the testing errors', size=16)
plt.show()

"""### **Linear regression with iterative optimization: SGDRegressor**

In this notebook, we will build linear regression model, with `SGDRegressor`.

SGD offers a lot of control over optimization procedure through a number of hyperparameters. However, we need to set them to right values in order to make it work for training the model.

### Importing the libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import permutation_test_score

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyRegressor

np.random.seed(306)
plt.style.use('seaborn')

"""We will use `ShuffleSplit` as a cross validation strategy."""

shuffle_split_cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

"""### STEP 1:  **Load the dataset**"""

features ,labels = fetch_california_housing(as_frame=True, return_X_y=True)

"""### STEP 2 :  **Preprocessing**

Split data into training and test sets.
"""

com_train_features, test_features, com_train_labels, test_labels = train_test_split(features, labels, random_state=42)

"""Divide the training data into train and dev sets."""

train_features ,dev_features, train_labels, dev_labels = train_test_split(
    com_train_features, com_train_labels, random_state=42)

"""### STEP 3 :  **Model Building**

#### **Baseline SGDRegressor**

* Step 1 : To begin with, we instantiate a baseline `SGDRegressor` model with default parameters.

* Step 2 : Train the model with training feature matrix and labels.

* Step 3 : Obtain the score on the training and dev data.
"""

sgd = SGDRegressor(random_state=42)
sgd.fit(train_features, train_labels)

train_mae = mean_absolute_error(train_labels, sgd.predict(train_features))
dev_mae = mean_absolute_error(dev_labels, sgd.predict(dev_features))

print('Train MAE: ', train_mae)
print('Dev MAE: ', dev_mae)

"""We can observe that the mean absolute error is too high. The baseline model doesn't train well. This may happen due to large learning rate.

Let's investigate this issue by training the model step by step and recording training loss in each step.

#### **Adding a feature scaling step**

We know that, SGD is sensitive to feature scaling. Let's add a feature scaling step and check if we get better MAE.
"""

sgd_pipeline = Pipeline([
    ('scaler' , StandardScaler()),
    ('sgd' , SGDRegressor())
])

sgd_pipeline.fit(train_features, train_labels)

train_mae = mean_absolute_error(train_labels, sgd.predict(train_features))
dev_mae = mean_absolute_error(dev_labels, sgd.predict(dev_features))

print('Train MAE: ', train_mae)
print('Dev MAE: ', dev_mae)

"""The error is still high.

**Let's run `SGDRegressor` step by step and investigate issues with training :**

* Step 1 : Instantiate `SGDRegressor`  with `warm_start = True` and `tol=-np.infty`.

* Step 2 : Train SGD step by step and record regression loss in each step.

* Step 3 : Plot learning curves and see if there are any issues in training.
"""

eta0 = 1e-2
sgd_pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('sgd',SGDRegressor(max_iter=1, tol = -np.infty, warm_start=True, random_state=42))
])

loss = []

for epoch in range(100):
    sgd_pipeline.fit(train_features, train_labels) #continues where it left off
    loss.append(mean_squared_error(train_labels, sgd_pipeline.predict(train_features)))

plt.plot(np.arange(len(loss)), loss, 'g-')

plt.xlabel('Number of iterations ')
plt.ylabel('MSE')
plt.title(f'Learning curve: eta0={eta0:.3f}')
plt.show()

eta0 = 1e-3
sgd_pipeline = Pipeline([
                        ('feature_scaling', StandardScaler()),
                        ('sgd',SGDRegressor(max_iter=1, tol = -np.infty, warm_start=True, eta0=eta0,random_state=42))
])

loss = []

for epoch in range(100):
    sgd_pipeline.fit(train_features, train_labels)
    loss.append(mean_squared_error(train_labels, sgd_pipeline.predict(train_features)))

plt.plot(np.arange(len(loss)), loss, 'g-')

plt.xlabel('Number of iterations ')
plt.ylabel('MSE')
plt.title(f'Learning curve: eta0={eta0:.3f}')
plt.show()

"""The is an ideal learning curve where the train loss reduces monotonically as the training progresses."""

print("Number of iteration before reaching convergence criteria :",sgd_pipeline[-1].n_iter_)

print("Number of weight updates : ", sgd_pipeline[-1].t_)

"""**Checking train and dev mean absolute error.**"""

train_mae = mean_absolute_error(train_labels, sgd_pipeline.predict(train_features))
dev_mae = mean_absolute_error(dev_labels, sgd_pipeline.predict(dev_features))

print('Train MAE: ', train_mae)
print('Dev MAE: ', dev_mae)

"""#### **Fixing learning rate through validation curves**


* Step 1 : Provide the list of values to be tried for a hyperparameter.

* Step 2 : Instantiate an object of `validation_curve` with estimator, training features and label. Set `scoring` parameter to relevant score.

* Step 3 : Convert scores to error.

* Step 4 : Plot validation curve with the value of hyper-parameter on x-axis and error on the y-axis

* Step 5 : Fix the hyper-parameter value where the test error is the least.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# eta0 = [1e-5, 1e-4, 1e-3, 1e-2]
# 
# train_scores, test_scores = validation_curve(
#     sgd_pipeline, com_train_features, com_train_labels, param_name="sgd__eta0", param_range=eta0, cv=shuffle_split_cv, scoring='neg_mean_squared_error', n_jobs=2
# )

train_errors, test_errors = -train_scores, -test_scores

plt.plot(eta0, train_errors.mean(axis=1),'g-x',label='Training error')
plt.plot(eta0, test_errors.mean(axis=1),'r--x', label='Test error')

plt.legend()
plt.xlabel('eta0')
plt.ylabel('Mean absolute error')
plt.title('Validation curve for SGD')
plt.show()

"""For `eta0=1e-3`, the test error is the least and hence we select that value as the value for `eta0`.

Next we also plot standard deviation in errors.
"""

plt.errorbar(eta0, train_errors.mean(axis=1), yerr=train_errors.std(axis=1), label='Training error')

plt.errorbar(eta0, test_errors.mean(axis=1),yerr=test_errors.std(axis=1), label='Testing error')

plt.legend(loc='best')
plt.xlabel('eta0')
plt.ylabel('Mean absolute error')
plt.title('Validation curve for SGD')
plt.show()

"""#### **Experimenting with learning rate parameter**

##### 1. No learning rate parameter
"""

sgd_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(max_iter=500,
                        early_stopping=True,
                        eta0=1e-3,
                        tol=1e-3,
                        validation_fraction=0.2,
                        n_iter_no_change=5,
                        average=10,
                        random_state=42))
])

sgd_pipeline.fit(train_features, train_labels)

train_mae = mean_absolute_error(train_labels, sgd_pipeline.predict(train_features))

dev_mae = mean_absolute_error(dev_labels, sgd_pipeline.predict(dev_features))

print('Train MAE: ', train_mae)
print('Dev MAE: ', dev_mae)
print()

# development set dev set
print('Number of SGD iterations :', sgd_pipeline[-1].n_iter_)
print('Number of weight updates : ', sgd_pipeline[-1].t_)

"""##### 2. learning rate = 'constant'"""

sgd_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(max_iter=500,
                         early_stopping=True,
                         eta0=1e-3,
                         tol=1e-3,
                         learning_rate= 'constant',
                         validation_fraction=0.2,
                         n_iter_no_change=5,
                         average=10,
                         random_state=42))
])

sgd_pipeline.fit(train_features, train_labels)

train_mae = mean_absolute_error(
    train_labels, sgd_pipeline.predict(train_features))

dev_mae = mean_absolute_error(dev_labels, sgd_pipeline.predict(dev_features))

print('Train MAE: ', train_mae)
print('Dev MAE: ', dev_mae)
print()

# development set dev set
print('Number of SGD iterations :', sgd_pipeline[-1].n_iter_)
print('Number of weight updates : ', sgd_pipeline[-1].t_)

"""##### 3. learning rate = 'adaptive'"""

sgd_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(max_iter=500,
                         early_stopping=True,
                         eta0=1e-3,
                         tol=1e-3,
                         learning_rate='adaptive',
                         validation_fraction=0.2,
                         n_iter_no_change=5,
                         average=10,
                         random_state=42))
])

sgd_pipeline.fit(train_features, train_labels)

train_mae = mean_absolute_error(
    train_labels, sgd_pipeline.predict(train_features))

dev_mae = mean_absolute_error(dev_labels, sgd_pipeline.predict(dev_features))

print('Train MAE: ', train_mae)
print('Dev MAE: ', dev_mae)
print()

# development set dev set
print('Number of SGD iterations :', sgd_pipeline[-1].n_iter_)
print('Number of weight updates : ', sgd_pipeline[-1].t_)

"""#### **Setting `max_iters` parameter**"""

max_iter = np.ceil(1e6/com_train_features.shape[0])
max_iter

sgd_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(max_iter=max_iter,
                         early_stopping=True,
                         eta0=1e-3,
                         tol=1e-3,
                         learning_rate='adaptive',
                         validation_fraction=0.2,
                         n_iter_no_change=5,
                         average=10,
                         random_state=42))
])

sgd_pipeline.fit(train_features, train_labels)

train_mae = mean_absolute_error(
    train_labels, sgd_pipeline.predict(train_features))

dev_mae = mean_absolute_error(dev_labels, sgd_pipeline.predict(dev_features))

print('Train MAE: ', train_mae)
print('Dev MAE: ', dev_mae)
print()

# development set dev set
print('Number of SGD iterations :', sgd_pipeline[-1].n_iter_)
print('Number of weight updates : ', sgd_pipeline[-1].t_)

"""### SUMMARY :

In this notebook, we saw:

* how to build `SGDRegressor` model.

* how to tune the learning rate.

* how to use different `learning_rates` and their impact on convergence.

* how to use early stopping and averaged SGD

* how to tune hyper-parameters using `validation_curves`.

### **California housing dataset**

This notebook introduces California housing dataset that we will be using for regression demonstration.

We also list down the steps for typical dataset exploration, which can be applied broadly to any dataset.

### **Loading the dataset**

This dataset can be fetched from sklearn with `fetch_california_housing` API.
"""

from sklearn.datasets import fetch_california_housing
from scipy.stats import loguniform
from scipy.stats import uniform

"""In order to analyze the dataset, let's load it as a dataframe."""

california_housing = fetch_california_housing(as_frame=True)
type(california_housing)

"""The bunch object is a dictionary like object with the following attributes:
* `data`, is a pandas object (since `as_frame=True`).

* Each row corresponds to 8 features values.

* `target` value contains average house value in units of 100_000. This is also a pandas object (since `as_frame=True`).
* DESCR contains description of the dataset.
* `frame` contains dataframe with data and target

Each of these attributes can be accessed as `<bunch_object>`.key. In our case, we can access these features as follows:

* `california_housing.data` gives us access to contents of `data` key.

* `california_housing.target` gives us access to contents of `target` key.

* `california_housing.feature_names` gives us access to contents of `feature_names` key.

* `california_housing.DESCR` gives us access to contents of `DESCR` key.

* `california_housing.frame` gives us access to contents of `frame` key.

### **Dataset Exploration**

#### STEP 1: Dataset description
Let's look at the description of the dataset.
"""

print(california_housing.DESCR)

"""Note down key statistics from this description such as number of examples (or sample or instances) from the description :

* There are **20640 examples** in the dataset.

* There are **8 numberical attributes** per example

* The target label is median house value.

* There are **no missing values** in this dataset.

#### STEP 2 : Examine shape of feature matrix

Number of examples and features can be obtained via shape of `california_housing.data`.
"""

california_housing.data.shape

type(california_housing.data)

"""#### STEP 3 : Examine shape of label

Let's look at the shape of label vector.
"""

california_housing.target.shape

type(california_housing.target)

"""#### STEP 4: Get Feature names
Let's find out names of the attributes / features.
"""

california_housing.feature_names

"""Note the attributes and their description, which is a key step in understanding the data.

* MedInc - median income in block

* HouseAge - median house age in  block

* AveRooms - average number of rooms

* AveBedrms - average number of bedrooms

* Population - block population

* AveOccup - Average house occupancy

* Latitude - house block latitude

* Longitude - house block longitude

#### STEP 5: Examine sample training examples

Let's look at a few training examples along with labels.
"""

# frame.head() for both features and labels

california_housing.frame.head()

"""The dataset contains aggregated data about each district in California

#### STEP 6: Examine features
Let's look at the features.
"""

# data.head() for only features

california_housing.data.head()

"""We have information about :
* Demography of each district (income, population, house occupancy,

* Location of the disctricts (latitude and longitude) &

* Characteristics of houses in the district (#rooms, #bedrooms, age of house)

Since the information is aggregated at the district levels, the features corresponds to average or median.

#### STEP 7: Examine target
Let's look at the target to be predicted.
"""

# target.head() for only labels

california_housing.target.head()

"""The target contains median of the house value for each district. We can see that the target is a real number and hence this is a regression problem.

#### STEP 8: Examine details of features and labels
Let's look at the details of features and target labels.
"""

california_housing.frame.info()

"""We observe that :
* The dataset contains 20640 examples with 8 features.

* All features are numerical features encoded as floating point numbers.

* There are no missing values in any features - the `non-null` is equal to the number of examples in the training set.

#### STEP 9: Feature and target histograms.
Let's look at the distribution of these features and target by plotting their histograms.
"""

import matplotlib.pyplot as plt
import seaborn as sns

california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.5, wspace=0.4)

"""Let's observe these histogram and note down our findings:

* **MedInc** has a long tail distribution-salary of people is more or less normally distributed with a few folks getting a high salary.

* **HouseAge** has more or less a uniform distribution.

* The range for features, **AveRooms, AveBedrms, AveOccups, Population**, is large and its contains a small number of large values(as there are unnoticable bins on the right in the histograms plots of these features). That would mean that there could be certain outlier values present in these features.

* **Latitude and Longitude** carry geographical information. Their combination helps us decide price of the house.

* **MedHouseVal** also has a long tail distribution. It spikes towards the end. The reason is that the houses with price more than 5 are given value of 5.

#### STEP 10: Feature and target statistics

Let's look at statistics of these features and the target.
"""

california_housing.frame.describe().T

"""We can observe that there is a large difference between 75% and `max` values of `AveRooms`, `AveBedrms`, `population` and `AveOccups`- which confirms our intuition about presence of outliers or extreme values in these features.

#### STEP 11 : Pairplot
"""

_ = sns.pairplot(data=california_housing.frame, hue = 'MedHouseVal', palette='viridis')

"""A few observations based on pairplot:

* `MedIncome` seems to be useful in distinguishing between low and high valued houses.

* A few features have extreme values.

* Latitude and logitude together seem to distinguish between low and high valued houses.

### Summary
* Explored california housing dataset that would be used for demonstrating implementation of linear regression models.

* Examined various statistics of the dataset - #samples, #labels
* Examined distribution of features through histogram and pairplots.

### **Linear Regression for house-price prediction**

In this notebook, we will build different regression models for california house price prediction:

1. Linear Regression (with normal equation)

2. SGD Regression (linear regression with iterative optimization)

2. Polynomial Regression

3. Regularized Regression models : RIDGE & LASSO

We will set regularization rate and polynomial degree with hyper-parameter tuning and cross validation.

We will compare different models in terms of their parameter vectors and mean absolute error on train, eval and test sets.

#### *Imports*

* For regression problems, we need to import classes and utilities from `sklearn.linear_model`.

* This module has implementation for different regression models like, LinearRegression, SGDRegressor, Ridge, Lasso, RidgeCV and LassoCV.

* We also need to import a bunch of model selection utilities from `sklearn.model_selection` module and metrics from `sklearn.metrics` module.

* The data preprocessing utilities are imported from `sklearn.preprocessing` modules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import loguniform
from scipy.stats import uniform

from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

"""#### **Common set up**

Set up random seed to a number of your choice.
"""

np.random.seed(306)

"""Let's use `ShuffleSplit` as cv with 10 splits and 20% examples set aside as test examples."""

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

"""#### **Data Loading and Splitting**

We use california housing dataset for this demo.

* We will load this dataset with `fetch_california_housing` API as a dataframe.

* We will load the data and split it inot three parts- train, dev  and test. Train+Dev will be used for cross validation and test will be used for evaluating the trained models.
"""

# Fetching dataset
features, labels = fetch_california_housing(as_frame=True, return_X_y=True)

# train-test-split
com_train_features, test_features, com_train_labels, test_labels = train_test_split(features, labels, random_state=42)

# train --> train + dev split
train_features, dev_features, train_labels, dev_labels = train_test_split(
    com_train_features, com_train_labels, random_state=42)

"""Throughout this notebook, we will have the following pattern for each estimator:

* We will be using `pipeline` for combining data preprocessing and modelling steps. `cross_validate` for training the model with `ShuffleSplit` cross validation and `neg_mean_absolute_error` as a scoring metric.

* Convert the scores to error and report mean absolute errors on the dev set.

### 1. **Linear Regression (with normal equation)**

Let's use normal equation method to train linear regression model.

We set up pipeline with two stages:

* Feature scaling to scale the features and

* Linear regression on the transformed feature matrix.

Throughout this notebook, we will have the following pattern for each estimator:

* We will be using `pipeline` for combining data preprocessing and modelling steps. `cross_validate` for training the model with `ShuffleSplit` cross validation and `neg_mean_absolute_error` as a scoring metric.

* Convert the scores to error and report mean absolute errors on the dev set.
"""

lin_reg_pipeline = Pipeline([
    ("feature_scaling", StandardScaler()),
    ("lin_reg", LinearRegression())
])


lin_reg_cv_results = cross_validate(lin_reg_pipeline,
                                    com_train_features,
                                    com_train_labels,
                                    cv=cv,
                                    scoring="neg_mean_absolute_error",
                                    return_train_score=True,
                                    return_estimator=True)

lin_reg_train_error = -1 * lin_reg_cv_results['train_score']
lin_reg_test_error = -1 * lin_reg_cv_results['test_score']


print(f"Mean absolute error of linear regression model on the train set:\n" f"{lin_reg_train_error.mean():.3f} +/- {lin_reg_train_error.std():.3f}")

print()
print(f"Mean absolute error of linear regression model on the test set:\n" f"{lin_reg_test_error.mean():.3f} +/- {lin_reg_test_error.std():.3f}")

"""Both the errors are close, but are not low. This points to underfitting. We can address it by adding more feature through polynomial regression.

### 2. **SGD Regression (iterative optimization)**

Let's use iterative optimization method to train linear regression model.

We set up pipeline with two stages:

* Feature scaling to scale features and

* SGD regression on the transformed feature matrix
"""

sgd_reg_pipeline = Pipeline([
    ('feature_scaling', StandardScaler()),
    ('sgd_reg', SGDRegressor(max_iter=np.ceil(1e6/com_train_features.shape[0]),
                            early_stopping=True,
                            eta0=1e-4,
                            learning_rate='constant',
                            tol=1e-5,
                            validation_fraction=0.1,
                            n_iter_no_change=5,
                            average=10,
                            random_state=42))
])

sgd_reg_cv_results = cross_validate(sgd_reg_pipeline,
                                    com_train_features,
                                    com_train_labels,
                                    cv=cv,  # shufflesplit declared above
                                    scoring='neg_mean_absolute_error',
                                    return_train_score=True,
                                    return_estimator=True)

sgd_train_error = -1 * sgd_reg_cv_results['train_score']
sgd_test_error = -1 * sgd_reg_cv_results['test_score']

print(f"Mean absolute error of SGD regression model on the train set:\n" f"{sgd_train_error.mean():.3f} +/- {sgd_train_error.std():.3f}")

print(f"Mean absolute error of SGD regression model on the test set:\n" f"{sgd_test_error.mean():.3f} +/- {sgd_test_error.std():.3f}")

"""#### SGD Regression : Regularization & Hyper-parameter tuning

We can also perform regularization with SGD. `SGDRegressor` has many hyperparameters that require careful tuning to achieve the same performance as wtih `LinearRegression`.
"""

poly_sgd_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('feature_scaling', StandardScaler()),
    ('sgd_reg', SGDRegressor(
        penalty='elasticnet',
        random_state=42
    ))])

poly_sgd_cv_results = cross_validate(poly_sgd_pipeline,
                                     com_train_features,
                                     com_train_labels,
                                     cv=cv,
                                     scoring='neg_mean_absolute_error',
                                     return_train_score=True,
                                     return_estimator=True)

poly_sgd_train_error = -1 * poly_sgd_cv_results['train_score']
poly_sgd_test_error = -1 * poly_sgd_cv_results['test_score']


print(f"Mean absolute error of SGD regression model on the train set. \n {poly_sgd_train_error.mean():.3f} +/- {poly_sgd_train_error.std():.3f}")

print(f"Mean absolute error of SGD regression model on the test set. \n {poly_sgd_test_error.mean():.3f} +/- {poly_sgd_test_error.std():.3f}")

"""The error is too high.

* So now, lets search for the best set of parameters for polynomial + SGD pipeline with `RandomizedSearchCV`.

* In `RandomizedSearchCV`, we need to specify distributions for hyperparameters.
"""

class uniform_int:
    """
    Integer valued version of the uniform distributions
    """

    def __init__(self, a, b):
        self._distribution = uniform(a, b)

    def rvs(self, *args, **kwargs):
        """ Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)

"""Let's specify `RandomizedSearchCV` set up."""

param_distributions = {
    'poly__degree':[1,2,3],
    'sgd_reg__learning_rate': ['constant', 'adaptive', 'invscaling'],
    'sgd_reg__l1_ratio': uniform(0,1),
    'sgd_reg__eta0': loguniform(1e-5,1),
    'sgd_reg__power_t': uniform(0,1)
}

poly_sgd_random_search_cv = RandomizedSearchCV(
    poly_sgd_pipeline, param_distributions=param_distributions, n_iter =10, cv=cv, verbose=1, scoring='neg_mean_absolute_error'
)

poly_sgd_random_search_cv.fit(com_train_features,com_train_labels)

"""The best score can be obtained as follows :"""

poly_sgd_random_search_cv.best_score_

"""The best set of parameters are obtained as follows:"""

poly_sgd_random_search_cv.best_params_

"""And the best estimator can be accessed as follows :"""

poly_sgd_random_search_cv.best_estimator_

"""### 3. **Polynomial Regression**

* We will train a polynomial model with degree 2 and later we will use `validation_curve` to find out right degree to use for polynomial models.

* `PolynomialFeatures` transforms the features to the user specified degrees (here it is 2).

* We perform feature scaling on the transformed features before using them for training the regression model.

"""

poly_reg_pipeline = Pipeline([
                             ('poly', PolynomialFeatures(degree=2)),
                             ('feature_scaling', StandardScaler()),
                             ('lin_reg', LinearRegression())])

poly_reg_cv_results = cross_validate(poly_reg_pipeline,
                                     com_train_features,
                                     com_train_labels,
                                     cv=cv,
                                     scoring='neg_mean_absolute_error',
                                     return_train_score=True,
                                     return_estimator=True)

poly_reg_train_error = -1 * poly_reg_cv_results['train_score']
poly_reg_test_error = -1 * poly_reg_cv_results['test_score']

print(f"Mean absolute error of polynomial regression model of degree 2 on the train set: \n" f"{poly_reg_train_error.mean():.3f} +/- {poly_reg_train_error.std():.3f}")

print(f"Mean absolute error of polynomial regression model of degree 2 on the test set: \n" f"{poly_reg_test_error.mean():.3f} +/- {poly_reg_test_error.std():.3f}")

"""Notice that the training and validation errors have reduced after using the second order polynomial features to represent the model.

Instead of using all polynomial feature, we use only interaction feature terms (i.e `interaction_only = True` ) in polynomial model and train the linear regression model.
"""

poly_reg_pipeline = Pipeline([
                             ('poly', PolynomialFeatures(
                                 degree=2, interaction_only=True)),
                             ('feature_scaling', StandardScaler()),
                             ('lin_reg', LinearRegression())])

poly_reg_cv_results = cross_validate(poly_reg_pipeline,
                                     com_train_features,
                                     com_train_labels,
                                     cv=cv,
                                     scoring='neg_mean_absolute_error',
                                     return_train_score=True,
                                     return_estimator=True)

poly_reg_train_error = -1 * poly_reg_cv_results['train_score']
poly_reg_test_error = -1*poly_reg_cv_results['test_score']

print(f"Mean absolute error of polynomial regression model of degree 2 on the train set: \n" f"{poly_reg_train_error.mean():.3f} +/- {poly_reg_train_error.std():.3f}")

print(f"Mean absolute error of polynomial regression model of degree 2 on the test set: \n" f"{poly_reg_test_error.mean():.3f} +/- {poly_reg_test_error.std():.3f}")

"""Notice that the training and validation errors have increased after using `interaction_only = True` to represent the model.

Let's figure out which degree polynomial is better suited for the regression problem at our hand. For that we will use `validation_curve`, which can be considered as a manual huperparameter tuning.

Here we specify a list of values that we want to try for polynomial degree and specify it as a parameter in `validation_curve`.
"""

degree = [1, 2, 3, 4, 5]

train_scores, test_scores = validation_curve(
    poly_reg_pipeline, com_train_features, com_train_labels, param_name='poly__degree',
    param_range=degree, cv=cv, scoring='neg_mean_absolute_error', n_jobs=2
)

train_errors, test_errors = -train_scores, -test_scores
plt.plot(degree, train_errors.mean(axis=1), 'b-x', label="Training error")
plt.plot(degree, test_errors.mean(axis=1), 'r-x', label="Test error")
plt.legend()

plt.xlabel("degree")
plt.ylabel("Mean absolute error (k$)")
plt.title("Validation curve for polynomial regression")
plt.show()

"""We would select a degree for which the mean absolute error is the least.

In this case, it is degree = 2 that yields the least mean absolute error and that would be selected as an optimal degree for polynomial regression.

### 4. **Ridge Regression**

* The polynomial models have a tendency to overfit - if we use higher order polynomial features.

* We will use `Ridge` regression - which penalizes for excessive model complexity in the polynomial regression by adding a regularization term.

* Here we specify the regularization rate `alpha` as 0.5 and train the regression model.

* Later we will launch hyperparameter search for the right value of `alpha` such that it leads to the least cross validation errors.
"""

ridge_reg_pipeline = Pipeline([
                             ('poly', PolynomialFeatures(degree=2)),
                             ('feature_scaling', StandardScaler()),
                             ('ridge', Ridge(alpha=0.5))])

ridge_reg_cv_results = cross_validate(ridge_reg_pipeline,
                                      com_train_features,
                                      com_train_labels,
                                      cv=cv,
                                      scoring='neg_mean_absolute_error',
                                      return_train_score=True,
                                      return_estimator=True)

ridge_reg_train_error = -1 * ridge_reg_cv_results['train_score']
ridge_reg_test_error = -1 * ridge_reg_cv_results['test_score']

print(f'Mean absolute error of ridge regression model (alpha=0.5) the train set: \n' f'{ridge_reg_train_error.mean():.3f} +/- {ridge_reg_train_error.std():.3f}')

print(f'Mean absolute error of ridge regression model (alpha=0.5) the test set: \n' f'{ridge_reg_test_error.mean():.3f} +/- {ridge_reg_test_error.std():.3f}')

"""#### Hyperparameter tuning for ridge regularization rate"""

alpha_list = np.logspace(-4, 0, num=20)

ridge_reg_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('feature_scaling', StandardScaler()),
    ('ridge_cv', RidgeCV(alphas=alpha_list, cv=cv, scoring='neg_mean_absolute_error'))
])

ridge_reg_cv_results = ridge_reg_pipeline.fit(com_train_features, com_train_labels)

print('The score with the best alpha is :',
      f'{ridge_reg_cv_results[-1].best_score_:.3f}')

print('The error with the best alpha is :',
      f'{-ridge_reg_cv_results[-1].best_score_:.3f}')

print('The best value for alpha :', ridge_reg_cv_results[-1].alpha_)

"""#### Ridge HPT through **GridSearchCV**"""

ridge_grid_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('feature_scaling', StandardScaler()),
    ('ridge', Ridge())])

param_grid = {'poly__degree': (1, 2, 3),
              'ridge__alpha': np.logspace(-4, 0, num=20)}

ridge_grid_search = GridSearchCV(ridge_grid_pipeline,
                                 param_grid=param_grid,
                                 n_jobs=-1,
                                 cv=cv,
                                 scoring='neg_mean_absolute_error',
                                 return_train_score=True)

ridge_grid_search.fit(com_train_features, com_train_labels)

"""`ridge_grid_search.best_index_` gives the index of the best parameter in the list.

"""

# best parameter
ridge_grid_search.cv_results_['params'][ridge_grid_search.best_index_]

mean_train_error = -1 * ridge_grid_search.cv_results_[
        'mean_train_score'][ridge_grid_search.best_index_]

mean_test_error = -1 * ridge_grid_search.cv_results_[
        'mean_test_score'][ridge_grid_search.best_index_]

std_train_error = -1 * ridge_grid_search.cv_results_[
        'std_train_score'][ridge_grid_search.best_index_]

std_test_error = -1 * ridge_grid_search.cv_results_[
        'std_test_score'][ridge_grid_search.best_index_]


print(f'Best Mean absolute error of polynomial ridge regression model on the train set:\n' f"{mean_train_error:.3f} +/- {std_train_error:.3f}")

print()
print(f'Best Mean absolute error of polynomial ridge regression model on the test set:\n' f"{mean_test_error:.3f} +/- {std_test_error:.3f}")

print('Mean cross validated score of the best estimator is : ',
      ridge_grid_search.best_score_)

print('Mean cross validated error of the best estimator is : ', -
      ridge_grid_search.best_score_)

ridge_grid_search.best_estimator_

"""### 5. **Lasso Regression**"""

lasso_reg_pipeline = Pipeline([
                                ('poly',PolynomialFeatures(degree=2)),
                                ('feature_scaling',StandardScaler()),
                                ('lasso',Lasso(alpha=0.001))
])

lasso_reg_cv_results = cross_validate(lasso_reg_pipeline,
                                       com_train_features,
                                       com_train_labels,
                                       scoring='neg_mean_absolute_error',
                                       return_train_score=True,
                                       return_estimator=True )

lasso_reg_train_error = -1 * lasso_reg_cv_results['train_score']
lasso_reg_test_error =-1 * lasso_reg_cv_results['test_score']

print(f'Mean absolute error of linear regression model on the train set : \n' f'{lasso_reg_train_error.mean():.3f} +/- {lasso_reg_train_error.std():.3f}')

print(f'Mean absolute error of linear regression model on the test set : \n' f'{lasso_reg_test_error.mean():.3f} +/- {lasso_reg_test_error.std():.3f}')

"""#### Lasso Regression with **GridSearchCV**"""

lasso_grid_pipeline =Pipeline([
                             ('poly',PolynomialFeatures()),
                             ('feature_scaling',StandardScaler()),
                             ('lasso',Lasso())])

param_grid ={"poly__degree": (1,2,3),
             "lasso__alpha": np.logspace(-4,0, num=20)}

lasso_grid_search = GridSearchCV(lasso_grid_pipeline,
                                param_grid=param_grid,                         n_jobs=2,
                                cv =cv,
                                scoring='neg_mean_absolute_error',
                                return_train_score = True)

lasso_grid_search.fit(com_train_features, com_train_labels)

mean_train_error = -1 * lasso_grid_search.cv_results_['mean_train_score'][lasso_grid_search.best_index_]

mean_test_error = -1 * lasso_grid_search.cv_results_['mean_test_score'][lasso_grid_search.best_index_]

std_train_error = -1 * lasso_grid_search.cv_results_['std_train_score'][lasso_grid_search.best_index_]

std_test_error = -1 * lasso_grid_search.cv_results_['std_test_score'][lasso_grid_search.best_index_]


print(f'Best Mean absolute error of polynomial lasso regression model on the train set : \n' f"{mean_train_error:.3f} +/- {std_train_error:.3f}")

print(f'Best Mean absolute error of polynomial lasso regression model on the test set : \n' f"{mean_test_error:.3f} +/- {std_test_error:.3f}")

"""### **Comparision of weight vectors**

Let's look at the weight vectors produced by different models.

#### 1. Polynomial Regression with CV
"""

feature_names = poly_reg_cv_results["estimator"][0][0].get_feature_names_out(
    input_features=train_features.columns)

print(feature_names)

print(poly_reg_cv_results['estimator'][0][-1])

coefs = [i[-1].coef_ for i in poly_reg_cv_results["estimator"]]
print(coefs[:2])

weights_poly_df = pd.DataFrame(coefs, columns=feature_names)

color = {'whiskers': 'black', 'medians': 'green', 'caps': 'blue'}
weights_poly_df.plot.box(color=color, vert=False, figsize=(12, 12))

plt.title('Polynomial regression coefficients')
plt.grid()
plt.show()

"""#### 2. Ridge Regression with CV"""

ridge_reg_pipeline = Pipeline([
                             ('poly', PolynomialFeatures(degree=2)),
                             ('feature_scaling', StandardScaler()),
                             ('ridge', Ridge(alpha=0.5))])

ridge_reg_cv_results = cross_validate(ridge_reg_pipeline,
                                      com_train_features,
                                      com_train_labels,
                                      cv=cv,
                                      scoring='neg_mean_absolute_error',
                                      return_train_score=True,
                                      return_estimator=True)

feature_names = ridge_reg_cv_results['estimator'][0][0].get_feature_names_out(
    input_features=train_features.columns)

feature_names

coefs = [i[-1].coef_ for i in ridge_reg_cv_results["estimator"]]
print(coefs[:1])

weights_ridge_df = pd.DataFrame(coefs, columns=feature_names)

color = {'whiskers': 'black', 'medians': 'green', 'caps': 'blue'}
weights_ridge_df.plot.box(color=color, vert=False, figsize=(12, 12))

plt.title('Ridge regression coefficients')
plt.grid()
plt.show()

"""### **Comparing Performance on test set**

#### 1. Baseline Model
"""

baseline_model_median = DummyRegressor(strategy='median')
baseline_model_median.fit(train_features, train_labels)

mean_absolute_percentage_error(test_labels, baseline_model_median.predict(test_features))

"""#### 2. Linear Regression with normal equation"""

mean_absolute_percentage_error(test_labels ,lin_reg_cv_results['estimator'][0].predict(test_features))

"""#### 3. SGD regression with randomsearchCV"""

mean_absolute_percentage_error(test_labels ,poly_sgd_random_search_cv.best_estimator_.predict(test_features))

"""#### 4. Polynomial Regression"""

poly_reg_pipeline.fit(com_train_features ,com_train_labels)

mean_absolute_percentage_error(test_labels ,poly_reg_pipeline.predict(test_features))

"""#### 5. Lasso Regression"""

mean_absolute_percentage_error(test_labels ,lasso_grid_search.best_estimator_.predict(test_features))

"""#### 6. Ridge Regression"""

mean_absolute_percentage_error(test_labels ,ridge_grid_search.best_estimator_.predict(test_features))

"""### **Introduction**

* Over the past four weeks we explored various data preprocessing techniques and solved some regression problems using linear and logistic regression models. The other side of the supervised learning paradigm is **classification problems**.

* To solve such problems we are going to consider **image classification** as a running example and solving it using **Perceptron()** method.

### **Imports**


* For classification problems, we need to import classes and utilities from sklearn.linear_model.

* This module has implementations for different classification models like `Perceptron, LogisticRegression, svm` and `knn`.

* We also need to import a bunch of model selection utilities from `sklearn.model_selection` module and metrics from `sklearn.metrics` module.

* The data preprocessing utilities are imported from `sklearn.preprocessing` modules.
"""

# Common imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns ;sns.set()
import os
import io
import warnings

# sklearn specific imports
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron

from sklearn.metrics import hinge_loss
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, classification_report

from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV

from pprint import pprint

"""## **Handwritten Digit Classification**

* We are going to use **Perceptron Classifier** to classify (recognize) given digit images.

* Since a single perceptron could **only** be used for **binary classification**, we consider only two classes in the first half. Eventually we will extend it to a multi-class setting.

* Suppose we want to recognize whether the given image is of digit zero or not  (digit other than zero). Then the problem could be cast as a binary classification problem.

* The first step is to **create a dataset** that contains a **collection of digit images** (also called examples, samples) written by humans. Then each image should be **labelled** properly.

### **Data Loading**
"""

# returns Data and Label as a pandas dataframe
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

"""The data matrix $X$ and the respective label vector $ y$ need to be converted to the numpy array by calling a `to_numpy` method."""

X = X.to_numpy()
y = y.to_numpy()

"""Let's get some information like number of features, number of classes about the dataset.

Observe that the labels are of string data type not integers.
"""

target_names = np.unique(y)
print('Number of samples : {0}, type : {1}'.format(X.shape[0], X.dtype))
print('Number of features : {0}'.format(X.shape[1]))

print('Minimum : ', np.min(X))
print('Maximum : ', np.max(X))

print('Number of classes : {0},type :{1}'.format(len(target_names), y.dtype))
print('Labels : {0}'.format(target_names))

"""* The **MNIST** dataset is clean and the range of values that each feature can take is also known. Therefore, the samples in the dataset may not require many data preprocessing techniques.

* However, it is often better to scale the range of features between 0 to 1.

* So, we can either use `MinMaxScaler` or `MaxAbsScaler`. They don't make any difference as the image pixels can takes only positive value from 0 to 255.
"""

X = MinMaxScaler().fit_transform(X)

print('Minimum : ', np.min(X))
print('Maximum : ', np.max(X))

"""### **Data Visualization**

Let us pick a few images (the images are already shuffled in the dataset) and display them with their respective labels.

As said above, the images are stacked as a row vector of size $ 1 \times 784$ and therefore must be reshaped to the matrix of size $ 28 \times 28$ to display them properly.
"""

# Choose a square number
num_images = 9
factor = int(np.sqrt(num_images))

fig,ax = plt.subplots(nrows=factor, ncols=factor,figsize=(8,6))
# take "num_images" starting from the index "idx_offset"
idx_offset = 0

for i in range(factor):
  index = idx_offset + i*(factor)
  for j in range(factor):
    ax[i,j].imshow(X[index+j].reshape(28,28), cmap='gray')
    ax[i,j].set_title('Label : {0}'.format(str(y[index+j])))
    ax[i,j].set_axis_off()

"""If we closely observe, we can see that there are moderate variations in the appearance of digits (ex: digit 1).

These matrices also close to sparse (i.e, there are lots of zero / black pixels in the matrix than non-zero pixels)

### **Data Splitting**

* Now, we know the details such as number of samples, size of each sample, number of features (784), number of classes (targets) about the dataset.

* So let's split the total number of samples into train and test set in the following ratio : 60000/10000 (i.e 60000 samples in the training set and 10000 samples in the testing set).


* Since the samples in the data set are already randomly shuffled, we need **not to** shuffle it again. Therefore using `train_test_split()` may be skipped.

## **Binary Classification : 0-Detector**
"""

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

"""### **Handling Imbalanced Data**

Before proceeding further, we need to check whether the dataset is balanced or imbalanced.

We can do it by plotting the distribution of samples in each classes.
"""

plt.figure(figsize=(10, 4))

sns.histplot(data=np.int8(y_train), binwidth=0.45, bins=11)
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

plt.xlabel('Class')
plt.title('Distribution of Samples')
plt.show()

"""### **Modifying Labels**

* Let us start with a simple classification problem, i.e **binary classification**.

* Since the original label vector contains **10** classes, we need to modify the number of classes to 2.

* Therefore, the label **0** will be changed **1** and all the other labels (1-9) will be changed to **-1**.

* We will name the label vectors as `y_train_0` and `y_test_0`.
"""

# initialize new variable names with all -1
y_train_0 = -1*np.ones(len(y_train))
y_test_0 = -1*np.ones(len(y_test))

# find indices of digit 0 image
indx_0 = np.where(y_train=='0')

# remember original labels are of type str not int, so use those indices to modify y_train_0 & y_test_0

y_train_0[indx_0] = 1
indx_0 = np.where(y_test =='0')
y_test_0[indx_0] = 1

"""#### Sanity check :

Let's display the elements of y_train and y_train_0 to verify whether the labels are properly modified.
"""

# Choose a square number
num_images = 9
factor = int(np.sqrt(num_images))

fig,ax = plt.subplots(nrows=factor, ncols=factor,figsize=(8,6))
# take "num_images" starting from the index "idx_offset"
idx_offset = 0

for i in range(factor):
    index = idx_offset + i*(factor)
    for j in range(factor):
        ax[i,j].imshow(X[index+j].reshape(28,28), cmap='gray')
        ax[i,j].set_title('Label : {0}'.format(str(y_train_0[index+j])))
        ax[i,j].set_axis_off()

"""## **Model**

### **Baseline Models**

Let us quickly construct a baseline model with the following rule : (you are free to choose different rules)

1. Count number of samples per class.

2. The model **always outputs** the class which has highest number of samples.

3. Then calculate the accuracy of the baseline model.
"""

num_pos = len(np.where(y_train_0 == 1)[0])
num_neg = len(np.where(y_train_0 == -1)[0])

print(num_pos)
print(num_neg)

base_clf = DummyClassifier(strategy='most_frequent')
base_clf.fit(X_train ,y_train_0)

print('Training accuracy : {0:.4f}'.format(base_clf.score(X_train, y_train_0)))
print('Testing accuracy : {0:.4f}'.format(base_clf.score(X_test,y_test_0)))

"""Now the reason is obvious. The model would have predicted 54077 samples correctly just by outputing -1 for all the input samples.

Therefore, the accuracy will be simply :  $ \frac{54077}{60000} = 0.90128 $

This is the reason why **"accuracy"** alone is **not always a good** measure!.

### **Perceptron Model**


Quick recap of various components in the general settings:

##### 1. **Training data**

* consists of features & labels or $(\mathbf X,y)$

* Here, $y$ is a **discrete** number from a finite set.

* **Features** in this case are **pixel** values of an image.


##### 2. **Model** :

\begin{align}
h_w:y&=&\text g(\mathbf w^T
\mathbf x) \\
&=&\text g(w_0+w_1x_1+\ldots + w_mx_m)
\end{align}

where,
* $\mathbf w$ is weight vector in $\mathbb{R}^{(m+1)}$ i.e. it has components : $\{w_0,w_1,\ldots,w_m\}$

* g(z) is a non-linear activation function given by a sign function:

$$\text g(z)=\begin{cases} +1 ,\text {if} \ z \ge 0 \\
-1, \text {otherwise}(i.e. z \lt 0)\end{cases}$$

##### 3. **Loss function** :

Let $ {\hat y}^{(i)} \in \{-1,+1\}$ be the prediction from perceptron and ${\hat y}^{(i)}$ be the actual label for $i-\text{th}$ example.
$ \\ $

The error is :

$$\text e^{(i)}=\begin{cases} 0 , \ \ \text { if} \ \ {\hat y}^{(i)} = y^{(i)} \\
-\mathbf {w^Tx^{(i)}}y^{(i)}, \text { otherwise} (i.e. {\hat y}^{(i)} \ne y^{(i)})\end{cases}$$

This can be compactly written as:

\begin{equation}
e^{(i)}=\max(-\mathbf{w^Tx^{(i)}}y^{(i)},0)=\max(-h_{\text w }(\mathbf x^{(i)})y^{(i)},0)
\end{equation}

##### 4. **Optimization** :

Perceptron learning algorithm :

1. Initialize $\mathbf {\text w}^{(0)}=0$

2. For each training example $(x^{(i)},y^{(i)})$

* ${\hat y}^{(i)}=\text{sign}(\mathbf {w^Tx}^{(i)})[\text {Calculates the output value}]$

* $\mathbf w^{(t+1)} := \mathbf w^{(t)}+ \alpha (y^{(i)}-{\hat y}^{(i)})\mathbf x^{(i)}[\text{Updates the weights}] $

**IMP** : Linearly separable examples lead to convergence of the algorithm with zero training loss, else it oscillates.

#### **Parameters of Perceptron Class**

* Let's quickly take a look into the important parameters of the Perceptron()

class sklearn.linear_model.Perceptron `(*,penalty=None, alpha = 0.0001, l1_ratio=0.15, fit_intercept = True, max_iter=1000,tol=0.001, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,class_weight=None, warm_start=False).`

* We need not to pay attention to all the arguments and their default values.

* Internally, the API uses the perceptron loss (i.e. it calls **Hinge(0,0)**, where 0.0 is a threshold) and uses SGD to update the weights.

* The other way of deploying perceptron is to use the general `linear_model.SGDClassifier` with `loss='perceptron'`

* The above loss is termed as hard Hinge-loss (as scores pass through the sign function) and hence we can't use SGD.

* Whereas, sklearn implements hinge-loss with the following definition: $\max  (0,-wx^iy^i$) and by default calls SGD to minimize the loss.

#### **Instantiation**

Create an instantiation of binary classifier (bin_clf).
"""

bin_clf = Perceptron(max_iter=100,random_state=1729)

"""#### **Training and Prediction**

* Call the `fit` method to train the model.

* It would be nice to plot the iteration vs loss curve for the training. However, sklearn does not have a direct function to plot it.

* Nevertheless, we can workaround this using `partial_fit` method (explained later)
"""

bin_clf.fit(X_train, y_train_0)

print('Dimension of Weights : {0}'.format(bin_clf.coef_.shape))
print('Bias : {0}'.format(bin_clf.intercept_))
print('Loss function : {0}'.format(bin_clf.loss_function_))

"""Let us make predictions on the training set and then calculate the training accuracy."""

y_hat_train_0 = bin_clf.predict(X_train)
print('Training Accuracy :', bin_clf.score(X_train,y_train_0))

"""Let us make the predictions on the test set and then calculate the testing accuracy."""

print('Test accuracy :',bin_clf.score(X_test,y_test_0))

"""#### **Displaying Predictions**
* Take few images from the test-set at random and display it with the corresponding predictions.

* Plot a few images in a single figure window along with their respective **Predictions**.
"""

y_hat_test_0 = bin_clf.predict(X_test)

num_images = 9
factor = int(np.sqrt(num_images))
fig,ax = plt.subplots(nrows=factor, ncols = factor, figsize=(8,6))
idx_offset  = 0

for i in range(factor):
    index = idx_offset + i*(factor)
    for j in range(factor):
        ax[i,j].imshow(X_test[index+j].reshape(28,28),cmap='gray')
        ax[i,j].set_title('Prediction: {0}'.format(str(y_hat_test_0[index+j])))
        ax[i,j].set_axis_off()

indx_0  = np.where(y_test_0==1)

zeroImgs = X_test[indx_0[0]]
zeroLabls = y_hat_test_0[indx_0[0]]

num_images = 9
factor = int(np.sqrt(num_images))
fig, ax = plt.subplots(nrows=factor, ncols=factor, figsize=(8,6))
idx_offset = 0

for i in range(factor):
    index = idx_offset + i*(factor)
    for j in range(factor):
        ax[i,j].imshow(zeroImgs[index+j].reshape(28,28),cmap='gray')
        ax[i,j].set_title('Prediction : {0}'.format(str(zeroLabls[index+j])))
        ax[i,j].set_axis_off()

"""It seems that there are a significant number of images that are correctly classified."""

num_misclassified = np.count_nonzero(zeroLabls == -1)
num_correctclassified = len(zeroLabls) - num_misclassified

accuracy = num_correctclassified / len(zeroLabls)
print(accuracy)

"""* This above score is less than the accuracy score of the model but it seems preety descent.

* Will it be the same if we consider another digit, say, 5 for positive class and all other class as negative. Of course not.

#### **Better Evaluation metrics**
* We now know that using the accuracy **alone** to measure the performance of the model is not suitable (especially for imbalanced datasets).

##### **1. Confusion Matrix**
"""

y_hat_train_0 = bin_clf.predict(X_train)

cm_display = ConfusionMatrixDisplay.from_predictions(
    y_train_0, y_hat_train_0, values_format='.5g', display_labels=bin_clf.classes_)
plt.show()

"""* Pay attention to the number of FPs and FNs. Suppose for some reasons, we want the classifer to avoid FPs to a good extent irrespective of FNs, how can we acheive it.

* To answer it, let's compute the other metrics which take FPs and FNs into account.

##### **2. Precision & Recall**

We can use the function `classification_report` to compute these parameters.

However, for the time being let's compute these parameters using the data from the confusion matrix manually.
"""

cf_matrix = cm_display.confusion_matrix

tn = cf_matrix[0,0]
fn = cf_matrix[1,0]
fp = cf_matrix[0,1]
tp = cf_matrix[1,1]

precision = tp/(tp+fp)
print('Precision : ', precision)

recall = tp/(tp+fn)
print('Recall : ', recall)

accuracy = (tn+tp)/(tn+tp+fn+fp)
print('Accuracy : ', accuracy)

"""* Precision is close to 0.98. Despite it, we still want to increase the precision.

* In general, we would like to know whether the model under consideration with the set hyper-parameters is a good one for a given problem.

##### **Cross validation (CV)**

* Well to address this, we have to use cross-validation folds and measure the same metrics across these folds for different values of hyperparameters.

* However, perceptron doesn't have many hyperparameters other than the learning rate.

* For the moment, we set the learning rate to its default value. Later, we will use `GridSearchCV` to find the better value for the learning rate.

**Generalization**
"""

bin_clf = Perceptron(max_iter=100, random_state=1729)

scores = cross_validate(bin_clf, X_train, y_train_0, cv=5, scoring=[
                        'precision', 'recall', 'f1'], return_estimator=True)
print(scores)

"""**NOTE :**

The perceptron estimator passed as an argument to the function `cross_validate` is internally cloned `num_fold (cv=5)` times and fitted independently on each fold. (you can check this by setting `warm_start=True`)

Compute the average and standard deviation of scores for all three metrics on (k=5) folds to measure the generalization!.

"""

print('Precision : avg : {0:.2f},  std : {1:.2f}'.format(
    scores['test_precision'].mean(), scores['test_precision'].std()))

print()
print('Recall : avg : {0:.2f},  std : {1:.2f}'.format(
    scores['test_recall'].mean(), scores['test_recall'].std()))

print()
print('F1 score : avg : {0:.2f},  std : {1:.3f}'.format(
    scores['test_f1'].mean(), scores['test_f1'].std()))

"""* Let us pick the first estimator returned by the cross-validate function.

* So, we can hope that the model might also perform well on test data.
"""

bin_clf = scores['estimator'][0]
y_hat_test_0 = bin_clf.predict(X_test)

cm_display = ConfusionMatrixDisplay.from_predictions(
    y_test_0, y_hat_test_0, values_format='.5g')

print('Precision : {0:.4f}'.format(precision_score(y_test_0, y_hat_test_0)))
print('Recall : {0:.4f}'.format(recall_score(y_test_0, y_hat_test_0)))

"""This is good !

Another way for '**Generalization**' (Optional)

* There is an **another approach** of getting predicted labels via cross-validation and using it to measure the generalization.

* In this case, each sample in the dataset will be part of only one test set in the splitted folds.
"""

y_hat_train_0 = cross_val_predict(bin_clf, X_train, y_train_0, cv=5)

cm_display = ConfusionMatrixDisplay.from_predictions(
    y_train_0, y_hat_train_0, values_format='.5g')

plt.show()

cf_matrix = cm_display.confusion_matrix
tn = cf_matrix[0,0]
fn = cf_matrix[1,0]
fp = cf_matrix[0,1]
tp = cf_matrix[1,1]

precision = tp/(tp+fp)
print('Precision : ', precision)

recall = tp/(tp+fn)
print('Recall : ', recall)

accuracy = (tn+tp)/(tn+tp+fn+fp)
print('Accuracy : ', accuracy)

"""Compare the precision and recall score obtained by the above method with that of the previous method (i.e. using `cross_validate`)

"""

print('Precision : {0:.4f}'.format(precision_score(y_train_0,y_hat_train_0)))
print('Recall : {0:.4f}'.format(recall_score(y_train_0,y_hat_train_0)))

"""Finally, we can print all these scores as a report using the `classification_report` function"""

print(classification_report(y_train_0,y_hat_train_0))

"""##### **3. Precision / Recall Tradeoff**

* Often time we need to make a trade off between precision and recall scores of a model.

* It depends on the problem at hand.

* It is important to note that we should **not** pass the **predicted labels** as input to `precision_recall_curve` function, instead we need to pass the probability scores or the output from the decision function!.

* The `Perceptron()` class contains a `decision_function` method, therefore we can make use of it.

* Then, internally the decision scores are sorted, **tps** and **fps** will be computed by changing the threshold from index[0] to index [-1].

* Let us compute the scores from decision function.
"""

bin_clf = Perceptron(random_state=1729)
bin_clf.fit(X_train, y_train_0)
y_scores = bin_clf.decision_function(X_train)

sns.histplot(np.sort(y_scores))
plt.show()

"""The reason for so many negative values than the positives is : **Class-Imbalance**.

* Suppose threshold takes the value of -600, then all the samples having score greater than -600 is set to 1 ( +ve label ) and less than it is set to -1 ( -ve label ).

* Therefore, the number of False Positives will be increased. This will in turn reduce the precision score to a greater extent.

* On the otherhand, if the threshold takes the value of say 400, Then, the number of False negatives will be increase and hence the recall will reduce to a greater extent.

"""

precisions, recalls, thresholds = precision_recall_curve(y_train_0,y_scores,pos_label=1)

plt.figure(figsize=(10, 6))
plt.plot(precisions[:-1], recalls[:-1], "g--")

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(thresholds, precisions[:-1], "b--", label='Precision')
plt.plot(thresholds, recalls[:-1], "g-", label='Recall')

plt.xlabel('Threshold')
plt.grid(True)
plt.legend(loc='best')
plt.show()

"""Getting the index of threshold around zero"""

idx_th = np.where(np.logical_and(thresholds >0, thresholds <1))
print('Precision for zero threshold : ',precisions[idx_th[0][0]])

"""* **The solution** to the question of how can we increase the precision of the classifier by compromising the recall is we can make use of the above plot.

##### **4. ROC Curve**
"""

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)
plt.figure(figsize=(10, 4))
plt.plot(fpr, tpr, linewidth=2, label='Perceptron')
plt.plot([0, 1], [0, 1], 'k--', label='baseEstimator')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend()
plt.show()

"""#### **Warm Start VS Cold Start**

##### **Cold Start**

* If we execute the `fit` method of `bin_clf` repeatedly, we get the same score for both training and testing accuracy.

* This because everytime the `fit` method is called, the model weights are initialized to the same values. Therefore, we obtain the same score.

* This is termed as **cold start**.

Let's execute the following cell 4 times and observe the score.
"""

bin_clf.fit(X_train, y_train_0)
y_hat_train_0 = bin_clf.predict(X_train)

print('Training Accuracy : ', bin_clf.score(X_train, y_train_0))
print('Test accuracy : ', bin_clf.score(X_test, y_test_0))

"""##### **Warm Start**

* Setting `warm_start=True` retains the weight values of the model after `max_iter` and hence produce different results for each execution.

* Warm starting is useful in many ways. It helps us train the model by initializing the weight values from the previous state. So, we can pause the training and resume it whenever we get the resource for computation.

* Of course, it is not required for simple models like perceptron and for a small dataset like **MNIST**.

* In this notebook, we use this feature to plot the iteratation vs loss curve.

Let us execute the following lines of code 4 times and observe how the training accuracy changes for each execution.
"""

bin_clf_warm = Perceptron(max_iter=100,random_state=1729,warm_start=True)

bin_clf_warm.fit(X_train,y_train_0)
print('Training Accuracy : ', bin_clf_warm.score(X_train,y_train_0))

"""## **Multiclass Classification (OneVsAll)**

* We know that the perceptron is a binary classifier. However,
MNIST dataset contains 10 classes. So, we need to extend the idea to handle multi-class problem.

* **Solution** : Combining multiple binary classifiers and devise a suitable scoring metric.

* Sklearn makes it extremely easy without modifying a single line of code that we have written for the binary classifier.

* Sklearn does this by counting a number of unique elements (10 in this case) in the label vector `y_train` and converting labels using `Labelbinarizer` to fit each binary classifier.

"""

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelBinarizer

"""Let's use **Label binarizer** just to see the encoding."""

y_train_ovr = LabelBinarizer().fit_transform(y_train)

for i in range(10):
    print('{0} : {1}'.format(y_train[i],y_train_ovr[i]))

"""* The `y_train_ovr` will be of size of size $60000 \times 10$.

* The first column will be (binary) label vector for 0-detector and the next one for 1-Detector and so on.
"""

clf = Perceptron(random_state=1729)
clf.fit(X_train,y_train)

"""* What had actually happened internally was that the API automatically created 10 binary classifiers, converted labels to binary sparse matrix and trained them with the binarized labels.

* During the inference time, the input will be passed through all these 10 classifiers and the highest score among the output from the classifiers will be considered as the predicted class.

* To see it in action, let us execute the following lines of code.
"""

print('Shape of Weight matrix : {0} and bias vector : {1}'.format(
    clf.coef_.shape, clf.intercept_.shape))

"""* So it is a matrix of size $ 10 \times 784 $, where each row represents the weights for a single binary classifier.

* Important difference to note is that there is no signum function associated with the perceptron.

* The class of a perceptron that outputs the maximum score for the input sample is considered as the predicted class.
"""

for i in range(10):
    scores = clf.decision_function(X_train[i].reshape(1, -1))
    print(scores)
    print()
    print('The predicted class : ', np.argmax(scores))
    print()
    print('Predicted output : ')
    print(clf.predict(X_train[i].reshape(1, -1)))
    print('-'*20)

"""Get the prediction for all training samples."""

y_hat = clf.predict(X_train)

"""Lets display the classification report."""

print(classification_report(y_train,y_hat))

"""Now let us display the confusion matrix and relate it with the report above."""

cm_display = ConfusionMatrixDisplay.from_predictions(
    y_train, y_hat, values_format='.5g')

"""#### **Making a Pipeline**

* Let's create a pipeline to keep the code compact.

* Recall that, the **MNIST** dataset is clean and hence doesn't require much preprocessing.

* The one potential preprocessing technique we may use is to scale the features within the range(0,1).

* It is **not** similar to scaling down the range values between 0 and 1.
"""

# create a list with named tuples
estimators = [('scaler', MinMaxScaler()), ('bin_clf', Perceptron())]
pipe = Pipeline(estimators)

pipe.fit(X_train,y_train_0)

y_hat_train_0 = pipe.predict(X_train)

cm_display = ConfusionMatrixDisplay.from_predictions(
    y_train_0, y_hat_train_0, values_format='.5g')
plt.show()

"""#### **Iteration vs Loss Curve**

The other way of plotting **Iteration Vs Loss Curve** with the `Partial_fit` method.
"""

iter = 100
bin_clf1 = Perceptron(max_iter=100,random_state=2094)
loss_clf1=[]

for i in range(iter):
    bin_clf1.partial_fit(X_train,y_train_0,classes=np.array([1,-1]))
    y_hat_0 = bin_clf1.decision_function(X_train)
    loss_clf1.append(hinge_loss(y_train_0,y_hat_0))

plt.figure()
plt.plot(np.arange(iter), loss_clf1)

plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.show()

"""#### **GridSearchCV**

* So, far we didn't perform any hyperparameter tuning & just accepted the default value for learning rate of the Perceptron class.

* Now, let us search for a better learning rate using `GridSearchCV`.

* No matter what the learning rate is, the loss will never converge to zero as the classes are not linearly separable.
"""

from sklearn.metrics import make_scorer

scoring = make_scorer(hinge_loss,greater_is_better=False)
lr_grid = [1/2**n for n in range(1,6)]

bin_clf_gscv = GridSearchCV(Perceptron(), param_grid={'eta0':lr_grid},scoring=scoring, cv=5)
bin_clf_gscv.fit(X_train,y_train_0)

bin_clf_gscv.cv_results_

"""Well, instead of instantiating a Perceptron class with a new learning rate and re-train the model, we could simply get the `best_estimator` from `GridSearchCV` as follows."""

best_bin_clf = bin_clf_gscv.best_estimator_
best_bin_clf

"""We can observe that the best learning rate is **0.125**."""

iter = 100
loss = []
best_bin_clf = Perceptron(max_iter=1000,random_state=2094,eta0=0.125)

for i in range(iter):
    best_bin_clf.partial_fit(X_train, y_train_0, classes=np.array([1,-1]))
    y_hat_0 = best_bin_clf.decision_function(X_train)
    loss.append(hinge_loss(y_train_0,y_hat_0))

plt.figure()
plt.plot(np.arange(iter), loss_clf1, label='eta0=1')
plt.plot(np.arange(iter), loss, label='eta0=0.125')

plt.grid(True)
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.show()

y_hat_train_0 = bin_clf.predict(X_train)
print(classification_report(y_train_0, y_hat_train_0))

"""Now, compare this classification report with the one when **eta0 = 1**

#### **Visualizing weight vectors** (Optional)

It will be interesting to look into the samples which are misclassified as False Positives (that is, images that are not zero but classified as zero).
"""

# repeating the code for readability
bin_clf = Perceptron(max_iter=100)
bin_clf.fit(X_train, y_train_0)
y_hat_train_0 = bin_clf.predict(X_train)

# index of true -ve samples
idx_n = np.where(y_train_0==-1)

# index of predicted positive samples
idx_pred_p = np.where(y_hat_train_0==1)

# index of predicted negative samples
idx_pred_n = np.where(y_hat_train_0==-1)

idx_fp = np.intersect1d(idx_n, idx_pred_p)
idx_tn = np.intersect1d(idx_n,idx_pred_p)

fig, ax = plt.subplots(nrows=factor, ncols=factor, figsize=(8,6))
idx_offset = 0

for i in range(3):
    index = idx_offset + i
    for j in range(3):
        ax[i,j].imshow(X_train[idx_fp[index+j]].reshape(28,28),cmap='gray')

        # we should not use x_train_with_dummy

        # GT : ground truth ; Pred : predicted
        ax[i,j].set_title('GT : {0}, Pred : {1}'.format(str(y_train_0[idx_fp[index+j]]),str(y_hat_train_0[idx_fp[index+j]])))

        ax[i,j].set_axis_off()

from matplotlib.colors import Normalize

w = bin_clf.coef_
w_matrix = w.reshape(28, 28)
#fig = plt.figure()
#plt.imshow(w_matrix, cmap='magma')
#plt.imshow(w_matrix, cmap='cividis')
#plt.imshow(w_matrix, cmap='viridis')
#plt.imshow(w_matrix, cmap='gray')
plt.imshow(w_matrix, cmap='inferno')

#plt.axis(False)
plt.rcParams['axes.grid'] = False
plt.colorbar()
plt.show()

#print(idx_fp.shape)

activation = w * X_train[idx_fp[0]].reshape(1, -1)
lin_out = activation.reshape(28, 28)
plt.subplot(1, 2, 1)
plt.imshow(X_train[idx_fp[0]].reshape(28, 28), cmap='gray')
plt.colorbar()

#lin_out[lin_out < 0]=0 # just set the value less than zero to zero

plt.subplot(1, 2, 2)
plt.imshow(lin_out, cmap='gray')
plt.colorbar()
plt.grid(False)
plt.axis(False)
plt.show()

"""Input to the signum"""

print(np.sum(lin_out)+bin_clf.intercept_)

activation = w*(X_train[idx_tn[0]].reshape(1, -1))
lin_out = activation.reshape(28, 28)

plt.subplot(1, 2, 1)
plt.imshow(X_train[idx_tn[0]].reshape(28, 28), cmap='gray')
plt.colorbar()

# just set the value less than zero to zero
lin_out[lin_out < 0] = 0
plt.subplot(1, 2, 2)
plt.imshow(lin_out, cmap='gray')

plt.colorbar()
plt.grid(False)
plt.axis(False)
plt.show()

"""Input to the signum"""

print(np.sum(lin_out) + bin_clf.intercept_)

"""### **Objective**

In this notebook we will solve the same problem of recognizing Handwritten digits using Logistic regression model.

### **Imports**
"""

# Commented out IPython magic to ensure Python compatibility.
# Common imports
import numpy as np
from pprint import pprint

# to make this notebook's output stable across runs
np.random.seed(42)

# sklearn specific imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model  import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.model_selection import cross_validate, RandomizedSearchCV, cross_val_predict

# log loss is also known as cross entropy loss
from sklearn.metrics import log_loss
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score

#scipy
from scipy.stats import loguniform

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

# global settings
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
mpl.rc('figure',figsize=(8,6))

# Ignore all warnings (convergence..) by sklearn
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

"""## **Handwritten Digit Classification**

* We are going to use **LogisticRegression** (despite it's name) to classify a given digit image. Again, we first apply the model for binary classification and then extend it to multiclass classification.

* Suppose we want to recognize whether the given image is of digit zero or not (digits other than zero). Then the problem could be case as binary classification problem.

* The first step is to create a dataset that contains collection of digit images (also called examples, samples) written by humans. Then each image should be labelled properly.

* Fortunately, we have a standard benchmark dataset called **MNIST**.
"""

from sklearn.datasets import fetch_openml

# it returns the data and labels as a panda dataframe.
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

"""The data matrix $X$ and the respective label vector $y$ need to be converted to numpy array by calling a `to_numpy` method."""

X = X.to_numpy()
y = y.to_numpy()

"""#### **Preprocessing**

* Unlike perceptron, where scaling the range is optional(but recommended), sigmoid requires range between 0 to 1.

* Contemplate the consequence if we don't apply the scaling operation on the input datapoints.

* **NOTE** : **Do not** apply mean centering as it removes zeros from the data, however zeros should be zeros in the dataset.

* Since we are using only one preprocessing step, using `pipeline` may not be required.
"""

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print('Mean of the features : ', np.mean(X))
print('Standard Deviation : ', np.std(X))
print('Minimum value : ', np.min(X))
print('Maximum value : ', np.max(X))

"""Let's get some information about the dataset.

"""

print('Number of targets : {0} ,type : {1}'.format(X.shape[0] ,X.dtype))
print('Number of features : {0}'.format(X.shape[1]))
print()
print('Number of classes : {0} ,type : {1}'.format(len(np.unique(y)) ,y.dtype))
print('Labels : {0}'.format(np.unique(y)))

"""Note that the labels are of string data type.

#### **Data visualization**
"""

num_images = 9
factor = int(np.sqrt(num_images))
fig,ax = plt.subplots(nrows=factor, ncols = factor, figsize=(8,6))
idx_offset  = 0

for i in range(factor):
    index = idx_offset + i*(factor)
    for j in range(factor):
        ax[i,j].imshow(X[index+j].reshape(28,28),cmap='gray')
        ax[i,j].set_title('Label : {0}'.format(str(y[index+j])))
        ax[i,j].set_axis_off()

"""#### **Data splitting**"""

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

"""Before proceeding further, we need to check whether the datasset is balanced or imbalanced.

We can do it by plotting the distribution of samples in each classes.
"""

plt.figure(figsize=(10,5))
sns.histplot(data=np.int8(y_train) ,binwidth=0.45 ,bins=11)

plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9] ,label=[0,1,2,3,4,5,6,7,8,9])
plt.xlabel('Class')
plt.title('Distribution of samples')
plt.show()

"""## **Binary Classification : 0 - Detector**

* Let us start with a simple classification problem, that is, binary classification.

* Since the original label vector contains 10 classes, we need to modify the number of classes to 2. Therefore, the label '0' will be changed to '1' and all other labels(1-9) will be changed to '0'

* **NOTE: For perceptron we set the negative labels to -1**
"""

# initialize new variables names with all 0.
y_train_0 = np.zeros((len(y_train)))
y_test_0 = np.zeros((len(y_test)))

# find indices of digit 0 image
# remember original labels are of type str not int
indx_0 = np.where(y_train=='0')

# use those indices to modify y_train_0 & y_test_0
y_train_0[indx_0] = 1
indx_0 = np.where(y_test == '0')
y_test_0[indx_0] = 1

"""#### **Visualization of new variables**"""

num_images = 9
factor = np.int(np.sqrt(num_images))

fig, ax = plt.subplots(nrows=factor, ncols=factor,figsize=(8,6))
idx_offset = 0

for i in range(factor):
    index = idx_offset+ i*(factor)
    for j in range(factor):
        ax[i,j].imshow(X[index+j].reshape(28,28), cmap='gray')
        ax[i,j].set_title('Label : {0}'.format(str(y_train_0[index+j])))
        ax[i,j].set_axis_off()

"""## **Model**

### **Baseline Models**

Let us quickly construct a baseline model with the following rule :

1. Count number of samples per class.

2. The model **always output** the class which has highest number of samples.

3. Then calculate the accuracy of the baseline model.
"""

num_pos = len(np.where(y_train_0 == 1)[0])
num_neg = len(np.where(y_train_0 == 0)[0])

print(num_pos)
print(num_neg)

base_clf = DummyClassifier(strategy='most_frequent')
base_clf.fit(X_train,y_train_0)

print(base_clf.score(X_train,y_train_0))

"""Now the reason is obvious. The model would have predicted 54077 samples correctly just by outputing 0 for all the input samples.

Therefore the accuracy will be $\frac{54077}{60000} = 90.12 \%$

### **Logistic Regression model**

Quick recap of various components in the general settings:


1. **Training data** :

    * consists of features & labels or $(\mathbf X,y)$

    * Here, $y$ is a **discrete** number from a finite set.

    * **Features** in this case are **pixel** values of an image.
2. **Model** :
$$ z = w_0x_0 + w_1x_1+ \ldots + w_mx_m$$

$$ = \mathbf w^{T} \mathbf x$$

and passing it through the sigmoid non-linear function (or Logistic function)

$$ \sigma(z)=\frac{1}{1+e^{-z}}$$

3. **Loss function**:

\begin{equation}
J(\mathbf w) = -\frac{1}{n} \mathbf \sum [y^{(i)} \log(h_w(\mathbf x^{(i)}))+(1-y^{(i)})(1-\log(h_w(\mathbf x^{(i)})))]
\end{equation}

4. **Optimization**:


Let's quickly take a look into the important parameters of the SGDClassifier() estimator:

**class sklearn.linear_model.SGDClassifier** **`(loss='hinge', * ,penalty='l2', alpha=0.0001, l1_ratio = 0.15, fit_intercept =True, max_iter =1000, tol=0.001, shuffle=True, verbose =0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate = 'optimal', eta0=0.0, power_t = 0.5, early_stopping = False, validation_fraction =0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)`**.


* **IMPORTANT** :
**Setting the loss parameter to `loss=log` makes it a logistic regression classifier**. We may refer to documentation for more details on the `SGDClassifier` class.

* Create an instant of binary classifier (**bin_sgd_clf**) and call the `fit` method to train the model.

* Let's use fit method of `SGDClassifier()` to plot the iteration vs loss curve. (also we could use `partial_fit()` method )

* Therefore, to capture the loss for each iterations during training we set the parameters `warm_start =True` and `max_iter=1`

#### **Training without regularization**

Set `eta0 = 0.01,learning_rate = 'constant' ` and `alpha = 0`.
"""

bin_sgd_clf = SGDClassifier(loss='log',
                            penalty='l2',
                            warm_start=True,
                            eta0=0.01,
                            alpha=0,
                            learning_rate='constant',
                            random_state=1729)

loss = []
iter = 100

for i in range(iter):
    bin_sgd_clf.fit(X_train, y_train_0)
    y_pred = bin_sgd_clf.predict_proba(X_train)
    loss.append(log_loss(y_train_0, y_pred))

plt.figure()
plt.plot(np.arange(iter), loss)
plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Label')
plt.show()

"""Let us calculate the training and testing accuracy of the model."""

print('Training accuracy : {0:.4f}'.format(bin_sgd_clf.score(X_train,y_train_0)))
print('Testing accuracy : {0:.4f}'.format(bin_sgd_clf.score(X_test,y_test_0)))

"""We know that accuracy alone is not a good metric for binary classification.

So let's compute Precision, recall and f1-score for the model.
"""

y_hat_train_0 = bin_sgd_clf.predict(X_train)

cm_display = ConfusionMatrixDisplay.from_predictions(
    y_train_0, y_hat_train_0, values_format='.5g')
plt.show()

print(classification_report(y_train_0,y_hat_train_0))

"""##### **Cross Validation**"""

estimator = SGDClassifier(loss='log',
                          penalty='l2',
                          max_iter=100,
                          warm_start=False,
                          eta0=0.01,
                          alpha=0,
                          learning_rate='constant',
                          random_state=1729)

cv_bin_clf = cross_validate(estimator, X_train, y_train_0, cv=5,
                            scoring=['precision', 'recall', 'f1'],
                            return_train_score=True,
                            return_estimator=True)
cv_bin_clf

"""* From the above result, we can see that **logistic regression is better than the perceptron**.

* However, it is good to check the weight values of all the features and decide whether regularization could be of any help.

"""

weights = bin_sgd_clf.coef_
bias = bin_sgd_clf.intercept_

print('Bias :', bias)
print('Shape of weights :', weights.shape)
print('Shape of bias :', bias.shape)

plt.figure()
plt.imshow(weights.reshape(28, 28), cmap='inferno')

plt.grid(False)
plt.colorbar()
plt.show()

plt.figure()
plt.plot(np.arange(0,784),weights[0,:])

plt.ylim(np.min(weights[0])-5,np.max(weights[0])+5)
plt.grid(True)

plt.xlabel('Feature Index')
plt.ylabel('Weight value')
plt.show()

"""* It is interesting to observe that how many weight values are exactly zero.

* Those features contribute nothing in the classification.
"""

zero_weight_idx = np.where(weights[0]==0)
print(len(zero_weight_idx[0]))

# num_zero_w = weights.shape[1]-np.count_nonzero(weights)
# print("Number of weights with value zero".format(num_zero_w))

"""From the above plot, it is also obvious that regularization is not required.

#### **Training with regularization**

However, what happens to the performance of the model if we penalize, out of temptation, the weight values even to a smaller degree.
"""

bin_sgd_clf_l2 = SGDClassifier(loss='log',
                               penalty='l2',
                               eta0=0.01,
                               alpha=0.001,
                               max_iter=1,
                               warm_start=True,
                               learning_rate='constant',
                               random_state=1729
                               )

loss = []
iter =100

for i in range(iter):
  bin_sgd_clf_l2.fit(X_train, y_train_0)
  y_pred = bin_sgd_clf_l2.predict_proba(X_train)
  loss.append(log_loss(y_train_0,y_pred))

plt.figure()
plt.plot(np.arange(iter), loss)

plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

"""Let us calculate training and testing accuracy."""

print('Training accuracy : {0:.4f}'.format(bin_sgd_clf_l2.score(X_train,y_train_0)))

print('Testing accuracy : {0:.4f}'.format(bin_sgd_clf_l2.score(X_test,y_test_0)))

"""Let's compute Precision, recall and f1-score for the model."""

y_hat_train_0 = bin_sgd_clf_l2.predict(X_train)
cm_display = ConfusionMatrixDisplay.from_predictions(y_train_0,y_hat_train_0,values_format='.5g')

print(classification_report(y_train_0,y_hat_train_0))

weights = bin_sgd_clf_l2.coef_
bias = bin_sgd_clf_l2.intercept_

print('Bias :', bias)
print('Shape of weights :', weights.shape)
print('Shape of bias :', bias.shape)

plt.figure()
plt.plot(np.arange(0, 784), weights[0, :])

plt.ylim(np.min(weights[0]-3), np.max(weights[0])+3)
plt.xlabel('Feature Index')
plt.ylabel('Weight Value')
plt.grid(True)
plt.show()

"""Zero weights calculation

**Note**: Zero weights can't contribute to features.
"""

num_zero_w = len(np.where(weights == 0)[0])
print('Number of zero weight count:', num_zero_w)

"""#### **Displaying input image and its prediction**"""

index = 7  # try some other index
plt.imshow(X_test[index, :].reshape(28, 28), cmap='plasma')
plt.colorbar()
pred = bin_sgd_clf_l2.predict(X_test[index].reshape(1, -1))
plt.title(str(pred))
plt.show()

"""Let's plot a few images and their respective predictions with SGDClassifier without regularization."""

y_hat_test_0 = bin_sgd_clf.predict(X_test)

num_images = 9
factor = np.int(np.sqrt(num_images))
fig,ax = plt.subplots(nrows=factor, ncols = factor, figsize=(8,6))
idx_offset  = 0

for i in range(factor):
    index = idx_offset + i*(factor)
    for j in range(factor):
        ax[i,j].imshow(X_test[index+j].reshape(28,28),cmap='plasma')

        ax[i,j].set_title('Prediction : {0}'.format(str(y_hat_test_0[index+j])))
        ax[i,j].set_axis_off()

indx_0 = np.where(y_test_0 == 1)

zeroImgs= X_test[indx_0[0]]
zeroLabls = y_hat_test_0[indx_0[0]]

num_images = 9
factor = np.int(np.sqrt(num_images))
fig,ax = plt.subplots(nrows=factor, ncols = factor, figsize=(8,6))
idx_offset  = 0

for i in range(factor):
    index = idx_offset + i*(factor)

    for j in range(factor):
        ax[i,j].imshow(zeroImgs[index+j].reshape(28,28),cmap='plasma')
        ax[i,j].set_title('Prediction : {0}'.format(str(zeroLabls[index+j])))
        ax[i,j].set_axis_off()

"""#### **Hyperparameter Tuning**

* We have to use `cross-validate` folds and mesure the same metrics across these folds for different values of hyper-parameters.

* Logistic regression uses **SGD** solver and hence the two important hyperparameters include :
    * **learning rate**

    * **regularization rate**

* For the moment, we skip penalizing the parameters of the model and just search for a better learning rate using `RandomizedSearchCV()` and draw the value from the uniform distribution.

"""

lr_grid = loguniform(1e-2,1e-1)

"""* **Note**:  `lr_grid` is an object that contains a method called `rvs()`, which can be used to get the samples of given size.

* Therefore, we pass this `lr_grid` object to `RandomizedSearchCV()`. Internally, it makes use of this `rvs()` method for sampling.
"""

print(lr_grid.rvs(3,random_state=42))

estimator = SGDClassifier(loss='log',
                          penalty='l2',
                          max_iter=1,
                          warm_start=True,
                          eta0=0.01,
                          alpha=0,
                          learning_rate='constant',
                          random_state=1729)

scores = RandomizedSearchCV(estimator,
                            param_distributions={'eta0': lr_grid},
                            cv=5,
                            scoring=['precision', 'recall', 'f1'],
                            n_iter=5,
                            refit='f1')

scores.fit(X_train,y_train_0)

scores.cv_results_

"""Let us pick the best estimator from the results"""

best_bin_clf = scores.best_estimator_
best_bin_clf

y_hat_train_best_0 = best_bin_clf.predict(X_train)
print(classification_report(y_train_0, y_hat_train_best_0))

"""#### **Other Evaluation metrics**

##### **1. Precision / Recall Tradeoff**
"""

y_scores = bin_sgd_clf.decision_function(X_train)
precisions, recalls, thresholds = precision_recall_curve(y_train_0,y_scores)

plt.figure(figsize=(10,4))
plt.plot(thresholds,precisions[:-1],'r--',label='precisions')
plt.plot(thresholds,recalls[:-1],'b-',label='recalls')

plt.title('Precision / Recall Tradeoff' ,fontsize=16)
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('thresholds')
plt.show()

"""##### **2. Precision Recall Curve**"""

plt.figure(figsize=(10, 4))
plt.plot(recalls[:-1], precisions[:-1], 'b-')

plt.title('Precision Recall Curve', fontsize=16)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()

"""##### **3. ROC curve**"""

fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)

plt.figure(figsize=(10, 4))
plt.plot(fpr, tpr, linewidth=2, label='Perceptron')
plt.plot([0, 1], [0, 1], 'k--', label='Best_estimator')

plt.title('ROC Curve', fontsize=16)
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.grid(True)
plt.legend()
plt.show()

"""##### **4. ROC-AUC score**"""

auc = roc_auc_score(y_train_0, y_scores)
print('AUC : {0:.6f}'.format(auc))

"""## **Classsification using Ridge Classifier**

* Ridge Classifier casts the problem as the **least-squares classification** & finds the optimal weight using some matrix decompostion technique such as **Singular-Value Decompostion (SVD)**.

* To train the ridge classifier, the labels should be $y ∈ {+1 ,-1}$.

* The classifer also by default implements **L2 regularization**. However, we first implement it without regularization by setting `alpha = 0`

#### **Importing new libraries**
"""

# Commented out IPython magic to ensure Python compatibility.
# Common imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import loguniform
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import log_loss

from sklearn.model_selection import cross_validate, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from pprint import pprint

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib inline
sns.set()

# global settings
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rc('figure', figsize=(8, 6))

import warnings
warnings.filterwarnings('ignore')

"""#### **Getting Data**"""

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

"""#### **Data Preprocessing and Splitting**"""

X = X.to_numpy()
y = y.to_numpy()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# initialize new variables names with all -1.
y_train_0 = np.ones((len(y_train)))
y_test_0 = np.ones((len(y_test)))

# find indices of digit 0 image
# remember original labels are of type str not int
indx_0 = np.where(y_train == '0')

# use those indices to modify y_train_0 & y_test_0
y_train_0[indx_0] = 1
indx_0 = np.where(y_test == '0')
y_test_0[indx_0] = 1

"""#### **Model Building**

First taking a look into the parameters of the class :

**RidgeClassifier** (
    `alpha=1.0,
    *,
    fit_intercept=True,
    normalize='deprecated',
    copy_X=True,
    max_iter=None,
    tol=0.001,
    class_weight=None,
    solver='auto',
    positive=False,
    random_state=None,`
)

**Note :** The parameter `normalize` is depreciated.
"""

estimator = RidgeClassifier(normalize=False ,alpha=0)

pipe_ridge = make_pipeline(MinMaxScaler() ,estimator)
pipe_ridge.fit(X_train ,y_train_0)

"""**Checking on performance of model**"""

y_hat_test_0 = pipe_ridge.predict(X_test)
print(classification_report(y_test_0, y_hat_test_0))

"""#### **Cross Validation**"""

cv_ridge_clf = cross_validate(
                            pipe_ridge,
                            X_train ,y_train_0 ,cv=5,
                            scoring=['precision' ,'recall', 'f1'],
                            return_train_score=True ,
                            return_estimator=True)

pprint(cv_ridge_clf)

"""Best estimator ID"""

best_estimator_id = np.argmax(cv_ridge_clf['train_f1'])
best_estimator_id

"""Best Estimator"""

best_estimator = cv_ridge_clf['estimator'][best_estimator_id]
best_estimator

"""Lets evaluate the performance of the best classsifier on the test set."""

y_hat_test_0 = best_estimator.predict(X_test)
print(classification_report(y_test_0 ,y_hat_test_0))

"""#### **Further Exploration**

Let's see what these classifiers learnt about the digit 0.
"""

# models = (pipe_sgd ,pipe_sgd_l2 ,pipe_logit ,pipe_ridge)
# titles = ('SGD' ,'Regularized SGD', 'Logit' ,'Ridge')

# plt.figure(figsize=(5,5))
# for i in range(0,4):
#     w = models[i][1].coef_
#     w_matrix = w.reshape(28,28)
#     plt.subplot(2,2,i+1)
#     plt.imshow(w_matrix ,cmap='gray')
#     plt.title(titles[i])
#     plt.axis('off')
#     plt.grid(False)
# plt.show()

"""## **Multiclass Classifier (OneVsAll)**

In this notenook, we will implement multiclass classification using LogisticRegression with both :

* **SGD** i.e. SGDClassifier(loss='log')

* **solvers** i.e. LogisticRegression(solver='lbfgs')

### **Imports**
"""

# Commented out IPython magic to ensure Python compatibility.
# Common imports
import numpy as np
from pprint import pprint

# to make this notebook's output stable across runs
np.random.seed(42)

# sklearn specific imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model  import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.model_selection import cross_validate, RandomizedSearchCV, cross_val_predict

from sklearn.metrics import log_loss
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score

#scipy
from scipy.stats import loguniform

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

# global settings
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
mpl.rc('figure',figsize=(8,6))

import warnings
warnings.filterwarnings('ignore')

"""#### **Getting Data**"""

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

"""#### **Data Preprocessing and Splitting**"""

X = X.to_numpy()
y = y.to_numpy()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

"""### **Multiclass LogisticRegression using SGDClassifier**

#### **Model Building**
"""

estimator = SGDClassifier(loss='log', penalty="l2", alpha=0, max_iter=1, random_state=1729, learning_rate="constant", eta0=0.01 ,warm_start=True)

pipe_sgd_ova = make_pipeline(MinMaxScaler() ,estimator)

loss = []
iter = 100

for i in range(iter):
    pipe_sgd_ova.fit(X_train ,y_train)

    y_pred = pipe_sgd_ova.predict_proba(X_train)
    loss.append(log_loss(y_train ,y_pred))

"""Visualization of Loss VS iterations"""

plt.figure()
plt.plot(np.arange(iter), loss)
plt.grid(True)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

"""What happened behind the screen is that the library automatically created 10 binary classifiers and trained them.

During the inference time, the input will be passed through all the 10 classifiers and the highest score among the outputs will be considered as the predicted class.

To see it in action, let's execute the following lines of code :
"""

pipe_sgd_ova[1]

pipe_sgd_ova[1].coef_

pipe_sgd_ova[1].coef_.shape

"""So it is a matrix of size $10$ X $784$ . A row represents the weights of a single binary classifier."""

y_hat = pipe_sgd_ova.predict(X_test)
y_hat[:5]

"""#### **Evaluating Metrics**"""

cm_display = ConfusionMatrixDisplay.from_predictions(y_test ,y_hat, values_format='.5g')

plt.show()

print(classification_report(y_test ,y_hat))

"""### **Multiclass LogisticRegression using solvers**

#### **Model Building**
"""

pipe_logreg_ova = make_pipeline(MinMaxScaler() ,LogisticRegression(solver='lbfgs' ,C=np.infty ,random_state=1729))

pipe_logreg_ova.fit(X_train ,y_train)

"""#### **Making predictions**"""

y_hat = pipe_logreg_ova.predict(X_test)

"""#### **Evaluating Metrics**"""

cm_display = ConfusionMatrixDisplay.from_predictions(y_test ,y_hat ,values_format='.5g')

plt.show()

print(classification_report(y_test ,y_hat))

"""### **Visualize weight values**"""

w = pipe_logreg_ova[1].coef_

# normalization
w = MinMaxScaler().fit_transform(w)
fig, ax = plt.subplots(3,3)
index = 1

for i in range(3):
    for j in range(3):
        ax[i][j].imshow(w[index, :].reshape(28,28) ,cmap='gray')
        ax[i][j].set_title('w{0}'.format(index))
        ax[i][j].set_axis_off()
        index += 1

"""## **Text Classification using Naive Bayes classifier**

In this notebook, we will use Niave Bayes classifier for classifying text.

Naive bayes is used for text classification & spam detection tasks.

Here is an example as to how to perform the text classification with Naive Bayes Classifier.
"""

import numpy as np

# data loading
from sklearn.datasets import fetch_20newsgroups

# preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

# model / estimator
from sklearn.naive_bayes import MultinomialNB

# pipeline utilty
from sklearn.pipeline import Pipeline

# model evaluation
from sklearn.metrics import ConfusionMatrixDisplay

# plotting
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

"""### **Getting dataset**
We will be using 20 newsgroup dataset for classification.

As a first step, let's download 20 newsgroup dataset with `fetch_20newsgroup` API.
"""

data = fetch_20newsgroups()

"""Lets look at the name of the classes."""

data.target_names

"""There are **20 categories** in the dataset. For simplicity, we will select **4** of these categories and download their training and test set."""

categories = ['talk.religion.misc',
              'soc.religion.christian', 'sci.space', 'comp.graphics']

train = fetch_20newsgroups(subset='train' ,categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

"""Lets look at a sample training document :"""

print(train.data[1])

"""This data is different than what we have seen so far. Here the training data contains document in text form.

### **Data Preprocessing and Modeling**

As we have mentioned in the first week of MLT, we need to convert th text data to numeric form.

* `TfidfVectorizer` is one such API that converts text input into a vector of numerical values.

* We will use `TfidfVectorizer` as as preprocessing step to obtain feature vector corresponding to the text document.

* We will be using `MultinomialNB` classifier for categorizing documents from 20 newsgroup corpus.
"""

from sklearn.pipeline import make_pipeline
model = make_pipeline(TfidfVectorizer() ,MultinomialNB())

"""Lets train the model."""

model.fit(train.data ,train.target)

"""### **Model Evaluation**
Lets first predict the labels for the test set and then calculate the confusion matrix for th test set.
"""

ConfusionMatrixDisplay.from_estimator(model, test.data ,test.target ,display_labels=test.target_names ,xticks_rotation='vertical')

plt.show()

"""**Obsereve that** :  

* There is a confusion between the documents of class `soc.religion.christian` and `talk.religion.misc` ,which is along the expected lines.

* The classes `comp.graphics` and `sci.space` are well separated by such a simple classifier.

Now we have the tool to classify statements into one of these four classes.

* Make use of `predict` function on pipeline for predicting category of a test string.
"""

def predict_category(s, train=train ,model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

"""Using the function of prediction :"""

predict_category('sending a payload to the ISS')

predict_category('what is your screen resolution')

predict_category('the Seven Sacraments are')

predict_category('discussing islam')

"""Here we can observe the confusion between the classes of `soc.religion.christian` and `talk.religion.misc` mentioned previously.

## **Softmax Regression on MNIST**

The objective of this notebook is to demonstrate **softmax regression in classification task**.

We make use of **MNIST** dataset for multiclass classification of images into digits they represent.

#### **Importing Libraries**
"""

# Commented out IPython magic to ensure Python compatibility.
# Common imports
import numpy as np
from pprint import pprint

# to make this notebook's output stable across runs
np.random.seed(42)

# sklearn specific imports
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from sklearn.pipeline import make_pipeline ,Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model  import SGDClassifier, RidgeClassifier, LogisticRegression ,LogisticRegressionCV
from sklearn.model_selection import cross_validate, RandomizedSearchCV, cross_val_predict

# log loss is also known as cross entropy loss
from sklearn.metrics import log_loss
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_curve ,make_scorer ,f1_score
from sklearn.metrics import roc_curve, roc_auc_score

#scipy
from scipy.stats import loguniform

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

# global settings
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
mpl.rc('figure',figsize=(8,6))

import warnings
warnings.filterwarnings('ignore')

"""#### **Data Loading**"""

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

"""It returns the data and labels as a panda dataframe.

#### **Data Splitting**
"""

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

"""#### **Model Building**

We scale the input features with `StandardScaler` and use `LogisticRegression` estimator with multi_class parameter set to `multinomial` and using `sag` solver.
"""

pipe = Pipeline([('scaler',StandardScaler()),
                 ('logreg',LogisticRegression(multi_class='multinomial',solver='sag'))])

pipe.fit(X_train, y_train)

pipe.score(X_train,y_train)

pipe.score(X_test,y_test)

image = pipe[-1].coef_[4].reshape(28, 28)
plt.imshow(image)

"""After training the model with the training feature matrix and labels, we learn the model parameters."""

pipe[-1].coef_.shape

pipe[-1].intercept_.shape

pipe[-1].classes_

"""#### **Model Evaluation**"""

print(classification_report(y_test ,pipe.predict(X_test)))

"""Most of the classses have `f1_score` greater than 90%, which is considered to be a good f1_score."""

ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
plt.show()

"""### **Using LogisticRegressionCV**"""

X_tr, X_te, y_tr, y_te = X[:10000], X[10000:10500], y[:10000], y[10000:10500]

scorer = make_scorer(f1_score, average='micro')

pipe = Pipeline([('scaler', StandardScaler()),
                 ('logreg', LogisticRegressionCV(cv=3,
                                                 multi_class='multinomial', solver='sag',
                                                 scoring=scorer, max_iter=100, random_state=1729))])

"""**Note :** takes quite a while to finish training (almost 10 mins)"""

pipe.fit(X_tr,y_tr)

"""#### **Learning the model parameters.**"""

pipe[-1].C_

pipe[-1].l1_ratio_

"""#### **Model Evaluation**"""

print(classification_report(y_te ,pipe.predict(X_te)))

ConfusionMatrixDisplay.from_estimator(pipe, X_te, y_te)
plt.show()

"""#### **Importing Libraries**"""

# Commented out IPython magic to ensure Python compatibility.
# Common imports
import numpy as np
from pprint import pprint

# to make this notebook's output stable across runs
np.random.seed(42)

# sklearn specific imports
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from sklearn.pipeline import make_pipeline ,Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model  import SGDClassifier, RidgeClassifier, LogisticRegression ,LogisticRegressionCV
from sklearn.model_selection import cross_validate, RandomizedSearchCV,GridSearchCV, cross_val_predict ,learning_curve

from sklearn.neighbors import KNeighborsClassifier

# log loss is also known as cross entropy loss
from sklearn.metrics import log_loss
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_curve ,make_scorer ,f1_score
from sklearn.metrics import roc_curve, roc_auc_score

#scipy
from scipy.stats import loguniform

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

# global settings
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)
mpl.rc('figure',figsize=(8,6))

import warnings
warnings.filterwarnings('ignore')

"""## **Handwritten Digit Classification**

### **Dataset**

Each datapoint is contained in $x_i$ ∊  $\mathbb{R}^{784}$ and the label $y_i$ ∊ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
"""

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

"""## **Binary Classification**

### **Changing labels to binary**

Let us do binary classification with KNN classifier and eventually extend it to Multiclass setup.
"""

# initialize new variable names with all -1
y_train_0 = -1*np.ones(len(y_train))
y_test_0 = -1*np.ones(len(y_test))

# find indices of digit 0 image
indx_0 = np.where(y_train == '0')

# remember original labels are of type str not int, so use those indices to modify y_train_0 & y_test_0

y_train_0[indx_0] = 1
indx_0 = np.where(y_test == '0')
y_test_0[indx_0] = 1

"""### **Data Visualization in Lower Dimensions**

* Let us apply PCA on the datapoints and reduce the dimensions to 2D and then to 3D.

* This will give us some rough idea about the points in $ \mathbb {R}^{784}$

* One interesting thing to look at is the change in the magnitude of the data points before and after applying PCA.
"""

from sklearn.decomposition import PCA

pipe_pca_2d = make_pipeline(MinMaxScaler(), PCA(n_components=2))
X_train_pca_2d = pipe_pca_2d.fit_transform(X_train)

"""**Visualization of the 2D data obtained through PCA**"""

plt.figure(figsize=(10, 10))
cmap = matplotlib.colors.ListedColormap(['r', 'b'])
sns.scatterplot(x=X_train_pca_2d[:, 0], y=X_train_pca_2d[:, 1],
                data=X_train_pca_2d, hue=y_train_0, palette=cmap)
plt.show()

"""**Projection in 3D using PCA**"""

pipe_pca_3d = make_pipeline(MinMaxScaler() ,PCA(n_components=3))
X_train_pca_3d = pipe_pca_3d.fit_transform(X_train)

import plotly.express as px

cmap = matplotlib.colors.ListedColormap(['r', 'b'])

fig = px.scatter_3d(x=X_train_pca_3d[:,0],
                    y=X_train_pca_3d[:,1],
                    z=X_train_pca_3d[:,2],
                    color=y_train_0,
                    color_discrete_map=cmap,
                    opacity=0.5)

fig.show()

"""## **KNN classifier**

#### **Algorithm :**

1. Set $k$ to desired value i.e. how many neighbors should be allowed to participate in prediction.

2. Calculate the distance between the new example and every example from the data. Thus, creating a distance vector.

3. Get indices of nearest $k$ neighbors.

4. Get the labels of the selected $k$ entries.

5. If it is a classification task, return the majority class by computing mode of $k$ labels.

To understand the working of sklearn built-in function, we first create a KNN classifier model with $k$=3 and consider a smaller number of samples of training and test set.

* The `KNeighborsClassifer` creates a classifier instance.

* There are many optional arguments such as `n_neighbors, metric, weights,` .... that can be set to suitable values while creating an instance.

Creating a new pipeline for classifier :

We use the variables `pipe_pca_2d` for preprocessing the samples alone and `pipe_clf_pca_2d` for classification.
"""

pipe_clf_pca_2d = make_pipeline(pipe_pca_2d, KNeighborsClassifier(n_neighbors=3))

"""Let us train the model with 10 samples from training set (i.e. we are just putting 10 datapoints in the metric space, not building any parameterized model)

Then test the model with 10 datapoints from  test set.
"""

index_neg = np.argsort(y_train_0)[:5]
index_pos = np.argsort(y_train_0)[-1:-6:-1]

# create a small dataset
x = np.vstack((X_train[index_pos, :], X_train[index_neg, :]))
y = np.hstack((y_train_0[index_pos], y_train_0[index_neg]))

y

pipe_clf_pca_2d.fit(x,y)

# for visulization
x_reduced = pipe_clf_pca_2d[0].transform(x)

plt.figure(figsize=(6, 4))
sns.scatterplot(x=x_reduced[:, 0], y=x_reduced[:, 1],
                hue=y, marker='o', palette=['r', 'b'])
plt.grid(True)
plt.show()

y_hat_0 = pipe_clf_pca_2d.predict(X_test[:10,:])

print('Test label : ',y_test_0[:10])
print('Predicted Label : ',y_hat_0[:10])

ConfusionMatrixDisplay.from_predictions(y_test_0[:10],y_hat_0)
plt.show()

"""**Observe that** :

* We can see that there are more FP's (as 9 out 10 actual labels are negative)

* Let us display both the training points and testing points with their predictions.

* We can visually validate the reason behind the performance.
"""

cmap = matplotlib.colors.ListedColormap(['r', 'b'])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_reduced[:, 0], y=x_reduced[:, 1],
                marker='o', hue=y, palette=cmap)

x_test_reduced = pipe_clf_pca_2d[0].transform(X_test[:10, :])
sns.scatterplot(x=x_test_reduced[:, 0], y=x_test_reduced[:, 1],
                s=100, marker='*', hue=y_test_0[:10], palette=cmap, legend=None)

dx, dy = -0.2, 0.2
for i in range(10):
    plt.annotate(str(y_hat_0[i]), xy=(x_test_reduced[i, 0]+dx, x_test_reduced[i, 1]+dy))

plt.grid(True)
plt.show()

"""* It would be much better if we know the distance of 3 neighbors for each testing points.

* Let us display the distance and connnectivity of neighbors to the test datapoints using th class `NearestNeighbors`.

* In fact, `KNeighborsClassifier` calls `NearestNeighbors` class internally to compute all these distances.
"""

from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=3)
neighbors.fit(pipe_pca_2d.transform(x))

"""which are the first three closest neighbors to the first three in the test set. And how close are they?"""

dist_neighbors, idx_neighbors = neighbors.kneighbors(pipe_pca_2d.transform(X_test[:10]), 3, return_distance=True)

import statistics

for i in range(3):
    print('Distance : {0} \nIndex : {1} \nLabels : {2} \nPrediction : {3}'.format(dist_neighbors[i],idx_neighbors[i],y[idx_neighbors[i].flatten()],
    statistics.mode(y[idx_neighbors[i].flatten()])))

    print('-'*20)

"""Let us train the model with 10000 samples from training set (i.e. we are just putting 10000 datapoints in the metric space, not building any parameterized model)."""

pipe_clf_pca_2d.fit(X_train[:10000],y_train_0[:10000])

y_hat_0 = pipe_clf_pca_2d.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test_0, y_hat_0)
plt.show()

print(classification_report(y_test_0,y_hat_0))

"""* Let's vary the n_neighbours from **k=1 to 19** and study the performance of the model.

* We use the first 10000 samples from training set.
"""

precision=[]

for k in range(1,20,2):
    pipe_clf_pca_2d.__n_neighbors=k
    pipe_clf_pca_2d.fit(X_train[:10000],y_train_0[:10000])

    y_hat_0 = pipe_clf_pca_2d.predict(X_test)
    precision.append(precision_score(y_test_0,y_hat_0))

plt.figure(figsize=(10, 8))
plt.plot(np.arange(1, 20, 2), precision)

plt.xlim((0, 20))
plt.ylim((0.64, 0.66))
plt.xlabel('k(odd values)')
plt.ylabel('Precision')

plt.xticks(ticks=np.arange(1, 20, 2), labels=np.arange(1, 20, 2))
plt.grid(True)
plt.show()

"""### **Going without PCA**

* Let us use KNN classifier with **all the features** in the training samples with the hope that it **increases the performance** of the model (of course at the cost of computation)

* Let's search for $k$ by using cross validation.

* **NOTE :** It takes about 4 minutes for entire computation.
"""

pipe_knn = make_pipeline(MinMaxScaler(),
                         KNeighborsClassifier(n_neighbors=1))

grid_k = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 7, 9, 11]}

cv = GridSearchCV(pipe_knn, param_grid=grid_k, scoring='precision', cv=5,n_jobs=1)

cv.fit(X_train, y_train_0)

pprint(cv)

pprint(cv.cv_results_)

"""**The best value obtained for k is 7.** (check in test_rank_score)"""

pipe_knn = make_pipeline(MinMaxScaler(),KNeighborsClassifier(n_neighbors=7))
pipe_knn.fit(X_train,y_train)

"""#### **Checking performance on test set**"""

y_hat_0 = pipe_knn.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_hat_0)
plt.show()

"""## **Multiclass Classification**

Extending KNN classifier to mulitclass classification is pretty simple straightforward.
"""

pprint(pipe_knn)

pipe_knn.fit(X_train, y_train)
y_hat = pipe_knn.predict(X_test)

pipe_knn.classes_

print(classification_report(y_test, y_hat))

"""### **Overview**
We know that k-NN can be used in addressing regressing problems.

In this notebook, we will demonstrate the use of **k-NN in regression** setup with **California housing dataset**, where we try to predict price of a house based on its features.

### **Imports**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler ,PolynomialFeatures
from sklearn.pipeline import Pipeline ,make_pipeline
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split ,GridSearchCV ,RandomizedSearchCV

"""### **STEP 1 : Training Data**

#### **Loading the dataset**

This dataset can be fetched from sklearn with `fetch_california_housing` API.
"""

from sklearn.datasets import fetch_california_housing
X,y = fetch_california_housing(return_X_y=True)

"""Lets check the shape of feature matrix and label vector."""

print('Shape of feature matrix : ' ,X.shape)
print('Shape of label vector : ',y.shape)

"""Perform quick sanity check to make sure  we have same number of rows in the feature matrix as well as the label vector."""

assert(X.shape[0] == y.shape[0])

"""#### **Split data into train & test sets**"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=8)

print('Shape of training feature matrix : ' ,X_train.shape)
print('Shape of training label vector : ',y_train.shape)
print()
print('Shape of test feature matrix : ' ,X_test.shape)
print('Shape of test label vector : ',y_test.shape)

assert(X_train.shape[0] == y_train.shape[0])
assert(X_test.shape[0] == y_test.shape[0])

"""#### **Preprocessing the dataset**

We have explored California housing set in detail earlier in the course

In order to refresh your memory, we have bargraphs corresponding to all the features and the output label plotted here.
"""

california_housing = fetch_california_housing(as_frame=True)

california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.5, wspace=0.4)

"""**Observe that :**

* The features are on a different scale and we need to bring them on the same scale for k-NN.

* k-NN uses Euclidean distance computation to identify the nearest neighbors to identify the nearest neighbors and it is crucial to have all the features on the same scale for that.

* If all the features are not on the same scale, the feature with wider variance would dominate the distance calculation.

### **STEP 2 : Model Building**

We instantiate a `pipeline` object with two stages;

* The first stage performs feature scaling with `MinMaxScaler`.

* And the second stage performs k-NN regressor with `n_neighbors=2`. In short, we are using 2-NN that is we use the price of the two nearest houses in feature space to decide the price of the new house.

* The model is trained with feature matrix and label vector from training set.

* After the model is trained, it is evaluated with the test set using the `mean squared error` metric.
"""

pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('knn', KNeighborsRegressor(n_neighbors=2))])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

error = mean_squared_error(y_test, y_pred, squared=False)
print(error)

"""### **STEP 3: Model Selection and Evaluation**

k-NN classifier has $k$, the number of neighbors, as a hyperparameter.

There are a couple of ways to tune the hyper-parameter
  1. Manual hyper-parameter tuning

  2. Using `GridSearchCV` or `RandomizedSearchCV`.

We will demonstrate both **Manual** as well as **Grid-Search** based hyperparameter tuning.

#### 3.A. **Manual HPT with cross-validation**

Here we train and evaluate the model pipeline with different values of k-1 to 31.
"""

rmse_val = []

for K in range(1, 31):
    pipe = Pipeline([('scaler', MinMaxScaler()),
                    ('knn', KNeighborsRegressor(n_neighbors=K))])

    # fit the model
    pipe.fit(X_train, y_train)

    # make prediction on test set
    pred = pipe.predict(X_test)

    # calculate rmse
    error = mean_squared_error(y_test, pred, squared=False)

    # store rmse values
    rmse_val.append(error)

"""At the end of this loop, we get a list of RMSEs-one for each value of k.

We plot the learning curve with **$k$ on x-axis** and **RMSE on y-axis**.

The  value of k that results in the **lowest RMSE is the best value of k** that we select.
"""

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(rmse_val)+1), rmse_val, color='red')

plt.xlabel('Different values of K', fontsize=12)
plt.ylabel('RMSE', fontsize=12, rotation=0)
plt.grid(True)

plt.title('Validations Loss vs K', fontsize=16)
plt.show()

rmse_val.index((min(rmse_val)))

"""#### 3.B. **HPT with GridSearchCV**

* We set up the parameter grid for values of k of our interest.

* Here we use the values between 1 and 31.

* The object of `GridSearchCV` is instantiated with a `KNeighborsRegressor` estimator along with the parameter grid and number of cross-validation folds equal to 10.

* The grid search is performed by calling the `fit` method with training feature matrix and labels as arguments.
"""

param_grid = {'knn__n_neighbors': list(range(1,31))}

pipe = Pipeline([('scaler',MinMaxScaler()),
                 ('knn',KNeighborsRegressor())
                ])

gs = GridSearchCV(pipe,param_grid = param_grid,
                  cv =10,
                  n_jobs=1,
                  return_train_score=True)

gs.fit(X_train,y_train)

"""Lets evaluate the best estimator on the test set."""

gs.best_estimator_

gs.best_params_

"""Making predictions on the test set"""

y_pred = gs.best_estimator_.predict(X_test)
mean_squared_error(y_test,y_pred,squared=False)

"""#### 3.C. **HPT with RandmizedSearchCV**"""

param_grid = {'knn__n_neighbors': list(range(1,31))}

pipe = Pipeline([('scaler',MinMaxScaler()),
                 ('knn',KNeighborsRegressor())
                ])

rs = RandomizedSearchCV(pipe, param_distributions=param_grid,
                      n_jobs=1,refit=True, cv=10,
                      return_train_score=True)

rs.fit(X_train,y_train)

"""Lets evaluate the best estimator on the test set."""

rs.best_estimator_

rs.best_params_

"""Making predictions on the test set"""

y_pred = rs.best_estimator_.predict(X_test)
mean_squared_error(y_test,y_pred,squared=False)

"""#### 3.D. **GridSearchCV + Polynomial Features**

In addition, we perform polynomial transformation on the features followed by scaling before using it in the nearest neighbor regressor.
"""

params = {'poly__degree': list(range(1, 4))}

pipe = Pipeline([('poly', PolynomialFeatures()),
                 ('scaler', MinMaxScaler()),
                 ('knn', KNeighborsRegressor())])

gs_poly = GridSearchCV(estimator=pipe,
                  param_grid=params,
                  cv=10,
                  n_jobs=1)

gs_poly.fit(X_train, y_train)

gs_poly.best_estimator_

"""We evaluate the model with the test set."""

y_pred = gs_poly.best_estimator_.predict(X_test)
error = mean_squared_error(y_test, y_pred, squared=False)

print('RMSE value of k is :', error)

"""### **Outline**

In this notebook, we study how to handle **large-scale datasets** in sklearn.

* In this course, so far we were able to load entire data in memory and were able to train and make inferences on all the data at once.

* The large scale data sets may not fit in memory and we need to devise strategies to handle it in the context of training and prediction use cases.

In this notebook, we will discuss the following topics :

* Overview of handling large-scale data.

* Incremental preprocessing and learning i.e. `fit()` vs `partial_fit()` : `partial_fit` is our friend in this cases.

* Combining preprocessing and incremental learning

## **Large-scale Machine Learning**

* Large-scale Machine Learning differs from traditional machine learning in the sense that it involves processing large amount of data in terms of its **size** or **number of samples, features or classes**

* There were many exciting developements in efficient large scale learning on many real world use cases in the last decade.

* Although scikit-learn is optimized for **smaller data**, it does offer a decent set of **feature preprocessing** and **learning algorithms** for large scale data such as classification, regression and clustering.

* Scikit-learn handles large data through `partial_fit()` method instead of using the usual `fit()` method.

The idea is to process data in **batches** and **update** the model parameters for each batch. This way of learning is referred to as **Incremental (or out-or-core) learning**.

### **Incremental Learning**

Incremental learning may be required in the following two scenarios :

* For **out-of-memory (large) datasets** ,where it's not possible to **load the entire data into the RAM** at once, one can load the data in chunks and fit the training model for each chunk of data.

* For machine learning tasks where a new batch of data comes with time,re-training the model with the previous and new batch of data is a computationally expensive process.

Instead of re-training the model with the entire set of data, one can employ an incremental learning approach, where the model parameters are updated with the new batch of data.

### **Incremental Learning in `sklearn`**

To perform incremental learning, Scikit-learn implements `partial_fit` method that helps in training an out-of-memory dataset.

In other words, it has the ability to learn incrementally from a batch of instances.

In this notebook, we will see an example of how to read, process, and train on such a large dataset that can't be loaded in memory entirely.

This method is expected to be called several times consecutively on different chunks of a dataset so as to implement out-of-core (online) learning.

This function has some performance overhead, so it's recommended to call it on a considerable large batch of data(that fits into the memory), to overcome the limitation of overhead.

### **partial_fit() attributes :**

`partial_fit(X,y,[classes], [sample_weight])`

where,

* `X` : array of shape(n_samples, n_features) where **n_samples** is the number of samples & **n_features** is the number of features.

* `y` : array of shape (n_samples,) of target values.

* `classes`: array of shape(n_classes,) containing a list of all the classes that can possibly appear in the y vector. Must be provided at the first call to partial_fit, can be omitted in subsequet calls.

* sample_`weight`: (optional) array of shape(n_samples,) containing weights applied to individual samples(1.for unweighted)

**Returns**: object(self)

For classification tasks, we have to pass the list of possible target class labels in `classes` parameter to cope-up with the unseen target classes in the 1st batch of the data.

The following estimators implement partial_fit method :

* **Classification** :
  * MultinomialNB

  * BernoulliNB

  * SGDClassifier
  
  * Perceptron

* **Regression** :
  * SGDRegressor


* **Clustering** :
  * `MiniBatchKmean`

`SGDRegressor` and `SGDClassifier` are commonly used for handling large data.

The problem with standard regression / classification implementations such as **batch gradient descent, support vector machines (SVMs), random forest** etc. is that because of the need to load all the data into memory at once, they can not be used in scenarios where we do not have sufficient memory.

SGD, however, can deal with large data sets effectively by breaking up the data into chunks and processing them sequentially.

The fact that we only need to load one chunk into memory at a time makes it useful for large-scale data as well as cases where we get streams of data at intervals.

### **fit() versus partial_fit()**

Below, we show the use of `partial_fit()` along with `SGDClassifier`.

For the purpose of illustration, we first use traditional `fit()` and then use `partial_fit()` on the same data.

### **Importing Libraries**
"""

# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report ,ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

"""### **Traditional Approach [using fit()]**

**Sample dataset**

We will use a synthetic classification dataset for demonstration.
Let us have 50000 samples with 10 features matrix.

Further, lets have 3 classes in the target label, each class having a single cluster.
"""

X, y = make_classification(n_samples=50000, n_features=10,
                           n_classes=3,
                           n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

"""We will make use of `SGDClassifier` to learn the classification model."""

clf1 = SGDClassifier(max_iter=1000, tol=0.01)

"""We will use traditional `fit()` method to train out model."""

clf1.fit(X_train, y_train)

"""Let's obtain the training and test scores on the trained model."""

train_score = clf1.score(X_train, y_train)
train_score

test_score = clf1.score(X_test, y_test)
test_score

"""We obtain the confusion matrix and classification report for evaluating the classifier."""

from sklearn.metrics import ConfusionMatrixDisplay

y_pred = clf1.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

print(classification_report(y_test, y_pred))

"""### **Incremental Approach [using partial_fit()]**

We will now assume that the data can't be kept completely in the main memory and hence, will load chunks of data and fit usng `partial_fit()`.
"""

X_train[:5]

y_train[:5]

"""In order to load data chunk, we will first store the given (training) data in a CSV file.

This is just for demonstration purpose. In a real-case scenario, the large dataset might already be in the form of say, a CSV file which we will be reading in multiple iterations.
"""

train_data = np.concatenate((X_train, y_train[:,np.newaxis]), axis=1)
train_data[:5]

a = np.asarray(train_data)
np.savetxt('train_data.csv',a, delimiter=',')

"""Now, our data for demonstration is ready in a csv file.

Let's create `SGDClassifier` object that we intend to train with `partial_fit()`.
"""

clf2 = SGDClassifier(max_iter=1000, tol=0.01)

"""#### **Processing data chunk by chunk**

* Pandas' read_csv() function has an attribute `chunksize` that can be used to read data chunk by chunk.

* The `chunksize` parameter specifies the number of rows per chunk. (The last chunk may contain fewer than chunksize rows, of course.)

* We can then use this data for `partial_fit()`.

* We can then repeat these two steps multiple times. That way entire data may not be required to be kept in memmory.
"""

import pandas as pd

chunksize = 1000
iter = 1

for train_df in pd.read_csv('train_data.csv', chunksize=chunksize, iterator=True):
    # print(train_data.shape)
    if iter ==1:
        # print(train_df)
        # In the first iteration, we are specifying all possible class labels.
        X_train_partial = train_df.iloc[:,0:10]
        y_train_partial = train_df.iloc[:,10]

        clf2.partial_fit(X_train_partial,y_train_partial,
                        classes=np.array([0,1,2]))

    else:
        X_train_partial = train_df.iloc[:,0:10]
        y_train_partial = train_df.iloc[:,10]

        clf2.partial_fit(X_train_partial,y_train_partial)

    print("After iter # : ", iter)
    print(clf2.coef_)
    print()
    print(clf2.intercept_)
    print('-'*30)
    iter+=1

"""**Note :**

* In the first call to `partial_fit()`, we passed the list of possible target class labels. For subsequent calls to `partial_fit()`, this is not required.

* Observe the changing values of the classifier attributes : `coef_` and `intercept_` which we are printing in each iteration.
"""

test_score = clf2.score(X_test ,y_test)
test_score

"""Let's evaluate the classifier by examining the `confusion_matrix`."""

y_pred = clf2.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.show()

print(classification_report(y_test, y_pred))

"""Apart from `SGDClassifier`, we can also train `Perceptron()`, `MultinomialNB()`, in a similar manner.

### **CountVectorizer vs HashingVectorizer**

Vectorizers are used to convert a collection of text documents to a vector representation, thus helping in preprocessing them before applying any model on these text documents.

`CountVectorizer` and `HashingVectorizer` both perform the task of vectorizing the text documents. However, there are some differences among them.

* One difference is that `HashingVectorizer` does not store the resulting vocabulary (i.e. the unique tokens). Hence, it can be used to learn from data that does not fit into the computer's main memory.

* Each mini-batch is vectorized using `HashingVectorizer` so as to guarantee that the input space of the estimator has always the same dimensionality.

* With `HashingVectorizer`, each token directly maps to a pre-defined column position in a matrix.

* For example, if there are 100 columns in the resultant (vectorized) matrix, each token (word) maps to 1 of the 100 columns. The mapping between the word and the position in matrix is done using hashing.

* In other words in `HashingVectorizer`, each token transforms to a column position instead of adding to the vocabulary.

Not storing the vocabulary is useful while handling large datasets. This is because holding a huge token vocabulary compromising of millions of words may be a challenging when the memory is limited.

Since `HashingVectorizer` does not store vocabulary , its object not only takes lesser space, it also alleviates any dependence with function calls performed on the previous chunk of data in case of incremental learning.

Let us take some sample text documents and vectorize them, first using **CountVectorizer** and then **HashingVectorizer**.
"""

text_documents = ['The well-known saying an apple a day keeps the doctor away has a very straightforward, literal meaning, that the eating of fruit maintains good health.',
'The proverb fist appeared in print in 1866 and over 150 years later is advice that we still pass down through generation.',
'British apples are one of the nations best loved fruit and according to Great British Apples, we consume around 122,000 tonnes of them each year.',
'But what are the health benefits, and do they really keep the doctor away?']

"""#### **1. CountVectorizer**

We will first import the library and then create an object of CountVectorizer class.
"""

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()

"""We will now use this object to vectorize the input text documents using the function `fit_transform()`."""

X_c = count_vectorizer.fit_transform(text_documents)

X_c.shape

"""Here, **66** is the **size of the vocabulary**.

We can also see the vocabulary using `vocabulary_` attribute.
"""

count_vectorizer.vocabulary_

"""And **4** is the **number of text documents**.

Following is the representation of four text documents :
"""

print(X_c)

"""#### **2. HashingVectorizer**

Let us now see how `HashingVectorizer` is different from `CountVectorizer`.

We will create an object of `HashingVectorizer`. While creating the object, we need to specify the number of features we wish to have in the feature matrix.
"""

from sklearn.feature_extraction.text import HashingVectorizer
hashing_vectorizer = HashingVectorizer(n_features=50)

"""An important parameter of `HashingVectorizer` class is `n_features`. It declares the number of features (columns) in the output feature matrix.

**NOTE :** Small numbers of features are likely to cause hash collisions, but large numbers will cause larger coefficient dimensions in linear learners.

Let's perform hashing vectorization with `fit_transform`.
"""

X_h = hashing_vectorizer.fit_transform(text_documents)

"""Let us examine the shape of the transformed feature matrix. The number of columns in this matrix is equal to the `n_features` attribute we specified."""

X_h.shape

"""Following is the representation of the four text documents :"""

print(X_h[0])

"""**IMP :**

Overall, `HashingVectorizer` is a good choice if we are falling short of memory and resources, or we need to perform incremental learning.

However, `CountVectorizer` is a good choice if we need to access the actual tokens.

### **Demonstration**

#### **1. Downloading the dataset**

We download a dataset from UCI ML datasets's library.

Instead of downloading, unzipping and then reading, we are directly reading the zipped csv file.

For that purpose, we are making use of `urllib.request`, `BytesIO` and `TextIOWrapper` classes.

This is a sentiment analysis dataset. There are only two columns in the dataset. One for the textual review and the other for the sentiment.
"""

from io import StringIO, BytesIO, TextIOWrapper
from zipfile import ZipFile
import urllib.request

resp = urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip')

zipfile = ZipFile(BytesIO(resp.read()))

data = TextIOWrapper(zipfile.open('sentiment labelled sentences/amazon_cells_labelled.txt'),encoding='utf-8')

df = pd.read_csv(data, sep='\t')
df.columns = ['review','sentiment']

"""#### **2. Exploring the dataset**"""

df.head()

df.info()

df.describe()

"""As we can see,

* There are 999 samples in the dataset.

* The possible classes for sentiment are 1 and 0.

#### **3. Splitting data into train and test**
"""

X = df.loc[:,'review']
# X2 = df.iloc[:,0]

y = df.loc[:,'sentiment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""#### **4. Preprocessing**"""

from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer()

"""#### **5. Creating an instance of the SGDClassifier**"""

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(penalty='l2', loss='hinge')

"""#### **6. Iteration 1 of partial_fit()**

* We will assume we do not have sufficient memory to handle all the 799 samples in one go for training purpose.

* So, we will take the first 400 samples from the training data and `partial_fit()` our classifier.

* Another use case of `partial_fit` here could also be a scenario where we only have 400 samples available at a time. So, we fit our classifier with them.

* However, we `partial_fit` it, to have the possibility of traning it with more data later whenever that becomes available.
"""

X_train_part1_hashed = vectorizer.fit_transform(X_train[0:400])
y_train_part1 = y_train[0:400]

# we need to mention all classes in the first iteration of partial_fit()
all_classes = np.unique(df.loc[:, 'sentiment'])

classifier.partial_fit(X_train_part1_hashed,
                       y_train_part1, classes=all_classes)

"""Let us now use this classifier on our test data that we had kept aside earlier."""

# first we will have to preprocess the X_test with the same vectorizer that was fit on the train data.
X_test_hashed = vectorizer.transform(X_test)

test_score = classifier.score(X_test_hashed, y_test)
print('Test score : ', test_score)

"""**Note :**
We can also store this classifier using pickle object and can access it later.

#### **7. Iteration 2 of partial_fit()**

We will now assume that more data became available. So, we will fit the same  classifier with more data and observe if our test score improves.
"""

X_train_part2_hashed = vectorizer.fit_transform(X_train[400:])
y_train_part2 = y_train[400:]

classifier.partial_fit(X_train_part2_hashed, y_train_part2)

test_score = classifier.score(X_test_hashed, y_test)
print('Test score : ', test_score)

"""We see that our test score has improved after we fed more data to the classifier in the second iteration of `partial_fit()`.

## **SVM classifer on MNIST**

In this notebook, we will implement **multiclass MNIST digit recognition classifier** with **SVM's**.

#### **Importing Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

# Import the libraries for performing classification
from keras.datasets import mnist
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score ,train_test_split ,GridSearchCV ,StratifiedShuffleSplit

"""#### **Loading MNIST dataset**"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()

"""Flatten each input image into a vector of length 784"""

X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

"""Normalizing"""

X_train = X_train/255
X_test = X_test/255

X_train.shape ,y_train.shape ,X_test.shape ,y_test.shape

"""Let us consider the first 10000 images in training dataset and first 2000 images in testing dataset."""

X_train = X_train[0:10000, :]
y_train = y_train[0:10000]

X_test = X_test[0:2000, :]
y_test = y_test[0:2000]

"""### **Linear SVM for MNIST multiclass classification**

#### **Using Pipeline**
"""

pipe_1 = Pipeline([('scaler', MinMaxScaler()),
                   ('classifier', SVC(kernel='linear', C=1))])

pipe_1.fit(X_train, y_train.ravel())

"""Evaluate the model using crossvalidation"""

accuracy = cross_val_score(pipe_1, X_train, y_train.ravel(), cv=3)
print('Training Accuracy : {:.4f}'.format(accuracy.mean()*100))

"""Visualizing the confusion matrix"""

y_pred = pipe_1.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion matrix')
plt.show()

"""Printing classification report"""

print(classification_report(y_test,y_pred))

"""### **Nonlinear SVM for MNIST multiclass classification**

#### **Using Pipeline**
"""

pipe_2 = Pipeline([('scalerr', MinMaxScaler()),
                   ('classifier', SVC(kernel='rbf', gamma=0.1, C=1))])

pipe_2.fit(X_train, y_train.ravel())

"""Evaluate the model using crossvalidation"""

accuracy = cross_val_score(pipe_2, X_train, y_train.ravel(), cv=2)
print('Training Accuracy : {:.4f}'.format(accuracy.mean()*100))

"""Visualizing the confusion matrix"""

y_pred = pipe_2.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.show()

"""Printing classification report"""

print(classification_report(y_test,y_pred))

"""#### **Using GridSearchCV**

We can use a grid search cross-validation to explore combinations of parameters.

Here we will adjust `C` (which controls the margin hardness) and `gamma` (which controls the size of the radial basis function kernel), and determines the best models.
"""

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)

cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

"""takes a very long amount of time to finish training (for me it took about 7 hrs)"""

grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train.ravel())

grid.best_params_

grid.best_score_

"""## **Decision Trees**

Decision Trees are capable of finding  **complex non-linear relationships** in the data.

They can perform both **classification** and **regression** tasks.

## **Decision Trees for Regression**

In the first half of this notebook, we will demonstrate decision trees for regression task with Califiornia housing dataset and `DecisionTreeRegressor` class in `sklearn`.

### **Importing Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns ; sns.set()

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import export_text

import warnings
warnings.filterwarnings('ignore')

np.random.seed(36)

"""Let's use `ShuffleSplit` as cv with 10 splits and 20 % examples set aside as test examples."""

cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=42)

"""### **Loading the dataset**"""

features, labels = fetch_california_housing(as_frame=True, return_X_y=True)

"""### **Data Splitting**"""

com_train_features, test_features, com_train_labels, test_labels = train_test_split(features, labels, random_state=42)


train_features, dev_features, train_labels, dev_labels = train_test_split(com_train_features, com_train_labels, random_state=42)

"""### **Model Setup**"""

dt_reg_pipeline = Pipeline([('scaler', StandardScaler()),
                            ('dt_reg', DecisionTreeRegressor(max_depth=3, random_state=42))])

dt_reg_cv_results = cross_validate(dt_reg_pipeline,
                                   com_train_features,
                                   com_train_labels,
                                   cv=cv,
                                   scoring='neg_mean_absolute_error',
                                   return_train_score=True,
                                   return_estimator=True)

dt_reg_train_error = -1 * dt_reg_cv_results['train_score']
dt_reg_test_error = -1 * dt_reg_cv_results['test_score']

print(f'Mean absolute error of linear regression model on the train set : \n'
      f'{dt_reg_train_error.mean():.3f}+/- {dt_reg_train_error.std():.3f}')

print()

print(f'Mean absolute error of linear regression model on the test set : \n'
      f'{dt_reg_test_error.mean():.3f}+/- {dt_reg_test_error.std():.3f}')

"""### **Visualizing the tree**
One of the advantages of using a decision tree classifier is that the output is intuitive to understand and can be easily visualized.

This can be done in two ways:
* As a tree digram

* As a text based diagram

#### 1. **As a tree diagram**

We need to call `fit` function on `pipeline` object before printing the tree.
"""

dt_reg_pipeline.fit(train_features, train_labels)

plt.figure(figsize=(25,5))

a = tree.plot_tree(dt_reg_pipeline[-1],
                   feature_names=features.columns,
                   rounded=True,
                   filled=True,
                   fontsize=12)

plt.show()

"""#### 2. **As a text-based diagram**"""

# export the decision rules
tree_rules = export_text(dt_reg_pipeline[-1])

print(tree_rules)

"""### **Using the tree for prediction**"""

test_labels_pred = dt_reg_pipeline.predict(test_features)

"""### **Evaluating the tree**"""

mae = mean_absolute_error(test_labels, test_labels_pred)
mse = mean_squared_error(test_labels, test_labels_pred)
r2 = r2_score(test_labels, test_labels_pred)

print('MAE : ',mae)
print('MSE : ',mse)
print('R2 score : ',r2)

"""### **HPT using GridSearchCV**

Let us now try to improve the model by tuning the hyperparameters.
"""

param_grid = {'dt_reg__max_depth': range(1, 20),
              'dt_reg__min_samples_split': range(2, 8)}

dt_grid_search = GridSearchCV(dt_reg_pipeline,
                              param_grid=param_grid,
                              n_jobs=2,
                              cv=cv,
                              scoring='neg_mean_absolute_error',
                              return_train_score=True)

dt_grid_search.fit(com_train_features, com_train_labels)

dt_grid_search.best_params_

print('Mean cross validated score of the best estimator : ', -
      1*dt_grid_search.best_score_)

mean_train_error = -1 * \
    dt_grid_search.cv_results_['mean_train_score'][dt_grid_search.best_index_]

mean_test_error = -1 * \
    dt_grid_search.cv_results_['mean_test_score'][dt_grid_search.best_index_]

std_train_error = -1 * \
    dt_grid_search.cv_results_['std_train_score'][dt_grid_search.best_index_]

std_test_error = -1 * \
    dt_grid_search.cv_results_['std_test_score'][dt_grid_search.best_index_]


print(f'Best Mean absolute error of decision tree regression model on the train set: \n'f'{mean_train_error:.3f} +/- {std_train_error:.3f}')

print()

print(f'Best Mean absolute error of decision tree regression model on the test set: \n'f'{mean_test_error:.3f} +/- {std_test_error:.3f}')

"""Let's retrain the model with the best hyperparameter value."""

dt_reg_pipeline.set_params(dt_reg__max_depth=11, dt_reg__min_samples_split=5).fit(com_train_features, com_train_labels)

"""Evaluating after HPT."""

test_labels_pred = dt_reg_pipeline.predict(test_features)

mae = mean_absolute_error(test_labels, test_labels_pred)
mse = mean_squared_error(test_labels, test_labels_pred)
r2 = r2_score(test_labels, test_labels_pred)

print('MAE : ',mae)
print('MSE : ',mse)
print('R2 score : ',r2)

"""## **Decision Trees using Pipelines**

For this section of the notebook, we will use **Abalone data**.

### **Loading the dataset**

* Abalone is a type of consumable snail whose price varies as per its age.

* The aim is to predict the age of abalone from physical measurements.

* The age of abalone is traditionally determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope-a boring and time-consuming task.

* Other measurements, which are easier to obtain, are used to predict the age.
"""

column_names = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

abalone_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', header=None,names=column_names)

abalone_data.info()

abalone_data.describe().T

"""Let's now see the type and name of the features :

 * **Sex**: The is the gender of the abalone and has categorical value (M, F or I)

 * **Length**: The longest measurement of the abalone shell in mm. Continuous numeric value.

 * **Diameter**: The measurement of the abalone shell perpendicular to lenght in mm. Continuous numeric value.

 * **Height**: Height of the shell in mm. Continuous numeric value. Whole Weight: Weight of the abalone in grams. Continous numeric value.

 * **Shucked Weight**: Weight of just the meat in abalone in grams. Continuous numeric value.

 * **Viscera Weight**: Weight of the abalone after bleeding in grams. Continuous numeric value.

 * **Shell Weight**: Weight of the abalone after being dried in grams. Continuous numeric value.

 * **Rings**: This is the target, that is the feature that we will train the model to predict. As mentioned earlier, we are interested in the age of the abalone and it has been established that number of rings + 1.5 gives the age. Discrete numeric value.

### **Visualization of Abalone Dataset**
"""

abalone_data.hist(bins=50, figsize=(15,15))
plt.show()

sns.pairplot(abalone_data, diag_kind='hist')
plt.show()

sns.heatmap(abalone_data.iloc[:, :-1].corr(), annot=True, square=True)
plt.show()

sns.boxplot(data=abalone_data.iloc[:, :-1], orient='h', palette='Set2')
plt.show()

"""We find different features to be having different ranges through this box-plot, which indicates that scaling the features may be useful.

### **Preprocessing**

From the information above, all features are continuous variables except for the Sex feature.

#### **Handling Missing values**
"""

abalone_data.describe().T

"""The count row shows that there are no missing values.

However, in the `Height feature`, the minimum value is zero. This possibility calls for a missing value in the data and we will process tha missing value.

We first check how many missing values are in the `Height feature` and which class is it in.
"""

(abalone_data['Height']==0).sum()

abalone_data[abalone_data['Height']==0]

"""The number of missing values is 2 and is in the infant sex.

Then we change the value 0 to null. We will fill in the missing value with the average Height feature for the infant gender.
"""

mean = pd.pivot_table(abalone_data, index=['Sex'], aggfunc={'Height': np.mean})
mean

"""So we will fill in the missing value with 0.107996. (will perform the next step a little later)

#### **Target Column**

Next, take a look at the target in this case in the Rings column
"""

abalone_data['Rings'].unique()

abalone_data['Rings'].value_counts().sort_index()

"""We can see that the target is 1 to 29 (but there is no 28), so the classification we are goinng to do is a multi-class classification.

### **Storing data in the form of X & y**
"""

X = abalone_data.iloc[:,:-1]
y = abalone_data.iloc[:,-1]

X[:5]

y[:5]

"""### **Splitting data into train and test sets.**"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

"""### **Pipelining**

We will use pipelines to perform preprocessing of the data, which will include: handling missing (or 0) values, scaling the features and handling the categorical feature (viz., sex in this case)
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

"""Identifying numeric and categorical features (to be able to preprocess them differently.)"""

numeric_features = ['Length','Diameter','Height','Whole weight','Shucked weight', 'Viscera weight', 'Shell weight']

categorical_features = ['Sex']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=0, strategy='constant',fill_value=0.107996)),
    ('scaler',StandardScaler())
    ])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features)]
)

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.

clf = Pipeline(
    steps=[('preprocessor', preprocessor),
           ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42))]
)

clf.fit(X_train, y_train)
print('Model score : {:.3f}'.format(clf.score(X_test,y_test)))

"""### **Evaluation**"""

y_pred = clf.predict(X_test)

"""Let us compare the actual and predicted values of y."""

# comparision = np.concatenate(
#     (y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1)

# for each in comparision:
#     print(each)

"""**Confusion Matrix by ConfusionMatrixDisplay**"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Confusion matrix')
plt.show()

"""**Confusion Matrix by heatmap**"""

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test ,y_pred) ,annot=True , yticklabels=False ,cbar=False ,cmap='Greens')

"""**Classification Report**"""

from sklearn.metrics import classification_report
print(classification_report(y_test ,y_pred))

"""**Cross-Val Score**"""

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)

print('Accuracy of each fold : \n', list(acc*100))
print()
print('Accuracy : ',acc.mean()*100)

"""### **Visualizing the decision tree**"""

plt.figure(figsize=(30,10))
a = tree.plot_tree(clf['classifier'],
                   feature_names=column_names,
                   rounded=True,
                   filled=True,
                   fontsize=12)

plt.show()

"""### **Finding the best parameters using GridSearchCV**"""

X_train_new = preprocessor.fit_transform(X_train)

tuned_parameters = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                     'min_samples_split':[2, 4, 6, 8, 10]}]

scores = ['recall']

for score in scores:

    clf_cv = GridSearchCV(DecisionTreeClassifier(),
                        tuned_parameters,
                        scoring=f'{score}_macro')

    clf_cv.fit(X_train_new, y_train)

    print('Best parameters :' ,clf_cv.best_params_)
    print()
    print('Grid Score is as follows : \n')
    means = clf_cv.cv_results_['mean_test_score']
    stds = clf_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf_cv.cv_results_['params']):
        print(f'{mean:0.3f} (+/-) {std*2:0.03f} for {params}')

"""Let us now create a new pipeline using the best features identified above."""

clf2 = Pipeline(steps=[('preprocessor',preprocessor),
                       ('classifier',DecisionTreeClassifier(max_depth=5,min_samples_split = 2, random_state=42))] )

clf2.fit(X_train, y_train)
print('Model score  : {:.3f}'.format(clf2.score(X_test,y_test)))

"""## **Decision Trees for Classification**

In this half of the notebook, we will demonstrate decision trees for classification task with **Iris dataset** and `DecisionTreeClassifier` class in `sklearn`.

Let's load Iris dataset with `load_iris`API
"""

from sklearn.datasets import load_iris
features, labels = load_iris(return_X_y=True, as_frame=True)

"""Let's split the data into train and test."""

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42)

"""Define the decision tree classifier as part of `pipeline`."""

from sklearn.preprocessing import MinMaxScaler

dt_pipeline = Pipeline([('scaler', MinMaxScaler()),
                        ('dt_classifier', DecisionTreeClassifier(max_depth=3,random_state=42))])

"""Training the classifier."""

dt_pipeline.fit(train_features, train_labels)

"""Now that the classifier is trained, let's evaluate it on the test set with :
 * Confusion matrix

 * Classification report
"""

ConfusionMatrixDisplay.from_estimator(dt_pipeline, test_features, test_labels)
plt.show()

print(classification_report(test_labels, dt_pipeline.predict(test_features)))

"""As a next step let's visualize the trained decision tree model."""

plt.figure(figsize=(20, 8))

a = tree.plot_tree(dt_pipeline[-1],
                   #use the feature names stored
                   feature_names=features.columns,
                   #use the class names stored
                   class_names=load_iris().target_names,
                   rounded=True,
                   filled=True,
                   fontsize=12)

plt.show()

"""Let's convert this tree representation into if-else rule set."""

#export the decision rules
tree_rules = export_text(dt_pipeline[-1], feature_names=list(features.columns))

print(tree_rules)

"""Let's get the feature importance from the trained decision tree model."""

importance = pd.DataFrame({'features': features.columns,
                           'importance': np.round(dt_pipeline[-1].feature_importances_, 4)})

importance.sort_values('importance', ascending=False, inplace=True)
print(importance)

"""Now, perform HPT using GridSearchCV :

There are two configurable parameters in the tree classifier :

* `max_depth`

* `min_samples_split`
"""

param_grid = [{'dt_classifier__max_depth': [1, 2, 3, 4, 5],
            'dt_classifier__min_samples_split': [2, 4, 6, 8, 10]}]

gs_clf = GridSearchCV(dt_pipeline, param_grid, scoring='f1_macro')
gs_clf.fit(train_features, train_labels)

print('Best parameters : ', gs_clf.best_params_)
print()

print('Grid scores are as follows : \n')
means = gs_clf.cv_results_['mean_test_score']
stds = gs_clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, gs_clf.cv_results_['params']):
  print(f'{mean:0.3f} (+/-) {std*2:0.03f} for {params}\n')

"""Confusion matrix for the best estimator obtained through the `GridSearchCV`."""

ConfusionMatrixDisplay.from_estimator(
    gs_clf.best_estimator_, test_features, test_labels)

plt.show()

"""### **Objective**

In this notebook, we will implement **multiclass MNIST digit recognition classifier** with **decision trees** and **ensemble techniques**.

### **Importing basic libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataset loading through mnist
from keras.datasets import mnist

#training three classifiers: decision tree, bagging and random forest.
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# model selection utilitities for training and test split and cross validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline

"""### **Loading MNIST dataset**

We begin by loading the MNIST dataset with `load_data` in `mnist` class.

We obtain :

 * Training feature matrix and labels

 * Test feature matrix and labels
"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()
data = mnist.load_data()

plt.imshow(X_train[0])
plt.show()

"""* There are 60000 examples in training set and 10000 examples in the test set.

* Each example is a grey scale image of size 28 X 28.

* There are 10 different labels - one for each digit - 0 to 9.
"""

print('Shape of training data : ', X_train.shape)
print('Shape of training labels : ', y_train.shape)
print()
print('Shape of testing data : ', X_test.shape)
print('Shape of testing labels : ', y_test.shape)

"""Before using the dataset for training and evaluation, we need to flatten it into a vector.

After flattening, we have training and test examples represented with a vector of 784 features.

Each feature records pixel intensity in each of 28 X 28 pixel.

We normalize the pixel intensity by dividing it with the maximum value i.e. 255. In that sense we have each feature value in the range 0 to 1.
"""

# Flatten each input image into a vector of length 784.
X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

# Normalizing
X_train = X_train/255
X_test = X_test/255

print('Shape of training data after flattening : ', X_train.shape)
print('Shape of testing data after flattening : ', X_test.shape)

"""We use ShuffleSplit cross validation with 10 splits and 20% data set aside for model evaluation as a test data ."""

cv = ShuffleSplit(n_splits=10,test_size=0.2, random_state=42)

"""### **Model Building**

We define two functions :

1. **train_classifiers** function :

* It contains a common code for training classifiers for MNIST multiclass classification problem.

* It takes `estimator`, feature matrix, labels, cross validation strategy and name of the classifier as input.

* It first fits the estimator with feature matrix and labels.

* It obtains cross validated `f1_macro` score for training set with 10-fold `ShuffleSplit` cross validation and prints it.
"""

def train_classifiers(estimator, X_train, y_train, cv, name):
    estimator.fit(X_train, y_train)
    cv_train_score = cross_val_score(
        estimator, X_train, y_train, cv=cv, scoring='f1_macro')

    print(
        f'On an average, {name} model has f1 score of 'f'{cv_train_score.mean():.3f} (+/-) {cv_train_score.std():.3f} on the training set')

"""2. `eval` function :

* It takes estimator, test feature matrix and labels as input and produce classification report and confusion matrix.

* It first predicts labels for the test set.

* Then it uses these predicted reports for calculating various evaluation metrics like precision, recall, f1 score and accuracy for each of the 10 classes.

* It also obtains a confusion matrix by comparing these predictions iwth the actual labels and displays it with `ConfusionMatrixDisplay` utility.
"""

def eval(estimator, X_test ,y_test):
    y_pred = estimator.predict(X_test)

    print('Classification Report :')
    print(classification_report(y_test ,y_pred))

    print('Confusion Matrix : ')
    sns.heatmap(confusion_matrix(y_test ,y_pred) ,cmap='Blues',annot=True ,cbar=True ,fmt='.5g')
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

"""Let's train three classifiers with default parameters.

 * **Decision tree**

 * **Bagging classifier** - which uses decision tree as a default classifier and trains multiple decision tree classifiers on different bags obtained through boostrap sampling of training set.

 * **Random forest classifier** - which is also a bagging technique, which trains different decision tree classifiers by randomly selecting attributes for splitting on bags of boostrap sample of training set.

### **Decision trees for MNIST multiclass classification**

We instantiate a decision tree classifier with default parameters and train it.

The `train_classifier` function prints the mean of cross validated accuracy and standard deviation of the trained classifier on the training set.
"""

dt_pipeline = Pipeline([('classifier', DecisionTreeClassifier())])

train_classifiers(dt_pipeline, X_train,
                  y_train.ravel(), cv, 'decision tree')

"""Let's evaluate the trained classifier on the test set."""

eval(dt_pipeline, X_test, y_test)

"""### **MNIST classification with Bagging**

First instantiate a bagging classifier object with default parameters and train it.

Observe the mean `f1_score` and its standard deviation obtained by the classifier based 10-fold cross validation of the training set.
"""

# bagging_pipeline = Pipeline([('scaler',MinMaxScaler()),('classifier',     BaggingClassifier())])

bag_pipeline = Pipeline([('classifier', BaggingClassifier())])

train_classifiers(bag_pipeline, X_train, y_train.ravel(), cv, 'bagging')

"""Let's evaluate the trained classifier on the test set."""

eval(bag_pipeline, X_test, y_test)

"""### **Random forest for MNIST multiclass classification**

Let's instantiate a random forest classifier object with default parameters and train it.

Observe the mean `f1_score` and its standard deviation obtained by the classifier based 10-fold cross validation of the training set.
"""

rf_pipeline = Pipeline([('classifier', RandomForestClassifier())])

train_classifiers(rf_pipeline,X_train, y_train.ravel(), cv, 'random forest')

"""Now let's evaluate a random forest classifier on the test set and obtain classification report containing precision, recall, f1-score and accuracy for each class.

It also calculates confusion matrix and displays it with seaborn heatmap utility.
"""

eval(rf_pipeline, X_test, y_test)

"""## **Summary**

* We trained three multi-class classifiers for handwritten digit recognition.

* The **decision tree classifier** is a baseline classifier, which obtained accuracy of **87%** on the test set.

* Using **bagging** and training the same decision tree classifier gave us an increase of 7 percentage point in the accuracy, which translates to **94%** accuracy on the test set.

* Finally, the **random forest classifier** pushed that further to **97%**.

* We can see that how ensemble techniques give better results on the classification task compared to a single classifier.

### **Objective**

In this notebook, we will apply ensemble techniques regression problem in california housing dataset.

We have already applied different regressors on california housing dataset. In this notebook, we will make use of :

  * Decision tree regressor

  * Bagging regressor

  * Random Forest regressor

We will observe the performance improvement when we use random forest over decision trees and bagging, which also uses decision tree regressor.

### **Importing basic libraries**
"""

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit

np.random.seed(306)

"""Let's use `ShuffleSplit` as cv with 10 splits and 20% examples set aside as text examples."""

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

"""Let's download the data and split it into training and test sets."""

features, labels = fetch_california_housing(as_frame=True, return_X_y=True)
labels *= 100

com_train_features, test_features, com_train_labels, test_labels = train_test_split(features, labels, random_state=42)

train_features, dev_features, train_labels, dev_labels = train_test_split(
    com_train_features, com_train_labels, random_state=42)

"""### **Training different Regressors**

Let's train different regressors :
"""

def train_regressor(estimator, X_train, y_train, cv, name):
    cv_results = cross_validate(estimator,
                                X_train,
                                y_train,
                                cv=cv,
                                scoring='neg_mean_absolute_error',
                                return_train_score=True,
                                return_estimator=True)

    cv_train_error = -1 * cv_results['train_score']
    cv_test_error = -1 * cv_results['test_score']

    print(f'On an average, {name} makes an error of ',
            f'{cv_train_error.mean():.3f} (+/-) {cv_train_error.std():.3f} on the training set.')

    print(f'On an average, {name} makes an error of ',
            f'{cv_test_error.mean():.3f} (+/-) {cv_test_error.std():.3f} on the testing set.')

"""#### **Decision Tree Regressor**"""

train_regressor(DecisionTreeRegressor() ,com_train_features, com_train_labels ,cv, 'decision tree')

"""#### **Bagging Regressor**"""

train_regressor(BaggingRegressor(), com_train_features, com_train_labels, cv, 'bagging regressor')

"""#### **Random Forest Regressor**"""

train_regressor(RandomForestRegressor(), com_train_features, com_train_labels, cv, 'random forest regressor')

"""### **Parameter search for random-forest-regressor**"""

param_grid = {
    'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500],
    'max_leaf_nodes': [2, 5, 10, 20, 50, 100]
}

search_cv = RandomizedSearchCV(
    RandomForestRegressor(n_jobs=2), param_grid,
    scoring='neg_mean_absolute_error', n_iter=10, random_state=0, n_jobs=-1,)

search_cv.fit(com_train_features, com_train_labels)

columns = [f'param_{name}' for name in param_grid.keys()]
columns += ['mean_test_error', 'std_test_error']

cv_results = pd.DataFrame(search_cv.cv_results_)

cv_results['mean_test_error'] = -cv_results['mean_test_score']
cv_results['std_test_error'] = cv_results['std_test_score']
cv_results[columns].sort_values(by='mean_test_error')

error = - search_cv.score(test_features, test_labels)
print(f'On average, our random forest regressor makes an error of {error:.2f}.')

"""### **Objective**

In this notebook, we will implement **multiclass MNIST digit recognition classifier** with **boosting** :

 * AdaBoost

 * GradientBoosting

 * XGBoost

### **Importing basic libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from keras.datasets import mnist

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report

from sklearn.pipeline import Pipeline

"""### **Loading MNIST dataset**

Begin by loading MNIST dataset with `load_data` function in `mnist` class.

We obtain:

* Training feature matrix and labels

* Test feature matrix and labels

"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()
data = mnist.load_data()

"""* There are 60000 examples in training set and 10000 examples in the test set.

* Each example is a grey scale image of size 28 X 28.

* There are 10 different labels - one for each digit - 0 to 9.
"""

# Flatten each input image into a vector of length 784
X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

# Normalizing
X_train = X_train / 255
X_test = X_test / 255

"""We use ShuffleSplit cross validation with 10 splits and 20% data set aside for model evaluation as a test data ."""

cv = ShuffleSplit(n_splits=10,test_size=0.2, random_state=42)

"""### **Model Building**

We define two functions :
"""

def train_classifiers(estimator, X_train, y_train, cv, name):
    estimator.fit(X_train, y_train)
    cv_train_score = cross_val_score(
        estimator, X_train, y_train, cv=cv, scoring='f1_macro')

    print(
        f'On an average, {name} model has f1 score of 'f'{cv_train_score.mean():.3f} (+/-) {cv_train_score.std():.3f} on the training set')

def eval(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)

    print('Classification Report :')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix : ')
    sns.heatmap(confusion_matrix(y_test, y_pred),
                cmap='Greens', annot=True, cbar=True, fmt='.5g')
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

"""### **AdaBoost for MNIST multiclass classification**

We instantiate a decision tree classifier with default parameters and train it.

The `train_classifier` function prints the  means of cross validated accuracy and standard deviation of the trained classifier on the training set.
"""

adb_pipeline = Pipeline([('classifier', AdaBoostClassifier())])

train_classifiers(adb_pipeline, X_train, y_train.ravel(),
                  cv, 'AdaBoostClassifier')

eval(adb_pipeline, X_test, y_test)

"""### **GradientBoostingClassifier for MNIST classification**

Let's instantiate a gradient boosting classifier object with default parameters and train it.

Observe the mean `f1_score` and its standard deviation obtained by the classifier based 10-fold cross validation of the training set.
"""

grb_pipeline = Pipeline(
    [('classifier', GradientBoostingClassifier(n_estimators=10))])

train_classifiers(grb_pipeline, X_train, y_train.ravel(),
                  cv, 'GradientBoostingClassifier')

"""Let's evaluate the trained classifier on the test set."""

eval(grb_pipeline, X_test, y_test)

"""### **XGBoost Classifier for MNIST classification**"""

from xgboost import XGBClassifier

xgbc_pipeline = Pipeline([("classifier",XGBClassifier())])

train_classifiers(xgbc_pipeline,X_train, y_train.ravel(), cv, 'GradientBoostingClassifier')

eval(xgbc_pipeline, X_test, y_test)

"""## **Summary**

* We trained three multi-class classifiers for handwritten digit recognition.

* Firstly, the **AdaBoost classifier** obtained an accuracy of **71%** on the test set.

* Next ,using **Gradient boosting clssifier** gave us an increase of 12 percentage in the accuracy, which translates to **83%** accuracy on the test set.

* Finally, the **XGBoost classifier** pushed that further to **97%**.

* We can see that how ensemble techniques give better results on the classification task compared to a single classifier.

### **Objective**

In this notebook, we will apply ensemble techniques regression problem in california housing dataset.

We have already applied different regressors on california housing dataset. In this notebook, we will make use of :

  * AdaBoost regressor

  * Gradient Boosting regressor

  * XGBoost regressor

### **Importing basic libraries**
"""

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit

import warnings
warnings.filterwarnings('ignore')

np.random.seed(306)

"""Let's use `ShuffleSplit` as cv with 10 splits and 20% examples set aside as text examples."""

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

"""Let's download the data and split it into training and test sets."""

features, labels = fetch_california_housing(as_frame=True, return_X_y=True)
labels *= 100

com_train_features, test_features, com_train_labels, test_labels = train_test_split(
    features, labels, random_state=42)

train_features, dev_features, train_labels, dev_labels = train_test_split(
    com_train_features, com_train_labels, random_state=42)

"""### **Training different Regressors**

Let's train different regressors :
"""

def train_regressor(estimator, X_train, y_train, cv, name):
    cv_results = cross_validate(estimator,
                                X_train,
                                y_train,
                                cv=cv,
                                scoring='neg_mean_absolute_error',
                                return_train_score=True,
                                return_estimator=True)

    cv_train_error = -1 * cv_results['train_score']
    cv_test_error = -1 * cv_results['test_score']

    print(f'On an average, {name} makes an error of ',
          f'{cv_train_error.mean():.3f} (+/-) {cv_train_error.std():.3f} on the training set.')

    print(f'On an average, {name} makes an error of ',
          f'{cv_test_error.mean():.3f} (+/-) {cv_test_error.std():.3f} on the testing set.')

"""#### **AdaBoost Regressor**"""

train_regressor(AdaBoostRegressor(), com_train_features,com_train_labels, cv, 'AdaBoostRegressor')

"""#### **Gradient Boosting Regressor**"""

train_regressor(GradientBoostingRegressor(), com_train_features,
com_train_labels, cv, 'GradientBoostingRegressor')

"""#### **XGBoost Regressor**"""

train_regressor(XGBRegressor(), com_train_features,
                com_train_labels, cv, 'XGBoostRegressor')

"""### **Clustering**
Clustering is concerned about grouping objects with  *similar attributes* or *characteristics*

The objects in the same cluster are closer to one another than the objects from the other clusters.

![cluster_week11.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASwAAADJCAYAAACZgw/wAAARLElEQVR4nO3dfWxU9Z7H8U/JJjcaoICwBWypQnlIKXIpLmt03bhREbnYVkoDFxEpLV5cnm/4H/jXbIwrAWGvPMglhUWQoqDLbqKL14dkES3RK8UKlFqFCwW1Ef8x6dk/pkPn4Zw5Z9qZM/M75/1KJmc6v1/PfDu0H87T73cKLMuyJMmyLBUUFCjxuZ1ou9P3JH5/Yv/YpSRP6/L6Xonvmar+dNfv9b3s1jnQ90pnnV7eK936U31mqfqn+je3q8HLurx8T3/XmdgWrdXtZ3b6XfbymXr9rNL5d0rnbzBT68xkX7vfEUkalPQKAOQpAguAMQgsAMYgsAAYg8ACYAwCC4AxCCwAxiCwABiDwAJgDAILgDEILADGILAAGIPAAmAMAguAMQgsAMYgsAAYg8ACYAwCC4AxCCwAxiCwABiDwAJgDAILgDEILADGILAAGIPAAmAMAguAMQgsAMYgsAAYg8ACYAwCC4AxCCwAxvi7XBdgiubmZrW0tLj2mz9/vqZNm+ZDReHy7rvvau7cua795s2bp7fffjtrdZSXl6u1tdW139dff62JEydmrY6wIrA8am5u1uuvv+7ar6ysjMDKqkZJdzu0/YdPNUyVtMCh7QtJbya9un//fi1dutR1zfX19dq1a9eAqgsy4wLrl19+0Ysvvujab/Dgwdq4cWNG3/uee6RLl5JftyzpwgWJ/1D90CjpH3ufW5IKYpbHfaqhQtKmhPe2etuaZBdYfdZKGu7Q9nKmCgwsIwNry5Ytrv2KiooyHljAwK2TNL73eWLg7slVUcYw9qD71q2RLZuensgy9rFyZa6rA5ANxgYWgPAhsAAYg8ACYAwCC4AxjDtLmEs//iht3mzfdvOmr6WE2AMu7fN8qOE/ex/wG4GVhh9/lDxcUYEsKCsr06ZNm1z7TZo0Kat1rFq1Sl1dXa79RowY4dAyIbMFhUyBZVmWJFmWpYKCAiU+txNtd/qexO9P7B+7lORpXdFlV1eXRo0apSeflGbNsq/vxAnp22+LdPXq1ZT1p/uzOH0uia/fuHFDW7dudfz8okaOHKlVq1b16zPsz+te6/f6mTgt3fpISqojnXV5+Z7+rjOxLVqr28/s9LscdfbsWR09ejTps080Y8YMVVVV2X5W6fw7pfM3mKl1ZrJv7M8dy9jAclNUlLvAamtr8/Q//ZQpU/TVV18RWCn6p/Mz5nNgpfsZRZfjxo1TZ2dn0r9Zou+//15jxowJfGAZt0s4cuRI9fT0ePojdNPZ2amSkhLXfqWlpWpvb0+71v37pWeeia1JKiiILH//e+ns2bRXiVCaIanKoe2M/BuSlHvGBVY2PPWUVFlp33bsmPTDD/7WA8SrlLRZfUN4oixFBnwTWKFSVSU1NES2fqS+LSFJunxZev/93NUGfx0+fFh1dXWu/RYuXKiDBw/6UBFiEViArRck/b1D2zY/C0EMAguw9a+KzHslJU8jwzVYucKV7gCMEYgtrBMnTuj06dOu/ebNm6f777/fh4oAZEMgAuudd97R9u3bXfuNGTNmwIG1bNkyT1MlS9Kbb0ptbfZtX345oDKAUApEYEnSqFHStWt9X8ee6btyRRo7NnPvNWyYtG6dfdvNm5HJBaVIYL2ZYrbcKVMyVxPyW1lZmS5evOjar729XePGjUt4dVfvA4EJrIFYsSLycFJaGv/18OHxg6Bjw7GtLRJY+/fv1+LFiz1d7YuwuE/S0w5tLZKOJb26YcMGdXd3u6558ODBA6rMFKEOrKFDh3oaUDts2DAfqkHwTVffzSuk+DOPe+UUWOkO1woyAmvTprTGRfbHkSNH9MUXX7j2q6ur09SpU137wQ9ut2qbnvUKSktL9e2337r2++677zQ2k8c88lioA8svR44c0YEDB1z7lZeXE1g5Vl5e7mmru6KiwodqJMYRxiOwfDJpktTa2jf4OXYY0LlzEjmVH6KB5WW2Bn/MUGQcoZQ8ljBc4wglLhwFYJDAbGHduuU8ffHPP/taCoAsCUxg/fKLf9MXX7rUt0uH8Dpw4IAWL17s2m/JkiU+VBMOgQisbdu2adu2+BH02TrVW1NTo3vuuce137RpbmeZEByrJd3l0PZKzPM/9z7QX4EILD/V1NSopqYm7rVU18cgDFZLik6JHTuzgyTtlyStXbtWP3iYCbKwsDAL9QUHgZXC8uXLtWfPHtd++/bt07PPPutDRTDV2rVr0573HslCEVhXrlzxdGFdcXFx0oV6hYXS+vX2/X/4QXrlFfu2RDduOB9ju37d2zoQRrt7H5BCElhR8+ZJM2fat731ln1wFBb2nX2MvX5Kki5ezExgAXY2bNign376ybXfkCFDfKgmP4QusP7wh+QLNwsKpM5O6eTJ7LxvU1OTmpqaPN0+C4ha37tpH/v78dprr+n555+P67fF5n/ClStX6tVXX81+kT4LVWABwfFHSU5bVv/mZyG+IrAAI/1R0lgln5W0JO3IVVFZx9AcYMCmKPKnNEiR4IguCyRdyGFdwcMWFmx9/PHHeuihh1z7Pfzwwzp16pQPFeWfiooKTzM73HfffY5tLS0tam5udl1HZWWlnnrqqbTqCyICK4uWLFmipqYm134HDx7UwoULfagofUuWSBMm2Lft2+dvLflm2rRpqqiocJzZIZbTCZWWlhbbg+aJGhoaCCwRWK46OgY2bnDECGnNGvu2ri5pW57fk/PZZ6XZs+3PrP7lL9Kvv+a2vuC4JKlUyfdAlKTEOd7DK1SBtXJl5OGkuDj+6+rqapsbAiRLtcl/113J13FFl+fO5X9gAfnEuMDq6urSqFGjXPsVFRXp6tWrkiIT9Hs51jB06NC4r6urq1VVVeU6XTLXTwH+MC6wop58Upo1y77txAkpdoTNkCFDtDlmsqxU47kAMxS7dwkgYwNr7lxp9erk4TKS9Le/xQfWQHR0dHiaTmbChAn65ptv+vUeixYt0qJFi+JeO3TokOrq6vq1PgRXZWWlp72FoN7h3NjA8lt1tfTb39q3HT0amfG0vxYs6JvT/fp1ycNNrBFSM2fO1MzeAbGphnoF9TAFgeVRTY303HP2ZwwvXJA++aT/616wQFq4MLK1+NVXBBbghMBCSk88kbr94Yf9qSP4XpbkdMNe9zs/h0XgA+v48eP69NNPXftVVVWpsrIy4+/f1mbm/O8lJSWejpWUlpb6UE0Y/HuuCzBCKAJr586drv2Ki4szHljz58/XxIkT4147cuSIrlz5UqtX972Wj/ckjAZWqrthB/14iR+WLVum5557LuXMo0w/1CfwgSVJo0dLV670fR17ZrGzUyopyc77zp8/X7W1tTHva+n8+fP69dcvHScFBODM2Nka1qyJ/KEPGhRZxj52BHd2DSDUjNvCuvPOOz0dWxk8eLAP1QDwk7GBla3hMh0dHbYHkuvrIw8nTjMa2Dl/nt1AoD+MCyy/VFVJM2bYtzU3S5cvD9O6detuvzZixAhP662trdXkyZNd+5WXl3taHxAmBJaD6mpp+XL7aVUuXZK6u4ffHp+YzjjEBQsWqLa2NuXZt+hzAPGMPegOIHxCsYX18899c1Il6uYiYmTI5MmT1dbW5trvwoULGj9+vA8VBU9oAoubmMIf0yTNd2g7K8l9/nY4C3xg7dixQzsSLswK0+h2+G2apE3qu+1WlCXpzyKwBoZjWACMQWABMAaB5aChIXnoT/R52G9vBeRK4I9hpauwsNDT0J/hw4f7UA2AWARWgmhguV3YCcB/7BICMAZbWEBGNfU+kA0EFpAha9as0Y0bN1z7cfyz/wgs+Oq9997TY4895tpv9uzZOnnypA8VZc7q1asdb8rLDXszg8BCTtRLGufQtsvPQmAUAgs5US8p9g5hliKDWSxJ/52TimACzhICMAaBBQTQ6NGjVVBQcPsxaNAg2+XNmzdzXWpa2CUEAut+Sb9zaPs/Se/6WEtmEFhAYP2DpM29z2OPEhZI2ioTA4tdQgDGILAAGIPAQk78syI7JtHHoJjlJzmsC/mNY1jw1b333utp+p4J6dyZFqFBYMFX0cByG7rCHPuwwy4hAGOwhQUE1qu9j+AgsIAA2rhxo27duuXa74477vChmswhsPLMtWvXtH37dtd+o0eP1sqVK32oCCbauHFj3HFCu2OEJh4rJLDyzPXr17XFw22qp0+fTmAhdDjonqcOHZIsq+/R09O3fPrpXFcH5AaBBcAYBBYAYxBYCITjx4/fnuMp1RxQT7M/bTQOuiNQnpc0xqFth5+FICsILATK85Jm9j5PnAGqOVdFIWPYJQRgDLaw8tQbb0h//at927lz0m9+4289QD4gsPLUG2+kbp8+3Z86sm337t3q6Ohw7dfY2Ki7777bh4qQzwisPDN16lRZlpV0h+Cg3jl49+7d+uijj1z7zZkzh8ACx7CQe49J6lHk4Hji0rzbJCCbCCwAxmCXEIFyv0v7vb5UgWwhsBAIkyZN8jRX/JQpU3yoBtlCYCEQooHlZQ4omItjWACMwRYWAm/nzp26evWqa78XXnhBRUVFPlSE/iKwkHMXJTnNsfpNBta/c+dOtbS0uParra0lsPJcqAOru7tbL730kmu/YcOGaf369T5UFE6pAsvOrFmzdPr0add+Z86cUWVlpSSpWtJRRQZBS/EDo9+UtCCN90fuhD6wvMyfXlpaSmBlyYcffigp9QFyuxsmTJS02GGd5yUdzG7ZyJFQB1bUn/4kNTRI0RNIltX3vL5eev/93NUGexMlbVL89DHR5VsisIKKs4QAjEFgATAGgQXAGAQWAGNw0B1Zt2fPHl2+fNm13/Lly1VSUpKVGlrlfOnEuay8I7KBwELW7d27Vx988IFrv8cffzxrgXVe6V3rhfxEYEl66y2ps9O+zcMF0vDgXyS9p8ilB1LfBZyS9D+SZqe5vnfk/XjGZ599lvLaLgZEm4PAkvT225GHk9JS/2qBuxUrVmju3Lmu/caMcbpDIUwV6sAqLi5WT0+P4/++ic+RH1asWCHJffqY2CvjEQycJYQRHnjgAU+3oj9z5kyuS0UWhXoLC2Ypk/SMQ9vXkg74WAtyg8CCMcokbY75OnYM4QkRWGHALiEAY7CFBV9cUvzWUayLPtYBsxFY8EW7cnvhZmtrqw4edJ90pry8XHV1dT5UhP4gsJB1p06dSpqEL/FSEbtJ+jKptbXV02SNtbW1BFYe4xgWQuVzRQ7S9yQsLUkVOawL3hBYAIzBLiGM8V+KH4OI8CGwYITGxkbNmTPHtR/jB4ONwIIRGhsbbccKBuVW9J9//rmOHTvm2q+yslJVVVU+VJSfCCwgD7S0tHg6i9nY2BjqwOKgO5BH2hV/BjP2LObduSsrb7CFhYxrb2/X3r17XfuNHz9eS5cuzX5BMXZIGu3Qdk3SZB9rQfoILGRce3u7p92bRx991PfA2unruyHT2CVE1ryv5As0o7s5/+TyvQ8++KDjvFexr58+fdpTLTU1Nerp6ZFlWbIs6/bz2Ncsy9Lhw4cH8BMj29jCQt6aIGmJQ1ubpCYfa0F+ILCQt6LzX8XOexW9aOGECKwwYpcQgDHYwgLyyMuSCh3auv0sJE8RWEAeeTnXBeQ5AgvIA/X19Vq2bFnSUCNJ3HIuBoGFrNkr6X8d2jokTfStEgQFgYWsed2lncBCuggsZNwjjzySdEdttymS7ZwU818hHoGFvNTQ0KAnnnjCtd/YsWN9qAb5gsBCXmpoaEg62JxqPiyEAxeOAjAGgQXAGAQWAGMQWACMQWABMAaBBcAYBBYAYxBYAIxBYAEwBoEFwBgEFgBjEFgAjEFgATAGgQXAGAQWAGMQWACMQWABMAaBBcAYBBYAYxBYAIxBYAEwBoEFwBgEFgBjEFgAjEFgATDG/wNrSyj43mbVxQAAAABJRU5ErkJggg==)

In the image above, the clusters with same color share similar properties(feature values represented on axis).

For instance, if the x-axis represents weight and y-axis represents height, the yellow color cluster represents people with low BMI.

Similar interpretations can be drawn for the remaining clusters.

### **Hierarchical Agglomerative Clustering (HAC)**

Earlier in this week, we studied k-means clustering algorithm.

In this notebook, we will discuss another clustering algorithm which is **Hierarchical agglomerative clustering (HAC)** algorithm.

* Hierarchical clustering start by considering each datapoint as a cluster and then combines closest clusters to form larger clusters i.e it follows a **bottoms-up approach**.

* There is an alternate approach, which is **top-down approach**, where the entire data is considered as a one single cluster, which is divided to form smaller clusters in each step.

* This is another type of hierarchical clustering also known as **Divisive Hierarchical Clustering (DHC)**.


The merging and splitting decisions are influenced by certain conditions that will be discussed shortly.

### **Metrics**

Certain metrics are used for calculating similarity between clusters.

**Note:**  Metric is a generalization of concept of distance.

The metrics follow certain properties like :

1. non-negative

2. sysmetric

3. follows triangle inequality

Some of the popular metric functions are :

1. **Euclidean distance -**

\begin{align}
d(x^{(i)} , x^{(j)}) = \sqrt{\sum{^m _{l=1}} {(x_l^{(i)} - x_l^{(j)})^2}}
\end{align}

2. **Manhattan distance -**

\begin{align}
d(x^{(i)} , x^{(j)}) = \sum{^m _{l=1}} {\left\lvert(x_l^{(i)} - x_l^{(j)})\right\rvert}
\end{align}

3. **Cosine distance -**

\begin{align}
d(x^{(i)} , x^{(j)}) = 1 - \frac{x^{(i)}. x^{(j)}}{\left\lvert \left\lvert x^{(i)} \right\rvert \right\rvert \left\lvert \left\lvert x^{(j)} \right\rvert \right\rvert}

= 1 - \cos{\theta}
\end{align}

### **Linkage**

Linkage is a strategy for aggregating clusters.

There are four linkages that we will study :
* Single linkage

* Average linkage

* Complete linkage

* Ward's linkage

The **Single linkage** criterion merges clusters based on the shortest distance over all possible pairs i.e.

$ \left ({ \mathbf \{ x_{r_1}^{(i)}\}_{i=1}^{|r_1|} },{\mathbf \{ x_{r_2}^{(j)}\}_{j=1}^{|r_2|} } \right) = \text {min}_{i,j} d\left(\mathbf x_{r_1}^{(i)}, \mathbf x_{r_2}^{(j)}\right) $

![](Images/single_linkage.png)

The **Complete linkage** merges clusters to minimize the maximum distance between the clusters (in other words, the distance of the furthest elements)

$ \left ({ \mathbf \{ x_{r_1}^{(i)}\}_{i=1}^{|r_1|} },{\mathbf \{ x_{r_2}^{(j)}\}_{j=1}^{|r_2|} } \right) = \text {max}_{i,j} d\left(\mathbf x_{r_1}^{(i)}, \mathbf x_{r_2}^{(j)}\right) $

![](Images/complete_linkage.png)

The **average linkage** criterion uses average distance over all possible pairs between the groups for merging clusters.

$ \left ({ \mathbf \{ x_{r_1}^{(i)}\}_{i=1}^{|r_1|} },{\mathbf \{ x_{r_2}^{(j)}\}_{j=1}^{|r_2|} } \right) = \frac {1}{|r_1r_2|} \sum_{i=1}^{|r_1|} \sum_{j=1}^{|r_2|} d\left(\mathbf x_{r_1}^{(i)}, \mathbf x_{r_2}^{(j)}\right) $

![](Images/average_linkage.png)

**Ward's linkage**

It computes the sum of squared distances withing the clusters.


$ \left ({ \mathbf \{ x_{r_1}^{(i)}\}_{i=1}^{|r_1|} } , {\mathbf \{ x_{r_2}^{(j)}\}_{j=1}^{|r_2|} } \right) = \sum_{i=1}^{|r_1|} \sum_{j=1}^{|r_2|} ||(\mathbf x_{r_1}^{(i)} - \mathbf x_{r_2}^{(j)} ||^2 $

### **Algorithm :**

1. Calculate the distance matrix between pairs of clusters.

2. While all the objects are clustered into one.
    * Detect the two closest groups (clusters) and merge them.

### **Dendrogram**

Dendrograms are a graphical representation of the agglomerative process which shows a how aggregation happens at each level.

### **Importing Libraries**
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns ;sns.set()

from sklearn.preprocessing import normalize

"""Lets take example of a toy dataset to understand this :"""

X = np.array([(8, 3), (5, 3), (6, 4), (1, 6), (2, 8)])
X_scaled = normalize(X)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

"""Let's plot the dendrogram with `scipy.cluster.hierarchy` library

"""

import scipy.cluster.hierarchy as sch

plt.figure(figsize=(8, 8))
plt.title('Dendrogram')
dend = sch.dendrogram(sch.linkage(X_scaled, method='ward'))

"""HAC is implemented in `sklearn.cluster` module as `AgglomerativeClustering` class.

### **Objective**

In this notebook, we will implement k-means algorithm using `sklearn`.

### **Importing Libraries**
"""

from IPython.display import display, Math, Latex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

from sklearn.cluster import KMeans

from sklearn.datasets import load_digits

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

"""### **Clustering of digits**

We will use digit dataset for clustering, which is loaded through the `load_digits` API.

* It loads `8x8` digit images with approximately **180 samples per class**.

* From 10 classes, it has **total of 1797 images**.

* Each pixel has value **between 0 and 16**.
"""

digits = load_digits()

data = np.column_stack((digits.data, digits.target))
columns = np.append(digits.feature_names, ['targets'])

df_digits = pd.DataFrame(data, columns=columns)

df_digits.head()

df_digits.info()

"""Some of the important parameters of KMeans are as follows :
* `init`

* `n_init`

* `max_iter`

* `random_state`

Since `KMeans` algorithm is susceptible to local minima, we perform multiple `KMeans` fit and select the ones with the lowest value of sum of squared error.

* The total number of time, we would like to run `KMeans` algorithm is specified through `n_init` parameter.

* `max_iter` specifies total number of iterations to perform before declaring the convergence.

Let's define parameters of KMeans clustering algorithm in a dictionary object.
"""

kmeans_kwargs = {
    'init': 'random',
    'n_init': 50,
    'max_iter': 500,
    'random_state': 0
}

"""### **Model Building**

Let's define a `pipeline` with two stages :

* preprocessing for feature scaling with `MinMaxScaler`.

* clustering with `KMeans` clustering algorithm.
"""

pipeline = Pipeline([('scaler', MinMaxScaler()),
                     ('clustering', KMeans(n_clusters=10, **kmeans_kwargs))])

pipeline.fit(digits.data)

"""The cluster centroids can be accessed via `cluster_centers_` member variable of `KMeans` class

"""

cluster_centers = pipeline[-1].cluster_centers_
cluster_centers.shape

"""##### **Dispalying the cluster centroids.**"""

fig, ax = plt.subplots(5, 2, figsize=(8,8))
for i, j in zip(ax.flat, cluster_centers.reshape(10, 8, 8)):
    i.imshow(j)

fig, ax = plt.subplots(2, 5, figsize=(8,8))
for i, j in zip(ax.flat, cluster_centers.reshape(10, 8, 8)):
    i.imshow(j)

"""In this case, the number of clusters were known, hence we set k=10 and got the clusters.

##### For deciding the optimal number of clusters through **elbow and silhouette**, we will pretend that we do not know the clusters in the data and we will try to discover the optimal number of clusters through these two methods one by one:

### **Elbow method**

Here we keep track of sum-of-squared error (SSE) in a list for each value of `k`.
"""

sse_digit = []
scaled_digits = MinMaxScaler().fit_transform(digits.data)

for k in range(1, 12):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_digits)
    sse_digit.append(kmeans.inertia_)

"""Note that the SSE for a given clustering output is obtained through `inertia_`."""

plt.plot(range(1, 12), sse_digit)
plt.xticks(range(1, 15))
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

"""There is a slight elbow at **k=9**, which could point to the fact that a few digits may have been merged in one cluster.

### **Silhoutte Score**
"""

sil_coef_digits = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_digits)
    score = silhouette_score(digits.data, kmeans.labels_)
    sil_coef_digits.append(score)

plt.plot(range(2, 15), sil_coef_digits)
plt.xticks(range(2, 15))

plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.grid(True)
plt.show()

"""Get the value of K for which the Silhouette coefficient is the highest."""

# 2 is added since iteration is starting at 2.
print(np.argmax(sil_coef_digits) + 2)

"""#### **Objective**

In this notebook, we will demonstrate working of `MLPClassifier` to classify handwritten digits in `MNIST` dataset.

#### **Importing Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, plot_confusion_matrix

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedShuffleSplit

import warnings
warnings.filterwarnings('ignore')

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)

"""#### **Loading the dataset**
Lets use the MNIST dataet for the demo of MLPClassifier.
"""

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.to_numpy()
y = y.to_numpy()

"""##### **Train test split**"""

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print('Shape of training data before flattening : ',X_train.shape)
print('Shape of testing data before flattening : ',X_test.shape)

"""##### **Reshaping**"""

X_train = X_train.reshape(X_train.shape[0] ,28*28)
X_test = X_test.reshape(X_test.shape[0] ,28*28)

"""##### **Normalizing**"""

X_train = X_train / 255
X_test = X_test / 255

print('Shape of training data after flattening : ',X_train.shape)
print('Shape of testing data after flattening : ',X_test.shape)

print('Shape of training data : ',X_train.shape)
print('Shape of testing data : ', X_test.shape)
print('Shape of training labels : ',y_train.shape)
print('Shape of testing labels :', y_test.shape)

"""### **Fit MLPClassifier on MNIST dataset**

Let us train a MLPClassifier with one hidden layer having 128 neurons.
"""

mlpc = MLPClassifier(hidden_layer_sizes=(128,))
mlpc.fit(X_train, y_train)

cv_score = cross_val_score(mlpc, X_train, y_train.ravel(), cv=cv)
print('Training accuracy : {:.2f} %'.format(cv_score.mean() *100))

"""**Prediction probabilities on testing data**"""

mlpc.predict_proba(X_test[:5])

"""**Prediction of class labels of testing data**"""

y_pred = mlpc.predict(X_test)

print('Training accuracy : {:.2f}'.format(accuracy_score(y_train, mlpc.predict(X_train)) *100))

print('Testing accuracy : {:.2f}'.format(accuracy_score(y_test, y_pred) *100))

"""**Confusion Matrix**"""

cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='.4g', cmap='Reds')
plt.show()

"""**Classification Report**"""

print(classification_report(y_test, y_pred))

"""#### **Plot of test data along with predicted class labels**"""

fig = plt.figure(figsize=(10,8))
for i in range(25):
    ax = fig.add_subplot(5, 5, i+1)
    ax.imshow(X_test[i].reshape(28,28), cmap=plt.get_cmap('gray'))
    ax.set_title('Label (y): {y}'.format(y=y_pred[i]))
    plt.axis('off')

"""#### **Visualization of MLP weights in hidden layer**

* Looking at the learned coefficients of a neural network can provide insiht into the learning behaviour.

* The input data comtains 784 features in the dataset.

* We have used one hidden layer with 128 neurons. Therefore, weight matrix has the shape (784, 128).

* We can therefore visualize a single column of the weight matrix as a 28x28 pixel image.
"""

w = mlpc.coefs_
w = np.array(w[0])
w.shape

w1 = np.array(w[:,0])
w1.shape

w_matrix = w1.reshape(28,28)
fig = plt.figure()
plt.imshow(w_matrix, cmap='gray')
plt.grid(False)
plt.axis(False)
plt.colorbar()
plt.show()

fig, axes = plt.subplots(4,4)
vmin, vmax = mlpc.coefs_[0].min(), mlpc.coefs_[0].max()

for coef, ax in zip(mlpc.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28,28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()

"""#### **Loss Curve**"""

plt.plot(mlpc.loss_curve_)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.show()

"""## **MLP Regressor**

MLPRegressor implements a multi-layer perceptron (MLP) that trains using backpropagation with no activation function in the output layer.

Therefore, it uses the square error as the loss function, and the output is a set of continuous values.

### **Importing Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from sklearn.pipeline import Pipeline ,make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from sklearn.model_selection import train_test_split ,GridSearchCV ,RandomizedSearchCV, cross_validate, ShuffleSplit

import warnings
warnings.filterwarnings('ignore')

np.random.seed(306)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

"""#### **Loading the dataset**

This dataset can be fetched from sklearn with `fetch_california_housing` API.
"""

from sklearn.datasets import fetch_california_housing
X,y = fetch_california_housing(return_X_y=True)

print('Shape of feature matrix : ' ,X.shape)
print('Shape of label vector : ',y.shape)

"""#### **Split data into train & test sets**"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

print('Shape of training feature matrix : ' ,X_train.shape)
print('Shape of training label vector : ',y_train.shape)
print()
print('Shape of test feature matrix : ' ,X_test.shape)
print('Shape of test label vector : ',y_test.shape)

"""#### **Fit a pipeline to implement MLPRegressor**

Let's train a MLPRegressor with 3 hidden layers having 128 neurons each.
"""

pipe = Pipeline([('scaler', StandardScaler()),
                 ('regressor', MLPRegressor(hidden_layer_sizes=(32)))])

cv_results = cross_validate(pipe,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring="neg_mean_absolute_percentage_error",
                            return_train_score=True,
                            return_estimator=True)

mlp_train_error = -1 * cv_results['train_score']
mlp_test_error = -1 * cv_results['test_score']


print(
    f"Mean absolute error of MLP regressor model on the train set :\n" f"{mlp_train_error.mean():.3f} +/- {mlp_train_error.std():.3f}")

print()
print(
    f"Mean absolute error of MLP regressor model on the test set :\n" f"{mlp_test_error.mean():.3f} +/- {mlp_test_error.std():.3f}")

pipe.fit(X_train, y_train)

mean_absolute_percentage_error(y_train, pipe.predict(X_train))

mean_absolute_percentage_error(y_test, pipe.predict(X_test))

"""#### **Plotting Predicitons**"""

plt.figure(figsize=(8,8))
plt.plot(y_test, pipe.predict(X_test), 'b.')
plt.plot(y_test, y_test ,'g-')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
