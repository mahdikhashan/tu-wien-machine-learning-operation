def preprocess(df, split_data=False, seed=42):
    """
    Preprocesses the input DataFrame by handling missing values, splitting data
    (optionally), and applying TF-IDF and One-Hot Encoding transformations.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw job posting data.
        train_test_split (bool): If True, splits the data into training and testing sets.
                                 If False, processes the entire dataset. Defaults to False.
        seed (int): Random state for reproducibility if train_test_split is True.

    Returns:
        tuple:
            If train_test_split is True:
                (y_train, X_train_processed_df, y_test, X_test_processed_df)
            If train_test_split is False:
                (y_processed, X_processed_df)
            Where:
                - y_train/y_processed: Target variable (normalized_salary).
                - X_train_processed_df/X_processed_df: Preprocessed feature DataFrame.
                - y_test: Test set target variable (only if train_test_split is True).
                - X_test_processed_df: Preprocessed test set feature DataFrame (only if train_test_split is True).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer

    df_processed = df.copy()

    target = "normalized_salary"
    features = [
        "company_name",
        "title",
        "description",
        "location",
        "remote_allowed",
        "work_type",
    ]
    df_processed.dropna(subset=[target], inplace=True)

    X = df_processed[features]
    y = df_processed[target]

    if split_data:
        print(f"Splitting data into training and test sets (test_size=0.2, random_state={seed})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        print("Data split complete.")
    else:
        print("Proceeding without train-test split. All data will be used for training/processing.")
        X_train, y_train = X, y
        X_test, y_test = pd.DataFrame(), pd.Series(dtype='float64')
        print("No train-test split performed.")

    categorical_features = [
        "remote_allowed",
        "work_type",
        "company_name",
        "location",
    ]

    # TF-IDF transformer for 'title'
    # TODO(mahdi): The 'max_features' here is a hyperparameter chosen to limit
    #              the vocabulary size to the most frequent words, preventing
    #              the feature matrix from becoming too large and sparse.
    #              Changing it will change the number of output features, which
    #              needs to be consistent if you're expecting a fixed shape
    #              or if models are trained on specific feature counts.
    title_transformer = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=1000, ngram_range=(1, 1), stop_words="english"
                ),
            )
        ]
    )

    # TF-IDF transformer for 'description'
    # TODO(mahdi): Similar to 'title', 'max_features' for description controls
    #              the dimensionality. A larger value (2000 vs 1000) for
    #              description often makes sense as descriptions are typically
    #              longer and contain more unique terms.
    description_transformer = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=2000, ngram_range=(1, 1), stop_words="english"
                ),
            )
        ]
    )

    # One-Hot Encoder for categorical features
    cat_transformer = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("title_tfidf", title_transformer, "title"),
            ("desc_tfidf", description_transformer, "description"),
            (
                "cat_onehot",
                cat_transformer,
                categorical_features,
            ),
        ],
        # Dropping numerical features related to salary (e.g., raw_salary, min_salary, max_salary)
        # TODO(mahdi): Dropping these features is crucial to prevent data leakage.
        #              If features directly derived from the target variable (like salary range
        #              columns that were used to calculate 'normalized_salary') are included,
        #              the model could "cheat" by learning these features instead of
        #              generalizable patterns from job posting text and categories.
        #              This leads to artificially high performance on training/test data
        #              but poor generalization to new, unseen data.
        remainder="drop",
    )

    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)
    print("Preprocessor fitted.")

    print("Transforming training data...")
    X_train_processed = preprocessor.transform(X_train)
    print(f"Training data transformed. Shape: {X_train_processed.shape}")

    try:
        feature_names = preprocessor.get_feature_names_out()
        print(f"Successfully retrieved {len(feature_names)} feature names.")
        print("Sample feature names:", feature_names[:20])  # print some names
    except Exception as e:
        print(f"Could not get feature names automatically: {e}")
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    print("Processed Training DataFrame created.")
    print(X_train_processed_df.head())

    if split_data:
        print("\nTransforming test data...")
        X_test_processed = preprocessor.transform(X_test)
        print(f"Test data transformed. Shape: {X_test_processed.shape}")

        X_test_processed_df = pd.DataFrame(
            X_test_processed, columns=feature_names, index=X_test.index
        )
        print("Processed Test DataFrame created.")
        print(X_test_processed_df.head())

        return (y_train, X_train_processed_df, y_test, X_test_processed_df)
    else:
        # If no split, X_train_processed_df contains all processed features,
        # and y_train contains the full target.
        print("\nReturning full processed dataset (no test set).")
        return (y_train, X_train_processed_df)
