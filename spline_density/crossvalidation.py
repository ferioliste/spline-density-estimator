import numpy as np
import numbers

def kfold_crossvalidation(
    estimator,
    sample,
    degree,
    q,
    lambda_vector = None,
    n_folds = 5,
    rng = None,
    verbose = True,
    output_stats = False,
    **estimator_kwargs,
    ):
    
    ## Data validation ##
    if not callable(estimator):
        raise ValueError("estimator must be a callable")
    sample = np.asarray(sample, dtype=float).ravel()
    n = sample.size
    
    if lambda_vector is None:
        lambda_vector = np.logspace(0, -8, 9)
    lambda_vector = np.asarray(lambda_vector, dtype=float).ravel()
    
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("`n_folds` must be at least 2.")
    if n_folds > n:
        raise ValueError("`n_folds` cannot be larger than the sample size.")
    
    if rng is None or isinstance(rng, numbers.Integral):
        rng = np.random.default_rng(rng)
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("`rng` must be None, an integer, or a numpy.random.Generator.")
    
    if "t" not in estimator_kwargs:
        if "L" not in estimator_kwargs:
            estimator_kwargs["L"] = min(sample)
        if "U" not in estimator_kwargs:
            estimator_kwargs["U"] = max(sample)
    
    ## Define variables ##
    n_lambda = lambda_vector.size
    cv_scores = np.zeros(n_lambda, dtype=float)
    
    total_steps = n_folds * n_lambda
    step = 0
    
    fold_indices = np.array_split(rng.permutation(n), n_folds)
    
    ## Iterate ##
    for i_lambda, lambda_n in enumerate(lambda_vector):
        fold_scores = np.zeros(n_folds, dtype=float)

        for i_fold in range(n_folds):
            test_idx = fold_indices[i_fold]
            train_idx = np.concatenate( [fold_indices[j] for j in range(n_folds) if j != i_fold] )

            train = sample[train_idx]
            test = sample[test_idx]

            fitted = estimator(
                sample=train,
                degree=degree,
                q=q,
                lambda_n=lambda_n,
                **estimator_kwargs,
            )
            
            fold_scores[i_fold] = -np.mean(fitted.logpdf(test))

            step += 1
            if verbose and (step % max(1,total_steps//20) == 0 or step == total_steps):
                print(f"Completed {step}/{total_steps} ({100.0 * step / total_steps:.2f}%)")

        cv_scores[i_lambda] = fold_scores.mean()
    
    ## Retrain model ##
    best_idx = int(cv_scores.argmin())
    best_lambda = float(lambda_vector[best_idx])

    if verbose:
        print(f"\nSelected lambda_n: {best_lambda}")
    
    estimator_kwargs["output_stats"] = output_stats
    best_estimator_tuple = estimator(
        sample=sample,
        degree=degree,
        q=q,
        lambda_n=best_lambda,
        **estimator_kwargs,
    )
    
    if output_stats:
        best_estimator = best_estimator_tuple[0]
        stats = best_estimator_tuple[1]
        stats["lambda_vector"] = lambda_vector
        stats["n_folds"] = n_folds
        stats["best_lambda"] = best_lambda
        stats["cv_scores"] = cv_scores
        return best_estimator, stats
    else:
        best_estimator = best_estimator_tuple
        return best_estimator

def simple_validation(
    estimator,
    sample,
    degree,
    q,
    lambda_vector = None,
    test_ratio = 0.2,
    rng = None,
    verbose = True,
    output_stats = False,
    **estimator_kwargs,
    ):
    
    ## Data validation ##
    if not callable(estimator):
        raise ValueError("estimator must be a callable")
    sample = np.asarray(sample, dtype=float).ravel()
    n = sample.size

    if lambda_vector is None:
        lambda_vector = np.logspace(0, -8, 9)
    lambda_vector = np.asarray(lambda_vector, dtype=float).ravel()
    
    test_ratio = float(test_ratio)
    if not 0. < test_ratio < 1.:
        raise TypeError("test_ratio must be between 0 and 1 (both excluded)")
    n_test = int(test_ratio * n)
    if n_test == 0:
        raise TypeError("test_ratio is invalid because the test set is empty")
    if n_test == n:
        raise TypeError("test_ratio is invalid because the train set is empty")
    
    if rng is None or isinstance(rng, numbers.Integral):
        rng = np.random.default_rng(rng)
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("`rng` must be None, an integer, or a numpy.random.Generator.")
    
    if "t" not in estimator_kwargs:
        if "L" not in estimator_kwargs:
            estimator_kwargs["L"] = min(sample)
        if "U" not in estimator_kwargs:
            estimator_kwargs["U"] = max(sample)
    
    ## Define variables ##
    n_lambda = lambda_vector.size
    cv_scores = np.zeros(n_lambda, dtype=float)
    
    total_steps = n_lambda
    step = 0
    
    perm = rng.permutation(n)
    train_idx = perm[n_test:]
    test_idx  = perm[:n_test]

    train = sample[train_idx]
    test = sample[test_idx]
    
    ## Iterate ##
    for i_lambda, lambda_n in enumerate(lambda_vector):
        fitted = estimator(
            sample=train,
            degree=degree,
            q=q,
            lambda_n=lambda_n,
            **estimator_kwargs,
        )
        
        cv_scores[i_lambda] = -np.mean(fitted.logpdf(test))
        
        step += 1
        if verbose and (step % max(1,total_steps//20) == 0 or step == total_steps):
            print(f"Completed {step}/{total_steps} ({100.0 * step / total_steps:.2f}%)")
    
    ## Retrain model ##
    best_idx = int(cv_scores.argmin())
    best_lambda = float(lambda_vector[best_idx])

    if verbose:
        print(f"\nSelected lambda_n: {best_lambda}")
    
    estimator_kwargs["output_stats"] = output_stats
    best_estimator_tuple = estimator(
        sample=sample,
        degree=degree,
        q=q,
        lambda_n=best_lambda,
        **estimator_kwargs,
    )
    
    if output_stats:
        best_estimator = best_estimator_tuple[0]
        stats = best_estimator_tuple[1]
        stats["lambda_vector"] = lambda_vector
        stats["test_ratio"] = test_ratio
        stats["best_lambda"] = best_lambda
        stats["cv_scores"] = cv_scores
        return best_estimator, stats
    else:
        best_estimator = best_estimator_tuple

        return best_estimator
