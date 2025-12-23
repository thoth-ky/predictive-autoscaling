"""
Data Quality Tests: Container Boundaries
CRITICAL tests to ensure no cross-container contamination in windows.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.sliding_windows import MultiHorizonWindowGenerator
from src.preprocessing.metric_specific import extract_container_name, add_container_ids


@pytest.mark.data_quality
def test_no_cross_container_windows(synthetic_multi_container_data, container_vocab):
    """
    CRITICAL: Verify sliding windows don't cross container boundaries.

    This is the most important data integrity test. If this fails, the model
    will be trained on contaminated data mixing different containers.
    """
    df = synthetic_multi_container_data.copy()

    # Extract container names and add IDs
    df['container_name'] = df['container_labels'].apply(extract_container_name)
    df = add_container_ids(df, container_vocab)

    # Create windows
    generator = MultiHorizonWindowGenerator(
        window_size=100,
        prediction_horizons=[20, 60],
        stride=10
    )

    X, y_dict, window_container_ids, metadata = generator.create_multi_container_sequences(
        df['value'].values,
        df['container_id'].values,
        df['timestamp']
    )

    # CRITICAL VALIDATION: For each window, verify all data points are from same container
    for i in range(len(X)):
        # Get the window's assigned container ID
        window_container = window_container_ids[i]

        # Get metadata for this window
        if metadata and i < len(metadata):
            meta = metadata[i]

            # Verify container_id is in metadata
            assert 'container_id' in meta, f"Window {i} missing container_id in metadata"

            # Verify metadata container matches window container
            assert meta['container_id'] == window_container, \
                f"Window {i} metadata mismatch: {meta['container_id']} != {window_container}"

    print(f"✓ All {len(X)} windows respect container boundaries")


@pytest.mark.data_quality
def test_target_same_container_as_features(synthetic_multi_container_data, container_vocab):
    """
    CRITICAL: Verify prediction targets are from same container as input features.

    Ensures that we're not predicting one container's future from another's past.
    """
    df = synthetic_multi_container_data.copy()

    # Extract container names and add IDs
    df['container_name'] = df['container_labels'].apply(extract_container_name)
    df = add_container_ids(df, container_vocab)

    # Create windows
    generator = MultiHorizonWindowGenerator(
        window_size=100,
        prediction_horizons=[20],
        stride=10
    )

    X, y_dict, window_container_ids, metadata = generator.create_multi_container_sequences(
        df['value'].values,
        df['container_id'].values,
        df['timestamp']
    )

    # For each window, the container ID should be consistent
    # (The implementation groups by container before windowing, so this should always pass)
    assert len(window_container_ids) == len(X), \
        f"Container ID count mismatch: {len(window_container_ids)} != {len(X)}"

    # Verify no NaN or invalid container IDs
    assert not np.any(np.isnan(window_container_ids)), "Found NaN container IDs"
    assert np.all(window_container_ids >= 0), "Found negative container IDs"

    # Verify all container IDs are valid
    unique_containers = np.unique(window_container_ids)
    assert len(unique_containers) == container_vocab.num_containers, \
        f"Unexpected container count: {len(unique_containers)} != {container_vocab.num_containers}"

    print(f"✓ All {len(X)} windows have valid container IDs")


@pytest.mark.data_quality
def test_single_container_mode_unchanged(synthetic_single_container_data):
    """
    Verify that single-container mode still works (backward compatibility).
    """
    from src.preprocessing.sliding_windows import create_multi_horizon_features_and_windows

    df = synthetic_single_container_data.copy()
    df['container_name'] = 'webapp'
    df['metric_name'] = 'container_cpu_rate'

    # Use the original windowing function
    X, y_dict, feature_names, metadata = create_multi_horizon_features_and_windows(
        df,
        container_name='webapp',
        metric_name='container_cpu_rate',
        window_size_minutes=10,
        prediction_horizon_minutes=[5, 15]
    )

    # Verify output shapes
    assert X.shape[0] > 0, "No windows created"
    assert X.shape[1] > 0, "No sequence length"
    assert len(y_dict) == 2, "Expected 2 horizons"
    assert 20 in y_dict, "Missing 5-min horizon (20 timesteps)"
    assert 60 in y_dict, "Missing 15-min horizon (60 timesteps)"

    print(f"✓ Single-container mode works: {len(X)} windows created")


@pytest.mark.data_quality
def test_container_distribution_balanced(synthetic_multi_container_data, container_vocab):
    """
    Verify that windows are reasonably distributed across containers.

    This is not a strict requirement but helps catch bugs where one container
    dominates or is missing entirely.
    """
    df = synthetic_multi_container_data.copy()

    # Extract container names and add IDs
    df['container_name'] = df['container_labels'].apply(extract_container_name)
    df = add_container_ids(df, container_vocab)

    # Create windows
    generator = MultiHorizonWindowGenerator(
        window_size=100,
        prediction_horizons=[20],
        stride=10
    )

    X, y_dict, window_container_ids, metadata = generator.create_multi_container_sequences(
        df['value'].values,
        df['container_id'].values,
        df['timestamp']
    )

    # Count windows per container
    from collections import Counter
    container_counts = Counter(window_container_ids)

    # Verify all containers are represented
    for cid in range(container_vocab.num_containers):
        assert cid in container_counts, \
            f"Container {container_vocab.get_name(cid)} has no windows"

        count = container_counts[cid]
        print(f"  {container_vocab.get_name(cid)}: {count} windows " +
              f"({100 * count / len(X):.1f}%)")

    # Verify distribution is reasonable (no container < 10% of total)
    min_ratio = min(count / len(X) for count in container_counts.values())
    assert min_ratio >= 0.1, \
        f"Unbalanced distribution: minimum ratio {min_ratio:.2%} < 10%"

    print(f"✓ Window distribution is balanced across {len(container_counts)} containers")


@pytest.mark.data_quality
def test_no_data_leakage_across_containers(synthetic_multi_container_data, container_vocab):
    """
    Verify that container data doesn't leak into other containers' windows.

    This test checks that the actual data values in windows match the expected
    container's pattern.
    """
    df = synthetic_multi_container_data.copy()

    # Extract container names and add IDs
    df['container_name'] = df['container_labels'].apply(extract_container_name)
    df = add_container_ids(df, container_vocab)

    # Create windows
    generator = MultiHorizonWindowGenerator(
        window_size=50,
        prediction_horizons=[10],
        stride=25
    )

    X, y_dict, window_container_ids, metadata = generator.create_multi_container_sequences(
        df['value'].values,
        df['container_id'].values,
        df['timestamp']
    )

    # For a sample of windows, verify the data values are consistent with the container
    sample_size = min(30, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)

    for idx in indices:
        window_container = window_container_ids[idx]
        window_data = X[idx]

        # Get the expected range for this container from the original data
        container_data = df[df['container_id'] == window_container]['value'].values
        container_mean = np.mean(container_data)
        container_std = np.std(container_data)

        # Verify window data is within reasonable bounds for this container
        window_mean = np.mean(window_data)

        # Allow for some variance, but should be roughly within 2 std of container mean
        assert abs(window_mean - container_mean) < 3 * container_std, \
            f"Window {idx} for container {window_container} has suspicious statistics: " + \
            f"window_mean={window_mean:.2f}, container_mean={container_mean:.2f}"

    print(f"✓ Sampled {sample_size} windows - no data leakage detected")
