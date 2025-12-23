"""
Unit Tests: Metric Preprocessing
Tests for container name extraction and preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.metric_specific import (
    extract_container_name,
    build_container_vocabulary,
    add_container_ids,
    MetricPreprocessor,
    MetricType
)


@pytest.mark.unit
def test_extract_container_name_standard():
    """Test extracting container name from standard Prometheus labels."""
    # Standard format
    labels = "container=webapp,pod=test-webapp-12345,namespace=default"
    assert extract_container_name(labels) == "webapp"

    # Different container
    labels = "container=database,pod=test-db,namespace=production"
    assert extract_container_name(labels) == "database"

    # Case insensitive
    labels = "Container=Redis,pod=test"
    assert extract_container_name(labels) == "Redis"


@pytest.mark.unit
def test_extract_container_name_fallback():
    """Test fallback to pod name when container label missing."""
    # No container= label, use pod=
    labels = "pod=webapp-abc123,namespace=default"
    result = extract_container_name(labels)
    assert result == "webapp"  # Strips trailing hash

    # Pod name without hash
    labels = "pod=redis,namespace=default"
    result = extract_container_name(labels)
    assert result == "redis"


@pytest.mark.unit
def test_extract_container_name_errors():
    """Test error handling for invalid labels."""
    # Empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_container_name("")

    # None
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_container_name(None)

    # No container or pod labels
    with pytest.raises(ValueError, match="Could not extract"):
        extract_container_name("namespace=default,service=api")


@pytest.mark.unit
def test_build_container_vocabulary():
    """Test building vocabulary from DataFrame."""
    df = pd.DataFrame({
        'container_labels': [
            'container=webapp,pod=test1',
            'container=database,pod=test2',
            'container=webapp,pod=test3',
            'container=redis,pod=test4',
            'container=database,pod=test5',
        ],
        'value': [1, 2, 3, 4, 5]
    })

    vocab = build_container_vocabulary(df)

    assert vocab.num_containers == 3
    assert vocab.contains('webapp')
    assert vocab.contains('database')
    assert vocab.contains('redis')

    # Check IDs are assigned
    assert vocab.get_id('database') >= 0
    assert vocab.get_id('redis') >= 0
    assert vocab.get_id('webapp') >= 0


@pytest.mark.unit
def test_add_container_ids():
    """Test adding container IDs to DataFrame."""
    df = pd.DataFrame({
        'container_name': ['webapp', 'database', 'webapp', 'redis'],
        'value': [1, 2, 3, 4]
    })

    vocab = build_container_vocabulary(
        pd.DataFrame({
            'container_labels': [
                'container=webapp,pod=test',
                'container=database,pod=test',
                'container=redis,pod=test'
            ]
        })
    )

    df = add_container_ids(df, vocab)

    assert 'container_id' in df.columns
    assert len(df) == 4

    # Verify IDs are correct
    webapp_id = vocab.get_id('webapp')
    database_id = vocab.get_id('database')
    redis_id = vocab.get_id('redis')

    assert df.loc[0, 'container_id'] == webapp_id
    assert df.loc[1, 'container_id'] == database_id
    assert df.loc[2, 'container_id'] == webapp_id
    assert df.loc[3, 'container_id'] == redis_id


@pytest.mark.unit
def test_add_container_ids_missing_column():
    """Test error when container_name column is missing."""
    df = pd.DataFrame({
        'value': [1, 2, 3]
    })

    vocab = build_container_vocabulary(
        pd.DataFrame({
            'container_labels': ['container=webapp,pod=test']
        })
    )

    with pytest.raises(ValueError, match="must have 'container_name' column"):
        add_container_ids(df, vocab)


@pytest.mark.unit
def test_metric_preprocessor_cpu():
    """Test CPU metric preprocessing."""
    preprocessor = MetricPreprocessor(MetricType.CPU)

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15s'),
        'container_labels': ['container=webapp,pod=test'] * 100,
        'value': np.random.uniform(0.3, 0.9, 100),  # Fraction (0-1)
    })

    result = preprocessor.process(df, container_name='webapp')

    # CPU should be converted to percentage
    assert result['value'].min() >= 0
    assert result['value'].max() <= 100

    # Should have required columns
    assert 'timestamp' in result.columns
    assert 'value' in result.columns
    assert 'container_name' in result.columns
    assert 'metric_name' in result.columns


@pytest.mark.unit
def test_metric_preprocessor_memory():
    """Test memory metric preprocessing."""
    preprocessor = MetricPreprocessor(MetricType.MEMORY)

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15s'),
        'container_labels': ['container=webapp,pod=test'] * 100,
        'value': np.random.uniform(1e9, 2e9, 100),  # Bytes
    })

    result = preprocessor.process(df, container_name='webapp')

    # Memory should be converted to MB
    assert result['value'].min() > 0
    assert result['value'].max() < 3000  # Less than 3GB in MB


@pytest.mark.unit
def test_metric_preprocessor_multi_container():
    """Test preprocessing with multiple containers."""
    preprocessor = MetricPreprocessor(MetricType.CPU)

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=300, freq='15s').tolist() * 3,
        'container_labels': (
            ['container=webapp,pod=test1'] * 300 +
            ['container=database,pod=test2'] * 300 +
            ['container=redis,pod=test3'] * 300
        ),
        'value': np.random.uniform(0.3, 0.9, 900),
    })

    # Process all containers
    result = preprocessor.process(df, container_name=['webapp', 'database', 'redis'])

    # Should have data for all containers
    assert result['container_name'].nunique() == 3
    assert 'webapp' in result['container_name'].values
    assert 'database' in result['container_name'].values
    assert 'redis' in result['container_name'].values

    # Each container should have roughly equal samples
    counts = result['container_name'].value_counts()
    assert counts.min() > 0
    assert counts.max() / counts.min() < 2  # Not too unbalanced


@pytest.mark.unit
def test_metric_preprocessor_outlier_handling():
    """Test outlier detection and clipping."""
    preprocessor = MetricPreprocessor(
        MetricType.CPU,
        config={'outlier_threshold': 2.0}  # 2 standard deviations
    )

    # Create data with outliers
    values = np.concatenate([
        np.random.normal(50, 5, 95),  # Normal values
        np.array([200, 250, 300, 400, 500])  # Outliers
    ])

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15s'),
        'container_labels': ['container=webapp,pod=test'] * 100,
        'value': values / 100,  # Convert to fraction
    })

    result = preprocessor.process(df, container_name='webapp')

    # Outliers should be clipped
    assert result['value'].max() < 150  # Much less than the 500 outlier
    assert result['value'].min() >= 0
