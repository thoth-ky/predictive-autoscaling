"""
Unit Tests: Container Vocabulary
Tests for container name to ID mapping functionality.
"""

import pytest
import tempfile
import os
from src.preprocessing.container_vocabulary import ContainerVocabulary


@pytest.mark.unit
def test_vocabulary_add_and_retrieve():
    """Test basic vocabulary operations."""
    vocab = ContainerVocabulary()

    # Add containers
    id1 = vocab.add_container("webapp")
    id2 = vocab.add_container("database")
    id3 = vocab.add_container("redis")

    # Check IDs are sequential starting from 0
    assert id1 == 0
    assert id2 == 1
    assert id3 == 2

    # Retrieve by ID
    assert vocab.get_name(0) == "webapp"
    assert vocab.get_name(1) == "database"
    assert vocab.get_name(2) == "redis"

    # Retrieve by name
    assert vocab.get_id("webapp") == 0
    assert vocab.get_id("database") == 1
    assert vocab.get_id("redis") == 2


@pytest.mark.unit
def test_vocabulary_idempotency():
    """Test that adding the same container twice returns the same ID."""
    vocab = ContainerVocabulary()

    # Add container twice
    id1 = vocab.add_container("webapp")
    id2 = vocab.add_container("webapp")

    assert id1 == id2
    assert id1 == 0

    # Add another container
    id3 = vocab.add_container("database")
    assert id3 == 1

    # Add first container again
    id4 = vocab.add_container("webapp")
    assert id4 == 0


@pytest.mark.unit
def test_vocabulary_contains():
    """Test the contains() method."""
    vocab = ContainerVocabulary()
    vocab.add_container("webapp")
    vocab.add_container("database")

    assert vocab.contains("webapp") is True
    assert vocab.contains("database") is True
    assert vocab.contains("redis") is False
    assert vocab.contains("unknown") is False


@pytest.mark.unit
def test_vocabulary_get_all():
    """Test getting all containers and IDs."""
    vocab = ContainerVocabulary()
    containers = ["webapp", "database", "redis", "cache"]

    for container in containers:
        vocab.add_container(container)

    # Get all names
    all_names = vocab.get_all_names()
    assert all_names == containers

    # Get all IDs
    all_ids = vocab.get_all_ids()
    assert all_ids == [0, 1, 2, 3]

    # Check count
    assert vocab.num_containers == 4
    assert len(vocab) == 4


@pytest.mark.unit
def test_vocabulary_error_handling():
    """Test error handling for invalid operations."""
    vocab = ContainerVocabulary()
    vocab.add_container("webapp")

    # Test getting nonexistent container by name
    with pytest.raises(KeyError, match="not in vocabulary"):
        vocab.get_id("nonexistent")

    # Test getting nonexistent container by ID
    with pytest.raises(KeyError, match="not in vocabulary"):
        vocab.get_name(999)


@pytest.mark.unit
def test_vocabulary_persistence(tmp_path):
    """Test save/load functionality."""
    vocab = ContainerVocabulary()
    vocab.add_container("webapp")
    vocab.add_container("database")
    vocab.add_container("redis")

    # Save
    save_path = tmp_path / "vocab.json"
    vocab.save(str(save_path))

    # Verify file exists
    assert save_path.exists()

    # Load
    loaded_vocab = ContainerVocabulary.load(str(save_path))

    # Verify loaded vocabulary matches original
    assert loaded_vocab.num_containers == vocab.num_containers
    assert loaded_vocab.get_all_names() == vocab.get_all_names()
    assert loaded_vocab.get_id("webapp") == 0
    assert loaded_vocab.get_id("database") == 1
    assert loaded_vocab.get_id("redis") == 2


@pytest.mark.unit
def test_vocabulary_to_from_dict():
    """Test dictionary serialization."""
    vocab = ContainerVocabulary()
    vocab.add_container("webapp")
    vocab.add_container("database")

    # Convert to dict
    vocab_dict = vocab.to_dict()

    assert "name_to_id" in vocab_dict
    assert "id_to_name" in vocab_dict
    assert "next_id" in vocab_dict

    assert vocab_dict["name_to_id"] == {"webapp": 0, "database": 1}
    assert vocab_dict["next_id"] == 2

    # Create from dict
    new_vocab = ContainerVocabulary.from_dict(vocab_dict)

    assert new_vocab.num_containers == 2
    assert new_vocab.get_id("webapp") == 0
    assert new_vocab.get_id("database") == 1


@pytest.mark.unit
def test_vocabulary_empty():
    """Test empty vocabulary behavior."""
    vocab = ContainerVocabulary()

    assert vocab.num_containers == 0
    assert len(vocab) == 0
    assert vocab.get_all_names() == []
    assert vocab.get_all_ids() == []
    assert vocab.contains("anything") is False


@pytest.mark.unit
def test_vocabulary_repr():
    """Test string representation."""
    vocab = ContainerVocabulary()
    vocab.add_container("webapp")
    vocab.add_container("database")

    repr_str = repr(vocab)

    assert "ContainerVocabulary" in repr_str
    assert "num_containers=2" in repr_str
    assert "webapp" in repr_str
    assert "database" in repr_str
