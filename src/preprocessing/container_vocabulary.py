"""
Container Vocabulary Management
Maps container names to numeric IDs for embedding layer.
"""

import json
from typing import Dict, Optional, List
from pathlib import Path


class ContainerVocabulary:
    """
    Manages mapping between container names and numeric IDs.

    Used to create consistent container ID mappings for multi-container training
    with embedding layers. Vocabulary is saved with trained models to ensure
    consistent ID assignment during inference.

    Example:
        >>> vocab = ContainerVocabulary()
        >>> vocab.add_container('webapp')  # Returns 0
        >>> vocab.add_container('database')  # Returns 1
        >>> vocab.get_id('webapp')  # Returns 0
        >>> vocab.get_name(1)  # Returns 'database'
    """

    def __init__(self):
        """Initialize empty vocabulary."""
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        self._next_id: int = 0

    def add_container(self, name: str) -> int:
        """
        Add container to vocabulary or retrieve existing ID.

        Args:
            name: Container name (e.g., 'webapp', 'database')

        Returns:
            Numeric ID for the container (0-indexed)

        Example:
            >>> vocab = ContainerVocabulary()
            >>> vocab.add_container('webapp')  # Returns 0
            >>> vocab.add_container('webapp')  # Returns 0 (idempotent)
        """
        if name in self._name_to_id:
            return self._name_to_id[name]

        container_id = self._next_id
        self._name_to_id[name] = container_id
        self._id_to_name[container_id] = name
        self._next_id += 1

        return container_id

    def get_id(self, name: str) -> int:
        """
        Get numeric ID for container name.

        Args:
            name: Container name

        Returns:
            Numeric ID

        Raises:
            KeyError: If container name not in vocabulary

        Example:
            >>> vocab.get_id('webapp')  # Returns 0
        """
        if name not in self._name_to_id:
            raise KeyError(
                f"Container '{name}' not in vocabulary. "
                f"Available containers: {list(self._name_to_id.keys())}"
            )
        return self._name_to_id[name]

    def get_name(self, container_id: int) -> str:
        """
        Get container name for numeric ID.

        Args:
            container_id: Numeric ID

        Returns:
            Container name

        Raises:
            KeyError: If ID not in vocabulary

        Example:
            >>> vocab.get_name(0)  # Returns 'webapp'
        """
        if container_id not in self._id_to_name:
            raise KeyError(
                f"Container ID {container_id} not in vocabulary. "
                f"Valid IDs: {list(self._id_to_name.keys())}"
            )
        return self._id_to_name[container_id]

    def contains(self, name: str) -> bool:
        """
        Check if container name exists in vocabulary.

        Args:
            name: Container name

        Returns:
            True if container exists, False otherwise
        """
        return name in self._name_to_id

    def get_all_names(self) -> List[str]:
        """
        Get all container names in vocabulary.

        Returns:
            List of container names sorted by ID
        """
        return [self._id_to_name[i] for i in sorted(self._id_to_name.keys())]

    def get_all_ids(self) -> List[int]:
        """
        Get all container IDs in vocabulary.

        Returns:
            List of IDs
        """
        return sorted(self._id_to_name.keys())

    @property
    def num_containers(self) -> int:
        """
        Get number of containers in vocabulary.

        Returns:
            Number of unique containers
        """
        return len(self._name_to_id)

    def to_dict(self) -> Dict:
        """
        Convert vocabulary to dictionary for serialization.

        Returns:
            Dictionary with vocab mapping and metadata

        Example:
            >>> vocab.to_dict()
            {'name_to_id': {'webapp': 0, 'database': 1}, 'next_id': 2}
        """
        return {
            "name_to_id": self._name_to_id,
            "id_to_name": {str(k): v for k, v in self._id_to_name.items()},
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ContainerVocabulary":
        """
        Create vocabulary from dictionary.

        Args:
            data: Dictionary with vocab mapping

        Returns:
            ContainerVocabulary instance

        Example:
            >>> data = {'name_to_id': {'webapp': 0}, 'next_id': 1}
            >>> vocab = ContainerVocabulary.from_dict(data)
        """
        vocab = cls()
        vocab._name_to_id = data["name_to_id"]
        vocab._id_to_name = {int(k): v for k, v in data["id_to_name"].items()}
        vocab._next_id = data["next_id"]
        return vocab

    def save(self, path: str) -> None:
        """
        Save vocabulary to JSON file.

        Args:
            path: Path to save JSON file

        Example:
            >>> vocab.save('vocab.json')
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ContainerVocabulary":
        """
        Load vocabulary from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            ContainerVocabulary instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is invalid JSON

        Example:
            >>> vocab = ContainerVocabulary.load('vocab.json')
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation of vocabulary."""
        return (
            f"ContainerVocabulary(num_containers={self.num_containers}, "
            f"containers={self.get_all_names()})"
        )

    def __len__(self) -> int:
        """Return number of containers in vocabulary."""
        return self.num_containers


if __name__ == "__main__":
    # Example usage and testing
    print("Container Vocabulary")
    print("=" * 60)

    # Create vocabulary
    vocab = ContainerVocabulary()

    # Add containers
    print("\nAdding containers:")
    containers = ["webapp", "database", "redis", "cache", "worker"]
    for container in containers:
        container_id = vocab.add_container(container)
        print(f"  {container}: ID={container_id}")

    # Test idempotency
    print("\nTesting idempotency:")
    webapp_id = vocab.add_container("webapp")
    print(f"  Adding 'webapp' again: ID={webapp_id} (should be 0)")

    # Lookup operations
    print("\nLookup operations:")
    print(f"  get_id('database'): {vocab.get_id('database')}")
    print(f"  get_name(2): {vocab.get_name(2)}")
    print(f"  contains('webapp'): {vocab.contains('webapp')}")
    print(f"  contains('unknown'): {vocab.contains('unknown')}")

    # Get all containers
    print(f"\nAll containers: {vocab.get_all_names()}")
    print(f"Total containers: {vocab.num_containers}")

    # Save and load
    print("\nTesting save/load:")
    test_path = "/tmp/test_vocab.json"
    vocab.save(test_path)
    print(f"  Saved to {test_path}")

    loaded_vocab = ContainerVocabulary.load(test_path)
    print(f"  Loaded vocabulary: {loaded_vocab}")
    print(f"  Loaded containers: {loaded_vocab.get_all_names()}")

    # Test error handling
    print("\nTesting error handling:")
    try:
        vocab.get_id("nonexistent")
    except KeyError as e:
        print(f"  Expected error: {e}")

    print("\nContainer vocabulary created successfully!")
