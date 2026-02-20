import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def set_seed() -> None:
    """Set random seed for reproducibility across all tests."""
    torch.manual_seed(1)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Provide torch device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
