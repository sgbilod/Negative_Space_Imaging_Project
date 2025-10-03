"""Control mode definitions for the Sovereign system."""

from enum import Enum, auto

class ControlMode(Enum):
    """Operating modes for the Master Controller."""

    STANDARD = auto()
    DEBUG = auto()
    SAFE = auto()
    SOVEREIGN = auto()  # Full autonomous control mode
    EMERGENCY = auto()

    def __str__(self) -> str:
        """Return string representation."""
        return self.name

    @classmethod
    def from_string(cls, mode_str: str) -> 'ControlMode':
        """Create mode from string representation.

        Args:
            mode_str: String name of the mode

        Returns:
            ControlMode instance

        Raises:
            ValueError: If mode string is invalid
        """
        try:
            return cls[mode_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid control mode: {mode_str}")

    def is_autonomous(self) -> bool:
        """Check if mode is autonomous.

        Returns:
            bool: True if mode is autonomous
        """
        return self in (self.SOVEREIGN,)
