"""
Tests for image_acquisition.py - Camera and SFTP Acquisition Methods

This module provides comprehensive tests for the camera and SFTP acquisition
methods implemented in the ImageAcquisition class.
"""

import pytest
import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from image_acquisition import (
    ImageAcquisition,
    ImageFormat,
    AcquisitionMode,
    AcquisitionError,
    SourceAuthenticationError,
)


class TestCameraAcquisition:
    """Tests for _acquire_from_camera method."""

    def test_camera_mode_initialization(self):
        """Test that CAMERA mode can be initialized when cv2 or PIL is available."""
        # Should not raise ImportError when cv2 is available
        acq = ImageAcquisition(mode=AcquisitionMode.CAMERA)
        assert acq.mode == AcquisitionMode.CAMERA

    def test_camera_device_id_parsing_int(self):
        """Test that integer device_id is parsed correctly."""
        acq = ImageAcquisition(mode=AcquisitionMode.CAMERA)

        # Mock cv2.VideoCapture to test device_id parsing
        with patch('image_acquisition.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FRAME_WIDTH = 3
            mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

            with pytest.raises(AcquisitionError, match="Failed to open camera"):
                acq._acquire_from_camera("0")

            mock_cv2.VideoCapture.assert_called_with(0)

    def test_camera_device_id_parsing_string(self):
        """Test that non-integer device_id defaults to camera 0."""
        acq = ImageAcquisition(mode=AcquisitionMode.CAMERA)

        with patch('image_acquisition.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FRAME_WIDTH = 3
            mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

            with pytest.raises(AcquisitionError, match="Failed to open camera"):
                acq._acquire_from_camera("my_camera")

            # Should default to camera index 0
            mock_cv2.VideoCapture.assert_called_with(0)

    def test_camera_capture_success(self):
        """Test successful camera capture."""
        acq = ImageAcquisition(mode=AcquisitionMode.CAMERA)

        with patch('image_acquisition.cv2') as mock_cv2:
            import numpy as np

            # Setup mock camera
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FRAME_WIDTH = 3
            mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
            mock_cv2.CAP_PROP_EXPOSURE = 15
            mock_cv2.CAP_PROP_GAIN = 14
            mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            mock_cv2.COLOR_BGR2GRAY = 6

            result = acq._acquire_from_camera("0", format='rgb')

            assert isinstance(result, bytes)
            assert len(result) > 0
            mock_cap.release.assert_called_once()

    def test_camera_capture_failure(self):
        """Test camera capture failure handling."""
        acq = ImageAcquisition(mode=AcquisitionMode.CAMERA)

        with patch('image_acquisition.cv2') as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FRAME_WIDTH = 3
            mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

            with pytest.raises(AcquisitionError, match="Failed to capture frame"):
                acq._acquire_from_camera("0", timeout=0.1)

            mock_cap.release.assert_called_once()

    def test_camera_configuration_options(self):
        """Test that camera configuration options are applied."""
        acq = ImageAcquisition(mode=AcquisitionMode.CAMERA)

        with patch('image_acquisition.cv2') as mock_cv2:
            import numpy as np

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FRAME_WIDTH = 3
            mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
            mock_cv2.CAP_PROP_EXPOSURE = 15
            mock_cv2.CAP_PROP_GAIN = 14
            mock_cv2.cvtColor.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4

            acq._acquire_from_camera(
                "0",
                resolution=(1280, 720),
                exposure=10.0,
                gain=2.5,
                format='rgb'
            )

            # Verify resolution was set
            mock_cap.set.assert_any_call(3, 1280)  # WIDTH
            mock_cap.set.assert_any_call(4, 720)   # HEIGHT
            mock_cap.set.assert_any_call(15, 10.0) # EXPOSURE
            mock_cap.set.assert_any_call(14, 2.5)  # GAIN

    def test_camera_grayscale_output(self):
        """Test grayscale output format."""
        acq = ImageAcquisition(mode=AcquisitionMode.CAMERA)

        with patch('image_acquisition.cv2') as mock_cv2:
            import numpy as np

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FRAME_WIDTH = 3
            mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
            mock_cv2.cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)
            mock_cv2.COLOR_BGR2GRAY = 6

            result = acq._acquire_from_camera("0", format='grayscale')

            mock_cv2.cvtColor.assert_called()
            assert isinstance(result, bytes)


class TestSFTPAcquisition:
    """Tests for _acquire_from_sftp method."""

    def test_sftp_mode_initialization(self):
        """Test that SFTP mode can be initialized when paramiko is available."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP)
        assert acq.mode == AcquisitionMode.REMOTE_SFTP

    def test_sftp_parse_source_string(self):
        """Test parsing of user@host:port/path format."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            mock_ssh = MagicMock()
            mock_sftp = MagicMock()
            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_file.read.return_value = b'test_image_data'

            mock_sftp.file.return_value = mock_file
            mock_ssh.open_sftp.return_value = mock_sftp
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()

            result = acq._acquire_from_sftp(
                "testuser@example.com:2222/images/test.png",
                secure=False,
                password="testpass"
            )

            # Verify connection params
            mock_ssh.connect.assert_called_once()
            call_kwargs = mock_ssh.connect.call_args[1]
            assert call_kwargs['hostname'] == 'example.com'
            assert call_kwargs['port'] == 2222
            assert call_kwargs['username'] == 'testuser'

    def test_sftp_with_kwargs_connection_details(self):
        """Test SFTP with connection details in kwargs."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            mock_ssh = MagicMock()
            mock_sftp = MagicMock()
            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_file.read.return_value = b'test_image_data'

            mock_sftp.file.return_value = mock_file
            mock_ssh.open_sftp.return_value = mock_sftp
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()

            result = acq._acquire_from_sftp(
                "/images/test.png",
                secure=False,
                hostname="sftp.example.com",
                port=22,
                username="user",
                password="pass"
            )

            call_kwargs = mock_ssh.connect.call_args[1]
            assert call_kwargs['hostname'] == 'sftp.example.com'
            assert call_kwargs['username'] == 'user'

    def test_sftp_missing_hostname_error(self):
        """Test error when hostname is not specified."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP)

        with pytest.raises(AcquisitionError, match="hostname not specified"):
            acq._acquire_from_sftp("/images/test.png", username="user", password="pass")

    def test_sftp_missing_username_error(self):
        """Test error when username is not specified."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP)

        with pytest.raises(AcquisitionError, match="username not specified"):
            acq._acquire_from_sftp("/images/test.png", hostname="example.com", password="pass")

    def test_sftp_missing_remote_path_error(self):
        """Test error when remote file path is not specified."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP)

        with pytest.raises(AcquisitionError, match="remote file path not specified"):
            acq._acquire_from_sftp("user@example.com", password="pass")

    def test_sftp_username_with_at_symbol(self):
        """Test parsing source string with @ in username."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            mock_ssh = MagicMock()
            mock_sftp = MagicMock()
            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_file.read.return_value = b'test_image_data'

            mock_sftp.file.return_value = mock_file
            mock_ssh.open_sftp.return_value = mock_sftp
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()

            result = acq._acquire_from_sftp(
                "user@domain.com@example.com:2222/images/test.png",
                secure=False,
                password="testpass"
            )

            # Verify connection params - username should be "user@domain.com"
            mock_ssh.connect.assert_called_once()
            call_kwargs = mock_ssh.connect.call_args[1]
            assert call_kwargs['hostname'] == 'example.com'
            assert call_kwargs['port'] == 2222
            assert call_kwargs['username'] == 'user@domain.com'

    def test_sftp_missing_auth_error(self):
        """Test error when no authentication method is provided."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            mock_ssh = MagicMock()
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()
            # Create proper exception classes that inherit from BaseException
            mock_paramiko.AuthenticationException = type('AuthenticationException', (Exception,), {})
            mock_paramiko.SSHException = type('SSHException', (Exception,), {})

            with pytest.raises(SourceAuthenticationError, match="No authentication method"):
                acq._acquire_from_sftp(
                    "/images/test.png",
                    secure=False,
                    hostname="example.com",
                    username="user"
                )

    def test_sftp_authentication_failure(self):
        """Test handling of authentication failure."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            # Create proper exception classes first
            AuthExc = type('AuthenticationException', (Exception,), {})
            SSHExc = type('SSHException', (Exception,), {})
            mock_paramiko.AuthenticationException = AuthExc
            mock_paramiko.SSHException = SSHExc

            mock_ssh = MagicMock()
            mock_ssh.connect.side_effect = AuthExc("Bad password")
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()

            with pytest.raises(SourceAuthenticationError, match="authentication failed"):
                acq._acquire_from_sftp(
                    "/images/test.png",
                    secure=False,
                    hostname="example.com",
                    username="user",
                    password="wrongpass"
                )

    def test_sftp_connection_error(self):
        """Test handling of SSH connection error."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            # Create proper exception classes first
            AuthExc = type('AuthenticationException', (Exception,), {})
            SSHExc = type('SSHException', (Exception,), {})
            mock_paramiko.AuthenticationException = AuthExc
            mock_paramiko.SSHException = SSHExc

            mock_ssh = MagicMock()
            mock_ssh.connect.side_effect = SSHExc("Connection refused")
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()

            with pytest.raises(AcquisitionError, match="connection error"):
                acq._acquire_from_sftp(
                    "/images/test.png",
                    secure=False,
                    hostname="example.com",
                    username="user",
                    password="pass"
                )

    def test_sftp_file_error(self):
        """Test handling of file read error."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            mock_ssh = MagicMock()
            mock_sftp = MagicMock()
            mock_sftp.file.side_effect = IOError("File not found")
            mock_ssh.open_sftp.return_value = mock_sftp
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()
            mock_paramiko.AuthenticationException = type('AuthenticationException', (Exception,), {})
            mock_paramiko.SSHException = type('SSHException', (Exception,), {})

            with pytest.raises(AcquisitionError, match="file error"):
                acq._acquire_from_sftp(
                    "/images/test.png",
                    secure=False,
                    hostname="example.com",
                    username="user",
                    password="pass"
                )

    def test_sftp_key_based_auth(self):
        """Test key-based authentication."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            mock_ssh = MagicMock()
            mock_sftp = MagicMock()
            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_file.read.return_value = b'test_image_data'

            mock_pkey = MagicMock()
            mock_paramiko.RSAKey.from_private_key_file.return_value = mock_pkey

            mock_sftp.file.return_value = mock_file
            mock_ssh.open_sftp.return_value = mock_sftp
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as f:
                f.write("fake key")
                key_path = f.name

            try:
                result = acq._acquire_from_sftp(
                    "/images/test.png",
                    secure=False,
                    hostname="example.com",
                    username="user",
                    private_key=key_path
                )

                mock_paramiko.RSAKey.from_private_key_file.assert_called_with(key_path)
                assert result == b'test_image_data'
            finally:
                os.unlink(key_path)

    def test_sftp_host_key_verification_high_security(self):
        """Test host key verification at high security level."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=2)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            with patch('image_acquisition.os.path.exists', return_value=False):
                # Create proper exception classes first
                AuthExc = type('AuthenticationException', (Exception,), {})
                SSHExc = type('SSHException', (Exception,), {})
                mock_paramiko.AuthenticationException = AuthExc
                mock_paramiko.SSHException = SSHExc

                mock_ssh = MagicMock()
                # Simulate host key rejection
                mock_ssh.connect.side_effect = SSHExc("Host key verification failed")
                mock_paramiko.SSHClient.return_value = mock_ssh
                mock_paramiko.RejectPolicy.return_value = MagicMock()

                with pytest.raises(AcquisitionError, match="connection error"):
                    acq._acquire_from_sftp(
                        "/images/test.png",
                        secure=True,
                        hostname="example.com",
                        username="user",
                        password="pass"
                    )

                mock_ssh.set_missing_host_key_policy.assert_called()

    def test_sftp_cleanup_on_error(self):
        """Test that SSH connection is cleaned up on error."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            mock_ssh = MagicMock()
            mock_sftp = MagicMock()
            mock_sftp.file.side_effect = IOError("File not found")
            mock_ssh.open_sftp.return_value = mock_sftp
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()
            mock_paramiko.AuthenticationException = type('AuthenticationException', (Exception,), {})
            mock_paramiko.SSHException = type('SSHException', (Exception,), {})

            with pytest.raises(AcquisitionError):
                acq._acquire_from_sftp(
                    "/images/test.png",
                    secure=False,
                    hostname="example.com",
                    username="user",
                    password="pass"
                )

            # Verify cleanup was called
            mock_ssh.close.assert_called_once()


class TestIntegrationAcquisition:
    """Integration tests for acquire() method with camera and SFTP modes."""

    def test_full_camera_acquisition_flow(self):
        """Test full acquisition flow with camera mode."""
        acq = ImageAcquisition(mode=AcquisitionMode.CAMERA)

        with patch('image_acquisition.cv2') as mock_cv2:
            import numpy as np

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.CAP_PROP_FRAME_WIDTH = 3
            mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
            mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4

            image_data, metadata = acq.acquire("0", secure=False, format='rgb')

            assert isinstance(image_data, bytes)
            assert isinstance(metadata, dict)
            assert 'acquisition_id' in metadata
            assert 'timestamp' in metadata
            assert 'sha256_hash' in metadata

    def test_full_sftp_acquisition_flow(self):
        """Test full acquisition flow with SFTP mode."""
        acq = ImageAcquisition(mode=AcquisitionMode.REMOTE_SFTP, security_level=0)

        with patch('image_acquisition.paramiko') as mock_paramiko:
            mock_ssh = MagicMock()
            mock_sftp = MagicMock()
            mock_file = MagicMock()
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=False)
            mock_file.read.return_value = b'test_image_data'

            mock_sftp.file.return_value = mock_file
            mock_ssh.open_sftp.return_value = mock_sftp
            mock_paramiko.SSHClient.return_value = mock_ssh
            mock_paramiko.AutoAddPolicy.return_value = MagicMock()

            image_data, metadata = acq.acquire(
                "testuser@example.com/images/test.png",
                secure=False,
                password="testpass"
            )

            assert image_data == b'test_image_data'
            assert isinstance(metadata, dict)
            assert 'acquisition_id' in metadata
            assert 'sha256_hash' in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
