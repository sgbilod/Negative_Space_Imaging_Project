"""
Security Web Interface
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Web interface for the Sovereign Security System, providing:
- User management
- Security monitoring
- Access control
- Configuration management
"""

from flask import (
    Blueprint, render_template, request, jsonify,
    redirect, url_for, session, flash
)
import json
import logging
import os
from pathlib import Path
from functools import wraps
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sovereign.security_web')

# Create Blueprint
security_bp = Blueprint(
    'security',
    __name__,
    template_folder='templates/security',
    static_folder='static'
)

# Get reference to the SecurityManager
# This will be initialized by the main app
security_manager = None

def init_security_web(app, sm):
    """Initialize the security web interface with the app and security manager"""
    global security_manager
    security_manager = sm
    app.register_blueprint(security_bp, url_prefix='/security')
    logger.info("Security web interface initialized")


def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'token' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('security.login'))

        # Validate token
        token = session['token']
        valid, username = security_manager.validate_session_token(token)

        if not valid:
            # Clear invalid session
            session.clear()
            flash('Your session has expired, please log in again', 'warning')
            return redirect(url_for('security.login'))

        # Store username in request context for access in the view
        request.username = username

        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin role for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'token' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('security.login'))

        # Validate token
        token = session['token']
        valid, username = security_manager.validate_session_token(token)

        if not valid:
            # Clear invalid session
            session.clear()
            flash('Your session has expired, please log in again', 'warning')
            return redirect(url_for('security.login'))

        # Check if user is an admin
        if not security_manager.check_authorization(username, 'administrator'):
            flash('You do not have permission to access this page', 'danger')
            return redirect(url_for('security.dashboard'))

        # Store username in request context for access in the view
        request.username = username

        return f(*args, **kwargs)
    return decorated_function


@security_bp.route('/')
def index():
    """Main security page - redirects to login or dashboard"""
    if 'token' in session:
        # Validate token
        token = session['token']
        valid, _ = security_manager.validate_session_token(token)

        if valid:
            return redirect(url_for('security.dashboard'))

    return redirect(url_for('security.login'))


@security_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    error = None

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Authenticate user
        success, user_data = security_manager.authenticate_user(username, password)

        if success:
            # Generate session token
            token = security_manager.generate_session_token(
                username,
                security_manager.settings['access_control']['session_timeout_minutes']
            )

            # Store token in session
            session['token'] = token
            session['username'] = username

            # Check if password change is required
            if user_data.get('require_password_change'):
                flash('You need to change your password', 'warning')
                return redirect(url_for('security.change_password'))

            return redirect(url_for('security.dashboard'))
        else:
            error = "Invalid username or password"
            security_manager.log_security_event(
                "WEB_LOGIN_FAILED",
                {"username": username, "source_ip": request.remote_addr}
            )

    return render_template('security/login.html', error=error)


@security_bp.route('/logout')
def logout():
    """Logout route"""
    if 'token' in session:
        security_manager.log_security_event(
            "WEB_LOGOUT",
            {"username": session.get('username', 'unknown'), "source_ip": request.remote_addr}
        )

    # Clear session
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('security.login'))


@security_bp.route('/dashboard')
@login_required
def dashboard():
    """Security dashboard"""
    return render_template('security/dashboard.html', username=request.username)


@security_bp.route('/users')
@admin_required
def users():
    """User management page"""
    return render_template('security/users.html', username=request.username)


@security_bp.route('/api/users/list')
@admin_required
def api_users_list():
    """API endpoint to list all users"""
    users_file = security_manager.users_path / "users.json"

    if not users_file.exists():
        return jsonify({"error": "Users file not found"}), 404

    try:
        # Load users data
        users = security_manager._load_encrypted_json(users_file, security_manager.auth_key)

        # Remove sensitive information
        for user in users.values():
            if 'password_hash' in user:
                del user['password_hash']
            if 'salt' in user:
                del user['salt']

        return jsonify({"users": users})

    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        return jsonify({"error": str(e)}), 500


@security_bp.route('/api/users/create', methods=['POST'])
@admin_required
def api_users_create():
    """API endpoint to create a new user"""
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = data.get('username')
    password = data.get('password')
    role = data.get('role')

    if not all([username, password, role]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        success = security_manager.create_user(username, password, role)

        if success:
            security_manager.log_security_event(
                "WEB_USER_CREATED",
                {
                    "admin": request.username,
                    "new_user": username,
                    "role": role
                }
            )
            return jsonify({"success": True, "message": f"User {username} created successfully"})
        else:
            return jsonify({"error": "Failed to create user"}), 500

    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        return jsonify({"error": str(e)}), 500


@security_bp.route('/api/users/delete', methods=['POST'])
@admin_required
def api_users_delete():
    """API endpoint to delete a user"""
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = data.get('username')

    if not username:
        return jsonify({"error": "Missing username"}), 400

    # Can't delete yourself
    if username == request.username:
        return jsonify({"error": "You cannot delete your own account"}), 403

    users_file = security_manager.users_path / "users.json"

    if not users_file.exists():
        return jsonify({"error": "Users file not found"}), 404

    try:
        # Load users data
        users = security_manager._load_encrypted_json(users_file, security_manager.auth_key)

        # Check if user exists
        if username not in users:
            return jsonify({"error": f"User {username} not found"}), 404

        # Delete user
        del users[username]

        # Save updated users
        security_manager._save_encrypted_json(users_file, users, security_manager.auth_key)

        security_manager.log_security_event(
            "WEB_USER_DELETED",
            {"admin": request.username, "deleted_user": username}
        )

        return jsonify({"success": True, "message": f"User {username} deleted successfully"})

    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        return jsonify({"error": str(e)}), 500


@security_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change password page"""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not all([current_password, new_password, confirm_password]):
            flash('All fields are required', 'danger')
            return render_template('security/change_password.html')

        if new_password != confirm_password:
            flash('New passwords do not match', 'danger')
            return render_template('security/change_password.html')

        # Verify current password
        username = request.username
        success, _ = security_manager.authenticate_user(username, current_password)

        if not success:
            flash('Current password is incorrect', 'danger')
            return render_template('security/change_password.html')

        # Update password
        users_file = security_manager.users_path / "users.json"

        try:
            # Load users data
            users = security_manager._load_encrypted_json(users_file, security_manager.auth_key)

            # Generate new salt and hash
            salt = secrets.token_bytes(16)
            password_hash = security_manager._hash_password(new_password, salt)

            # Update user data
            users[username]['password_hash'] = base64.b64encode(password_hash).decode('utf-8')
            users[username]['salt'] = base64.b64encode(salt).decode('utf-8')
            users[username]['require_password_change'] = False

            # Save updated users
            security_manager._save_encrypted_json(users_file, users, security_manager.auth_key)

            security_manager.log_security_event(
                "WEB_PASSWORD_CHANGED",
                {"username": username}
            )

            flash('Password changed successfully', 'success')
            return redirect(url_for('security.dashboard'))

        except Exception as e:
            logger.error(f"Error changing password: {str(e)}")
            flash(f'Error changing password: {str(e)}', 'danger')

    return render_template('security/change_password.html')


@security_bp.route('/audit-logs')
@admin_required
def audit_logs():
    """Audit logs page"""
    return render_template('security/audit_logs.html', username=request.username)


@security_bp.route('/api/audit-logs')
@admin_required
def api_audit_logs():
    """API endpoint to get audit logs"""
    limit = request.args.get('limit', 100, type=int)
    event_type = request.args.get('event_type')
    username = request.args.get('username')

    log_entries = []
    log_files = sorted(security_manager.logs_path.glob('audit_*.log'), reverse=True)

    for log_file in log_files:
        if len(log_entries) >= limit:
            break

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if len(log_entries) >= limit:
                        break

                    try:
                        entry = json.loads(line.strip())

                        # Filter by event type if specified
                        if event_type and entry.get('event_type') != event_type:
                            continue

                        # Filter by username if specified
                        if username:
                            user_match = False
                            if 'details' in entry:
                                if entry['details'].get('username') == username:
                                    user_match = True
                                elif entry['details'].get('admin') == username:
                                    user_match = True
                                elif entry['details'].get('new_user') == username:
                                    user_match = True
                                elif entry['details'].get('deleted_user') == username:
                                    user_match = True

                            if not user_match:
                                continue

                        log_entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {str(e)}")

    return jsonify({"logs": log_entries})


@security_bp.route('/settings')
@admin_required
def settings():
    """Security settings page"""
    return render_template(
        'security/settings.html',
        settings=security_manager.settings,
        security_level=security_manager.security_level,
        username=request.username
    )


@security_bp.route('/api/settings', methods=['POST'])
@admin_required
def api_settings():
    """API endpoint to update security settings"""
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Update settings
        for category in data:
            if category in security_manager.settings:
                for setting in data[category]:
                    if setting in security_manager.settings[category]:
                        security_manager.settings[category][setting] = data[category][setting]

        # Save settings to file
        settings_file = security_manager.security_path / "settings.json"
        security_manager._save_encrypted_json(settings_file, security_manager.settings, security_manager.config_key)

        security_manager.log_security_event(
            "WEB_SETTINGS_UPDATED",
            {"admin": request.username}
        )

        return jsonify({"success": True, "message": "Settings updated successfully"})

    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@security_bp.route('/api/security-level', methods=['POST'])
@admin_required
def api_security_level():
    """API endpoint to update security level"""
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    level = data.get('level')

    if not level:
        return jsonify({"error": "Missing security level"}), 400

    try:
        # Validate level
        valid_levels = [
            SecurityLevel.STANDARD,
            SecurityLevel.ENHANCED,
            SecurityLevel.MAXIMUM,
            SecurityLevel.QUANTUM
        ]

        if level not in valid_levels:
            return jsonify({"error": f"Invalid security level: {level}"}), 400

        # Update security level
        security_manager.security_level = level

        # Update settings based on new level
        security_manager.settings = security_manager._load_default_settings()

        # Save settings to file
        settings_file = security_manager.security_path / "settings.json"
        security_manager._save_encrypted_json(settings_file, security_manager.settings, security_manager.config_key)

        security_manager.log_security_event(
            "WEB_SECURITY_LEVEL_CHANGED",
            {"admin": request.username, "level": level}
        )

        return jsonify({
            "success": True,
            "message": f"Security level updated to {level}",
            "settings": security_manager.settings
        })

    except Exception as e:
        logger.error(f"Error updating security level: {str(e)}")
        return jsonify({"error": str(e)}), 500


@security_bp.route('/api/rotate-keys', methods=['POST'])
@admin_required
def api_rotate_keys():
    """API endpoint to rotate encryption keys"""
    try:
        success = security_manager.rotate_keys()

        if success:
            security_manager.log_security_event(
                "WEB_KEYS_ROTATED",
                {"admin": request.username}
            )
            return jsonify({"success": True, "message": "Encryption keys rotated successfully"})
        else:
            return jsonify({"error": "Failed to rotate keys"}), 500

    except Exception as e:
        logger.error(f"Error rotating keys: {str(e)}")
        return jsonify({"error": str(e)}), 500


@security_bp.route('/file-security')
@admin_required
def file_security():
    """File security page"""
    return render_template('security/file_security.html', username=request.username)


@security_bp.route('/api/update-file-checksums', methods=['POST'])
@admin_required
def api_update_file_checksums():
    """API endpoint to update file checksums"""
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    paths = data.get('paths', [])

    if not paths:
        return jsonify({"error": "No paths provided"}), 400

    results = {
        "success": [],
        "failed": []
    }

    for path_str in paths:
        try:
            path = Path(path_str)

            # Ensure path is within project
            if not str(path).startswith(str(security_manager.project_root)):
                results["failed"].append({
                    "path": path_str,
                    "error": "Path is outside project directory"
                })
                continue

            # Check if path exists
            if not path.exists():
                results["failed"].append({
                    "path": path_str,
                    "error": "Path does not exist"
                })
                continue

            # Update checksum
            if path.is_file():
                success = security_manager.update_file_checksum(path)

                if success:
                    results["success"].append(path_str)
                else:
                    results["failed"].append({
                        "path": path_str,
                        "error": "Failed to update checksum"
                    })
            elif path.is_dir():
                # Update checksums for all files in directory
                for file_path in path.glob('**/*'):
                    if file_path.is_file():
                        success = security_manager.update_file_checksum(file_path)

                        if success:
                            results["success"].append(str(file_path))
                        else:
                            results["failed"].append({
                                "path": str(file_path),
                                "error": "Failed to update checksum"
                            })

        except Exception as e:
            logger.error(f"Error updating checksum for {path_str}: {str(e)}")
            results["failed"].append({
                "path": path_str,
                "error": str(e)
            })

    security_manager.log_security_event(
        "WEB_FILE_CHECKSUMS_UPDATED",
        {
            "admin": request.username,
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"])
        }
    )

    return jsonify(results)


@security_bp.route('/api/verify-file-integrity', methods=['POST'])
@admin_required
def api_verify_file_integrity():
    """API endpoint to verify file integrity"""
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    paths = data.get('paths', [])

    if not paths:
        return jsonify({"error": "No paths provided"}), 400

    results = {
        "verified": [],
        "failed": [],
        "not_found": []
    }

    for path_str in paths:
        try:
            path = Path(path_str)

            # Ensure path is within project
            if not str(path).startswith(str(security_manager.project_root)):
                results["failed"].append({
                    "path": path_str,
                    "error": "Path is outside project directory"
                })
                continue

            # Check if path exists
            if not path.exists():
                results["not_found"].append(path_str)
                continue

            # Verify integrity
            if path.is_file():
                integrity = security_manager.verify_file_integrity(path)

                if integrity:
                    results["verified"].append(path_str)
                else:
                    results["failed"].append({
                        "path": path_str,
                        "error": "Integrity check failed"
                    })
            elif path.is_dir():
                # Verify integrity for all files in directory
                for file_path in path.glob('**/*'):
                    if file_path.is_file():
                        integrity = security_manager.verify_file_integrity(file_path)

                        if integrity:
                            results["verified"].append(str(file_path))
                        else:
                            results["failed"].append({
                                "path": str(file_path),
                                "error": "Integrity check failed"
                            })

        except Exception as e:
            logger.error(f"Error verifying integrity for {path_str}: {str(e)}")
            results["failed"].append({
                "path": path_str,
                "error": str(e)
            })

    security_manager.log_security_event(
        "WEB_FILE_INTEGRITY_CHECK",
        {
            "admin": request.username,
            "verified_count": len(results["verified"]),
            "failed_count": len(results["failed"]),
            "not_found_count": len(results["not_found"])
        }
    )

    return jsonify(results)


@security_bp.route('/api/encrypt-file', methods=['POST'])
@admin_required
def api_encrypt_file():
    """API endpoint to encrypt a file"""
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    paths = data.get('paths', [])

    if not paths:
        return jsonify({"error": "No paths provided"}), 400

    results = {
        "success": [],
        "failed": []
    }

    for path_str in paths:
        try:
            path = Path(path_str)

            # Ensure path is within project
            if not str(path).startswith(str(security_manager.project_root)):
                results["failed"].append({
                    "path": path_str,
                    "error": "Path is outside project directory"
                })
                continue

            # Check if path exists
            if not path.exists():
                results["failed"].append({
                    "path": path_str,
                    "error": "Path does not exist"
                })
                continue

            # Encrypt file
            if path.is_file():
                success = security_manager.encrypt_file(path)

                if success:
                    results["success"].append(path_str)
                else:
                    results["failed"].append({
                        "path": path_str,
                        "error": "Failed to encrypt file"
                    })
            elif path.is_dir():
                # Encrypt all files in directory
                for file_path in path.glob('**/*'):
                    if file_path.is_file():
                        success = security_manager.encrypt_file(file_path)

                        if success:
                            results["success"].append(str(file_path))
                        else:
                            results["failed"].append({
                                "path": str(file_path),
                                "error": "Failed to encrypt file"
                            })

        except Exception as e:
            logger.error(f"Error encrypting file {path_str}: {str(e)}")
            results["failed"].append({
                "path": path_str,
                "error": str(e)
            })

    security_manager.log_security_event(
        "WEB_FILES_ENCRYPTED",
        {
            "admin": request.username,
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"])
        }
    )

    return jsonify(results)


@security_bp.route('/api/decrypt-file', methods=['POST'])
@admin_required
def api_decrypt_file():
    """API endpoint to decrypt a file"""
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    paths = data.get('paths', [])

    if not paths:
        return jsonify({"error": "No paths provided"}), 400

    results = {
        "success": [],
        "failed": []
    }

    for path_str in paths:
        try:
            path = Path(path_str)

            # Ensure path is within project
            if not str(path).startswith(str(security_manager.project_root)):
                results["failed"].append({
                    "path": path_str,
                    "error": "Path is outside project directory"
                })
                continue

            # Check if path exists
            if not path.exists():
                results["failed"].append({
                    "path": path_str,
                    "error": "Path does not exist"
                })
                continue

            # Decrypt file
            if path.is_file():
                success = security_manager.decrypt_file(path)

                if success:
                    results["success"].append(path_str)
                else:
                    results["failed"].append({
                        "path": path_str,
                        "error": "Failed to decrypt file"
                    })
            elif path.is_dir():
                # Decrypt all files in directory
                for file_path in path.glob('**/*.enc'):
                    if file_path.is_file():
                        success = security_manager.decrypt_file(file_path)

                        if success:
                            results["success"].append(str(file_path))
                        else:
                            results["failed"].append({
                                "path": str(file_path),
                                "error": "Failed to decrypt file"
                            })

        except Exception as e:
            logger.error(f"Error decrypting file {path_str}: {str(e)}")
            results["failed"].append({
                "path": path_str,
                "error": str(e)
            })

    security_manager.log_security_event(
        "WEB_FILES_DECRYPTED",
        {
            "admin": request.username,
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"])
        }
    )

    return jsonify(results)
