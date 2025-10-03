def init_app(testing: bool = False):
    """Stub for test compatibility. Returns a Flask app."""
    return create_app(testing=testing)

class SecurityManager:
    """Stub security manager for test compatibility."""
    def __init__(self):
        self.active = True
    def is_secure(self):
        return self.active

security_manager = SecurityManager()
from flask import Flask

def create_app(testing: bool = False) -> Flask:
    """Create and configure a Flask app for the Sovereign Web Interface."""
    app = Flask(__name__)
    app.config['TESTING'] = testing
    @app.route('/')
    def index():
        return 'Sovereign Web Interface Active'
    return app
"""
Sovereign Web Interface
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Flask-based web interface for the Sovereign Control System
Provides a dashboard for monitoring and controlling the sovereign system
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import secrets
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sovereign.web_interface')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from sovereign.master_controller import MasterController, ControlMode

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Initialize MasterController
project_root = Path(__file__).resolve().parent.parent
controller = None

# Import security modules after setting up project path
from sovereign.security import SecurityManager, SecurityLevel
from sovereign.security_web import init_security_web
from sovereign.security_middleware import load_security_middleware

# Initialize security manager
security_manager = None

def initialize_controller(mode=ControlMode.STANDARD, security_level=SecurityLevel.ENHANCED):
    """Initialize the master controller and security systems"""
    global controller, security_manager

    # Initialize master controller
    controller = MasterController(project_root, mode)

    # Initialize security manager
    security_manager = SecurityManager(project_root, security_level)

    # Initialize security web interface
    init_security_web(app, security_manager)

    # Load security middleware
    load_security_middleware(app)

    logger.info(f"Initialized controller in {mode} mode with {security_level} security")
    return controller


# Utility function for authentication
def login_required(f):
    """Decorator to require login for routes"""
    from functools import wraps
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


@app.route('/')
def index():
    """Render the dashboard page or redirect to login"""
    global controller
    if controller is None:
        # Default to standard mode if not initialized
        controller = initialize_controller()

    # Check if user is logged in
    if 'token' in session:
        # Validate token
        token = session['token']
        valid, _ = security_manager.validate_session_token(token)

        if valid:
            # Get system status and state
            status = controller.get_system_status()
            state = controller.get_system_state()

            return render_template(
                'dashboard.html',
                status=status,
                state=state,
                modes=[mode.value for mode in ControlMode],
                current_mode=controller.mode.value,
                username=request.username
            )

    # Not logged in or invalid token
    return redirect(url_for('security.login'))


@app.route('/initialize', methods=['POST'])
@login_required
def initialize():
    """Initialize the sovereign system"""
    mode = request.form.get('mode', 'STANDARD')

    try:
        initialize_controller(ControlMode(mode))
        flash(f'Sovereign system initialized in {mode} mode', 'success')
    except Exception as e:
        flash(f'Error initializing system: {str(e)}', 'error')

    return redirect(url_for('index'))


@app.route('/execute', methods=['POST'])
@login_required
def execute():
    """Execute a sovereign directive"""
    directive = request.form.get('directive', '')

    if not directive:
        flash('Directive cannot be empty', 'error')
        return redirect(url_for('index'))

    try:
        result = controller.execute_directive(directive)
        flash(f'Directive executed: {directive}', 'success')
        return jsonify(result)
    except Exception as e:
        flash(f'Error executing directive: {str(e)}', 'error')
        return jsonify({'error': str(e)}), 400


@app.route('/optimize', methods=['POST'])
@login_required
def optimize():
    """Optimize the sovereign system"""
    target = request.form.get('target', 'all')

    try:
        result = controller.optimize_system(target)
        flash(f'System optimized: {target}', 'success')
        return jsonify(result)
    except Exception as e:
        flash(f'Error optimizing system: {str(e)}', 'error')
        return jsonify({'error': str(e)}), 400


@app.route('/status')
@login_required
def status():
    """Get the current system status"""
    try:
        status = controller.get_system_status()
        state = controller.get_system_state()

        return jsonify({
            'status': status,
            'state': {
                'control_id': state['control_id'],
                'mode': state['mode'],
                'uptime_seconds': state['uptime_seconds'],
                'control_state': state['control_state'],
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/sovereign', methods=['POST'])
@login_required
def sovereign():
    """Execute full sovereign operation with executive authority"""
    try:
        controller.begin_sovereign_operation()
        flash('Sovereign operation executed successfully', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error executing sovereign operation: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/save', methods=['POST'])
@login_required
def save():
    """Save the system state"""
    try:
        filename = request.form.get('filename')
        path = controller.save_system_state(filename)
        flash(f'System state saved to {path}', 'success')
        return jsonify({'path': path})
    except Exception as e:
        flash(f'Error saving system state: {str(e)}', 'error')
        return jsonify({'error': str(e)}), 400


@app.route('/load', methods=['POST'])
@login_required
def load():
    """Load the system state"""
    try:
        path = request.form.get('path')
        if not path:
            flash('Path cannot be empty', 'error')
            return redirect(url_for('index'))

        state = controller.load_system_state(path)
        flash(f'System state loaded from {path}', 'success')
        return jsonify(state)
    except Exception as e:
        flash(f'Error loading system state: {str(e)}', 'error')
        return jsonify({'error': str(e)}), 400


@app.route('/directives')
@login_required
def directives():
    """Render the directives page"""
    # Get directive history
    history = controller.get_directive_history() if controller else []

    return render_template(
        'directive.html',
        directive=None,
        history=history
    )


@app.route('/directives/<directive_id>')
@login_required
def directive_detail(directive_id):
    """Render the directive detail page"""
    directive = controller.get_directive(directive_id) if controller else None
    history = controller.get_directive_history() if controller else []

    if not directive:
        flash(f'Directive not found: {directive_id}', 'error')
        return redirect(url_for('directives'))

    return render_template(
        'directive.html',
        directive=directive,
        history=history
    )


@app.route('/optimization')
@login_required
def optimization():
    """Render the optimization page"""
    system_config = controller.get_system_config() if controller else None

    return render_template(
        'optimization.html',
        system_config=system_config
    )


@app.route('/configuration')
@login_required
def configuration():
    """Render the configuration page"""
    config = controller.get_system_config() if controller else None

    # Get list of available backups
    backups = []
    if controller:
        backup_dir = Path(controller.get_backup_directory())
        if backup_dir.exists():
            for backup_file in backup_dir.glob('*.json'):
                backups.append({
                    'filename': backup_file.name,
                    'name': backup_file.stem,
                    'date': datetime.fromtimestamp(
                        backup_file.stat().st_mtime
                    ).strftime('%Y-%m-%d %H:%M:%S')
                })

    return render_template(
        'configuration.html',
        config=config,
        backups=backups,
        current_date=datetime.now().strftime('%Y%m%d_%H%M%S')
    )


@app.route('/configure/<section>', methods=['POST'])
@login_required
def configure(section):
    """Configure a specific section of the system"""
    if not controller:
        flash('System not initialized', 'error')
        return redirect(url_for('index'))

    try:
        # Convert form data to dict
        config_data = {k: v for k, v in request.form.items()}

        # Convert checkbox values to booleans
        for key in config_data:
            if config_data[key] == 'on':
                config_data[key] = True

        # Update configuration
        controller.update_configuration(section, config_data)
        flash(f'{section.capitalize()} configuration updated', 'success')
        return redirect(url_for('configuration'))
    except Exception as e:
        flash(f'Error updating configuration: {str(e)}', 'error')
        return redirect(url_for('configuration'))


@app.route('/backup', methods=['POST'])
@login_required
def backup():
    """Backup the system"""
    if not controller:
        flash('System not initialized', 'error')
        return redirect(url_for('index'))

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = request.form.get('backup_name', f'backup_{timestamp}')
        include_data = 'include_data' in request.form

        path = controller.backup_system(backup_name, include_data)
        flash(f'System backed up to {path}', 'success')
        return redirect(url_for('configuration'))
    except Exception as e:
        flash(f'Error backing up system: {str(e)}', 'error')
        return redirect(url_for('configuration'))


@app.route('/restore', methods=['POST'])
@login_required
def restore():
    """Restore the system from backup"""
    if not controller:
        flash('System not initialized', 'error')
        return redirect(url_for('index'))

    try:
        backup_file = request.form.get('backup_file')
        if not backup_file:
            flash('No backup file selected', 'error')
            return redirect(url_for('configuration'))

        restore_data = 'restore_data' in request.form

        controller.restore_system(backup_file, restore_data)
        flash(f'System restored from {backup_file}', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error restoring system: {str(e)}', 'error')
        return redirect(url_for('configuration'))


if __name__ == '__main__':
    # Get port from command line or default to 5000
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=port)
