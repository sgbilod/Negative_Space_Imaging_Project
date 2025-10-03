"""
Performance Optimization Web Interface
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Web interface for the Sovereign Performance Optimization System.
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
from datetime import datetime, timedelta
import platform
import threading
import multiprocessing
try:
    import psutil
except ImportError:
    logging.warning("psutil not installed. Some features will be unavailable.")
    psutil = None

from sovereign.performance import (
    PerformanceManager,
    OptimizationLevel,
    OptimizationTarget,
    get_performance_manager
)
from sovereign.security import SecurityManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sovereign.performance_web')

# Create Blueprint
performance_bp = Blueprint(
    'performance',
    __name__,
    template_folder='templates/performance',
    static_folder='static'
)

# Get reference to the PerformanceManager
performance_manager = get_performance_manager()


def init_performance_web(app, security_manager=None):
    """Initialize the performance web interface with the app"""
    app.register_blueprint(performance_bp, url_prefix='/performance')
    logger.info("Performance web interface initialized")


def admin_required(f):
    """Decorator to require admin privileges for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from sovereign.web_interface import security_manager

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

        # Check if user has admin privileges
        if not security_manager.check_authorization(username, 'administrator'):
            flash('You do not have permission to access this page', 'error')
            return redirect(url_for('index'))

        return f(*args, **kwargs)
    return decorated_function


def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from sovereign.web_interface import security_manager

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


@performance_bp.route('/')
@performance_bp.route('/')
@performance_bp.route('/dashboard')
@login_required
def dashboard():
    """Render the dashboard page"""
    # Get current metrics
    metrics = performance_manager.collect_metrics()

    # Get memory usage history
    memory_history = (
        performance_manager.memory_optimizer.get_memory_usage_history()
    )

    # Get cache statistics
    cache_stats = performance_manager.cache.get_stats()

    # Get slow queries (more than 1 second)
    slow_queries = performance_manager.database_optimizer.get_slow_queries(
        threshold=1.0
    )

    # Get current optimization level
    optimization_level = performance_manager.optimization_level.value

    # Get registered components
    components = performance_manager.optimization_profiles

    # Format memory history for charts
    memory_labels = []
    memory_data = []

    for entry in memory_history[-20:]:  # Last 20 entries
        timestamp = datetime.fromtimestamp(entry['timestamp'])
        memory_labels.append(timestamp.strftime('%H:%M:%S'))
        memory_data.append(entry['rss'])

    return render_template(
        'performance/dashboard.html',
        metrics=metrics.to_dict(),
        memory_history=memory_history,
        cache_stats=cache_stats,
        slow_queries=slow_queries,
        optimization_level=optimization_level,
        components=components,
        memory_labels=memory_labels,
        memory_data=memory_data
    )


@performance_bp.route('/optimize', methods=['POST'])
@admin_required
def optimize():
    """Optimize system performance"""
    # Get form data
    target = request.form.get('target', 'ALL')
    level = request.form.get('level')
    component = request.form.get('component')

    # Set optimization level if specified
    if level:
        performance_manager.set_optimization_level(OptimizationLevel(level))

    # Run optimization
    target_enum = OptimizationTarget(target)

    if component:
        # Optimize specific component
        profile = performance_manager.get_component_profile(component)
        if not profile:
            performance_manager.register_component(
                component,
                targets=[target_enum]
            )

        flash(f'Component {component} optimization profile updated', 'success')
    else:
        # Optimize entire system
        results = performance_manager.optimize_system(target_enum)

        # Format message
        optimizations = len(results['optimizations_applied'])
        flash(f'Applied {optimizations} performance optimizations', 'success')

    return redirect(url_for('performance.dashboard'))


@performance_bp.route('/monitor')
@admin_required
def monitor():
    """Render the performance monitoring page"""
    # Get historical metrics
    metrics_history = performance_manager.metrics_history

    # Format data for charts
    cpu_labels = []
    cpu_data = []
    memory_labels = []
    memory_data = []
    thread_labels = []
    thread_data = []

    for entry in metrics_history[-60:]:  # Last 60 entries
        timestamp = datetime.fromisoformat(entry['timestamp'])

        # CPU data
        cpu_labels.append(timestamp.strftime('%H:%M:%S'))
        cpu_data.append(entry['cpu_usage'])

        # Memory data
        memory_labels.append(timestamp.strftime('%H:%M:%S'))
        memory_data.append(entry['memory_usage'])

        # Thread data
        thread_labels.append(timestamp.strftime('%H:%M:%S'))
        thread_data.append(entry['thread_count'])

    return render_template(
        'performance/monitor.html',
        cpu_labels=cpu_labels,
        cpu_data=cpu_data,
        memory_labels=memory_labels,
        memory_data=memory_data,
        thread_labels=thread_labels,
        thread_data=thread_data
    )


@performance_bp.route('/profiler')
@admin_required
def profiler():
    """Render the profiler page"""
    # Get all profiles
    profiles = performance_manager.profiler.get_all_profiles()

    return render_template(
        'performance/profiler.html',
        profiles=profiles
    )


@performance_bp.route('/profile/<profile_id>')
@admin_required
def profile_detail(profile_id):
    """Render the profile detail page"""
    # Get the profile
    profile = performance_manager.profiler.get_profile(profile_id)

    if not profile:
        flash(f'Profile not found: {profile_id}', 'error')
        return redirect(url_for('performance.profiler'))

    return render_template(
        'performance/profile_detail.html',
        profile=profile,
        profile_id=profile_id
    )


@performance_bp.route('/cache')
@admin_required
def cache():
    """Render the cache management page"""
    # Get cache statistics
    cache_stats = performance_manager.cache.get_stats()

    return render_template(
        'performance/cache.html',
        cache_stats=cache_stats
    )


@performance_bp.route('/cache/clear', methods=['POST'])
@admin_required
def cache_clear():
    """Clear the cache"""
    performance_manager.cache.clear()
    flash('Cache cleared successfully', 'success')
    return redirect(url_for('performance.cache'))


@performance_bp.route('/cache/resize', methods=['POST'])
@admin_required
def cache_resize():
    """Resize the cache"""
    # Get form data
    size = request.form.get('size', type=int)

    if not size or size < 1:
        flash('Invalid cache size', 'error')
        return redirect(url_for('performance.cache'))

    # Resize the cache
    performance_manager.cache.resize(size)
    flash(f'Cache resized to {size}', 'success')
    return redirect(url_for('performance.cache'))


@performance_bp.route('/cache/clear_type', methods=['POST'])
@admin_required
def cache_clear_type():
    """Clear a specific cache type"""
    # Get form data
    cache_type = request.form.get('cache_type')

    if not cache_type:
        flash('Invalid cache type', 'error')
        return redirect(url_for('performance.cache'))

    # Clear the specific cache type
    performance_manager.cache.clear_type(cache_type)
    flash(f'Cache {cache_type} cleared successfully', 'success')
    return redirect(url_for('performance.cache'))


@performance_bp.route('/cache/invalidate_key', methods=['POST'])
@admin_required
def cache_invalidate_key():
    """Invalidate a specific cache key"""
    # Get form data
    cache_key = request.form.get('cache_key')
    cache_type = request.form.get('cache_type')

    if not cache_key:
        flash('Invalid cache key', 'error')
        return redirect(url_for('performance.cache'))

    # Invalidate the specific key
    performance_manager.cache.invalidate(cache_key, cache_type)
    flash(f'Cache key {cache_key} invalidated', 'success')
    return redirect(url_for('performance.cache'))


@performance_bp.route('/cache/configure_type', methods=['POST'])
@admin_required
def cache_configure_type():
    """Configure a specific cache type"""
    # Get form data
    cache_type = request.form.get('cache_type')
    ttl = request.form.get('ttl', type=int)
    max_size = request.form.get('max_size', type=int)
    enabled = 'enabled' in request.form

    if not cache_type:
        flash('Invalid cache type', 'error')
        return redirect(url_for('performance.cache'))

    # Configure the specific cache type
    performance_manager.cache.configure_type(
        cache_type=cache_type,
        ttl=ttl,
        max_size=max_size,
        enabled=enabled
    )

    flash(f'Cache {cache_type} configuration updated', 'success')
    return redirect(url_for('performance.cache'))


@performance_bp.route('/memory')
@admin_required
def memory():
    """Render the memory management page"""
    # Get current memory usage
    memory_usage = performance_manager.memory_optimizer.get_current_memory_usage()

    # Get memory usage history
    memory_history = performance_manager.memory_optimizer.get_memory_usage_history()

    # Get garbage collection stats
    gc_stats = performance_manager.memory_optimizer.get_gc_stats()

    # Get allocation tracking
    allocation_tracking = performance_manager.memory_optimizer.get_allocation_tracking()

    # Format memory history for charts
    labels = []
    rss_data = []
    vms_data = []

    for entry in memory_history[-60:]:  # Last 60 entries
        timestamp = datetime.fromtimestamp(entry['timestamp'])
        labels.append(timestamp.strftime('%H:%M:%S'))
        rss_data.append(entry['rss'])
        vms_data.append(entry['vms'])

    return render_template(
        'performance/memory.html',
        memory_usage=memory_usage,
        gc_stats=gc_stats,
        allocation_tracking=allocation_tracking,
        labels=labels,
        rss_data=rss_data,
        vms_data=vms_data
    )


@performance_bp.route('/memory/gc', methods=['POST'])
@admin_required
def memory_gc():
    """Run garbage collection"""
    # Run garbage collection
    stats = performance_manager.memory_optimizer.force_garbage_collection()

    # Format message
    flash(
        f'Garbage collection completed: '
        f'{stats["collected"]} objects collected in '
        f'{stats["collection_time"]:.4f} seconds',
        'success'
    )

    return redirect(url_for('performance.memory'))


@performance_bp.route('/settings')
@admin_required
def settings():
    """Render the performance settings page"""
    # Get current optimization level
    optimization_level = performance_manager.optimization_level.value

    # Get registered components
    components = performance_manager.optimization_profiles

    # Get thresholds
    thresholds = performance_manager.thresholds

    return render_template(
        'performance/settings.html',
        optimization_level=optimization_level,
        components=components,
        thresholds=thresholds,
        optimization_levels=[level.value for level in OptimizationLevel],
        optimization_targets=[target.value for target in OptimizationTarget]
    )


@performance_bp.route('/settings/update', methods=['POST'])
@admin_required
def settings_update():
    """Update performance settings"""
    # Get form data
    level = request.form.get('optimization_level')

    # Update thresholds
    for threshold in performance_manager.thresholds:
        value = request.form.get(f'threshold_{threshold}', type=float)
        if value is not None:
            performance_manager.thresholds[threshold] = value

    # Set optimization level
    if level:
        performance_manager.set_optimization_level(OptimizationLevel(level))

    flash('Performance settings updated', 'success')
    return redirect(url_for('performance.settings'))


@performance_bp.route('/component/<component_name>')
@admin_required
def component_detail(component_name):
    """Render the component detail page"""
    # Get component profile
    profile = performance_manager.get_component_profile(component_name)

    if not profile:
        flash(f'Component not found: {component_name}', 'error')
        return redirect(url_for('performance.settings'))

    return render_template(
        'performance/component_detail.html',
        profile=profile,
        component_name=component_name,
        optimization_levels=[level.value for level in OptimizationLevel],
        optimization_targets=[target.value for target in OptimizationTarget]
    )


@performance_bp.route('/component/<component_name>/update', methods=['POST'])
@admin_required
def component_update(component_name):
    """Update component settings"""
    # Get component profile
    profile = performance_manager.get_component_profile(component_name)

    if not profile:
        flash(f'Component not found: {component_name}', 'error')
        return redirect(url_for('performance.settings'))

    # Get form data
    level = request.form.get('level')
    enabled = 'enabled' in request.form
    cache_enabled = 'cache_enabled' in request.form
    parallel_enabled = 'parallel_enabled' in request.form
    max_threads = request.form.get('max_threads', type=int)
    max_processes = request.form.get('max_processes', type=int)
    cache_size = request.form.get('cache_size', type=int)
    cache_ttl = request.form.get('cache_ttl', type=int)

    # Update profile
    if level:
        profile.level = OptimizationLevel(level)

    profile.enabled = enabled
    profile.cache_enabled = cache_enabled
    profile.parallel_enabled = parallel_enabled

    if max_threads is not None and max_threads > 0:
        profile.max_threads = max_threads

    if max_processes is not None and max_processes > 0:
        profile.max_processes = max_processes

    if cache_size is not None and cache_size > 0:
        profile.cache_size = cache_size

    if cache_ttl is not None and cache_ttl > 0:
        profile.cache_ttl = cache_ttl

    flash(f'Component {component_name} settings updated', 'success')
    return redirect(url_for('performance.component_detail', component_name=component_name))


@performance_bp.route('/api/metrics')
@login_required
def api_metrics():
    """API endpoint for current metrics"""
    # Get current metrics
    metrics = performance_manager.collect_metrics()

    return jsonify(metrics.to_dict())


@performance_bp.route('/api/memory')
@login_required
def api_memory():
    """API endpoint for memory usage"""
    # Get current memory usage
    memory_usage = (
        performance_manager.memory_optimizer.get_current_memory_usage()
    )

    return jsonify(memory_usage)


@performance_bp.route('/api/cache')
@login_required
def api_cache():
    """API endpoint for cache statistics"""
    # Get cache statistics
    cache_stats = performance_manager.cache.get_stats()

    return jsonify(cache_stats)


@performance_bp.route('/api/report')
@login_required
def api_report():
    """API endpoint for performance report"""
    # Get performance report
    report_data = performance_manager.get_performance_report()

    return jsonify(report_data)

    return jsonify(report_data)


@performance_bp.route('/api/database')
@login_required
def api_database():
    """API endpoint for database statistics"""
    query_stats = performance_manager.database_optimizer.get_query_stats()

    # Convert datetime objects to strings for JSON serialization
    for query_id, stats in query_stats.items():
        if stats['last_executed']:
            stats['last_executed'] = stats['last_executed'].isoformat()

    slow_queries = (
        performance_manager.database_optimizer.get_slow_queries(threshold=1.0)
    )
    cache_stats = performance_manager.database_optimizer.cache.get_stats()

    return jsonify({
        'query_stats': query_stats,
        'slow_queries': slow_queries,
        'cache_stats': cache_stats
    })


@performance_bp.route('/api/details')
@login_required
def api_details():
    """API endpoint for detailed performance metrics"""
    # Get current metrics
    current_metrics = performance_manager.current_metrics.to_dict()

    # Get system information
    system_info = {
        'cpu_count': multiprocessing.cpu_count(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'process_id': os.getpid()
    }

    return jsonify({
        'current_metrics': current_metrics,
        'system_info': system_info
    })


@performance_bp.route('/details')
@admin_required
def details():
    """Render the performance details page"""
    # Get performance report
    report = performance_manager.get_performance_report()

    # Add color coding for health status
    if report.get('health_status') == 'EXCELLENT':
        report['health_status_color'] = 'success'
    elif report.get('health_status') == 'GOOD':
        report['health_status_color'] = 'info'
    elif report.get('health_status') == 'FAIR':
        report['health_status_color'] = 'warning'
    else:
        report['health_status_color'] = 'danger'

    # Add color coding for performance score
    score = report.get('performance_score', 0)
    if score >= 80:
        report['score_color'] = 'success'
    elif score >= 60:
        report['score_color'] = 'info'
    elif score >= 40:
        report['score_color'] = 'warning'
    else:
        report['score_color'] = 'danger'

    return render_template(
        'performance/details.html',
        report=report
    )


@performance_bp.route('/database')
@admin_required
def database():
    """Render the database optimization page"""
    # Get database statistics
    db_stats = performance_manager.get_database_stats()

    # Get connection history
    connection_history = performance_manager.get_connection_history()

    # Format connection history for charts
    connection_labels = []
    connection_data = []

    for entry in connection_history[-30:]:  # Last 30 entries
        timestamp = datetime.fromtimestamp(entry['timestamp'])
        connection_labels.append(timestamp.strftime('%H:%M:%S'))
        connection_data.append(entry['active_connections'])

    return render_template(
        'performance/database.html',
        db_stats=db_stats,
        connection_labels=connection_labels,
        connection_data=connection_data
    )


@performance_bp.route('/database/optimize', methods=['POST'])
@admin_required
def database_optimize():
    """Optimize database performance"""
    # Get form data
    level = request.form.get('level', 'STANDARD')
    optimize_indexes = 'optimize_indexes' in request.form
    analyze_statistics = 'analyze_statistics' in request.form
    vacuum_tables = 'vacuum_tables' in request.form

    # Run database optimization
    results = performance_manager.optimize_database(
        level=level,
        optimize_indexes=optimize_indexes,
        analyze_statistics=analyze_statistics,
        vacuum_tables=vacuum_tables
    )

    # Format message
    optimizations = len(results.get('optimizations_applied', []))
    flash(f'Applied {optimizations} database optimizations', 'success')

    return redirect(url_for('performance.database'))


@performance_bp.route('/database/optimize_query', methods=['POST'])
@admin_required
def database_optimize_query():
    """Optimize a specific database query"""
    # Get form data
    query_id = request.form.get('query_id')
    optimized_query = request.form.get('optimized_query')

    if not query_id or not optimized_query:
        flash('Invalid query information', 'error')
        return redirect(url_for('performance.database'))

    # Apply the optimized query
    performance_manager.optimize_query(query_id, optimized_query)

    flash('Query optimization applied', 'success')
    return redirect(url_for('performance.database'))


@performance_bp.route('/database/create_index', methods=['POST'])
@admin_required
def database_create_index():
    """Create a database index"""
    # Get form data
    table = request.form.get('table')
    columns = request.form.get('columns')
    index_name = request.form.get('index_name')
    unique = 'unique' in request.form
    concurrently = 'concurrently' in request.form

    if not table or not columns or not index_name:
        flash('Missing required index information', 'error')
        return redirect(url_for('performance.database'))

    # Create the index
    performance_manager.create_index(
        table=table,
        columns=columns.split(','),
        index_name=index_name,
        unique=unique,
        concurrently=concurrently
    )

    flash(f'Index {index_name} created successfully', 'success')
    return redirect(url_for('performance.database'))


@performance_bp.route('/database/pool_update', methods=['POST'])
@admin_required
def database_pool_update():
    """Update database connection pool settings"""
    # Get form data
    pool_size = request.form.get('pool_size', type=int)
    pool_timeout = request.form.get('pool_timeout', type=int)

    if not pool_size or pool_size < 1:
        flash('Invalid pool size', 'error')
        return redirect(url_for('performance.database'))

    # Update connection pool settings
    performance_manager.update_db_pool_settings(
        pool_size=pool_size,
        pool_timeout=pool_timeout
    )

    flash('Database connection pool settings updated', 'success')
    return redirect(url_for('performance.database'))


@performance_bp.route('/database/cache_clear', methods=['POST'])
@admin_required
def database_cache_clear():
    """Clear database query cache"""
    # Clear the query cache
    performance_manager.clear_query_cache()

    flash('Database query cache cleared', 'success')
    return redirect(url_for('performance.database'))


@performance_bp.route('/memory/config_update', methods=['POST'])
@admin_required
def memory_config_update():
    """Update memory configuration settings"""
    # Get form data
    max_memory_usage = request.form.get('max_memory_usage', type=int)
    gc_threshold = request.form.get('gc_threshold', type=int)
    memory_check_interval = request.form.get('memory_check_interval', type=int)
    enable_tracking = 'enable_tracking' in request.form
    auto_gc = 'auto_gc' in request.form
    leak_detection = 'leak_detection' in request.form

    # Update memory configuration
    performance_manager.memory_optimizer.update_config(
        max_memory_mb=max_memory_usage,
        gc_threshold_mb=gc_threshold,
        check_interval_sec=memory_check_interval,
        enable_tracking=enable_tracking,
        auto_gc=auto_gc,
        leak_detection=leak_detection
    )

    flash('Memory configuration updated', 'success')
    return redirect(url_for('performance.memory'))


@performance_bp.route('/database')
@admin_required
def database_optimization():
    """Database optimization view"""
    # Get database query stats
    query_stats = performance_manager.database_optimizer.get_query_stats()

    # Get slow queries (more than 1 second)
    slow_queries = performance_manager.database_optimizer.get_slow_queries(
        threshold=1.0
    )

    # Get optimization rules
    optimization_rules = (
        performance_manager.database_optimizer.optimization_rules
    )

    # Get database cache stats
    cache_stats = performance_manager.database_optimizer.cache.get_stats()

    return render_template(
        'performance/database.html',
        query_stats=query_stats,
        slow_queries=slow_queries,
        optimization_rules=optimization_rules,
        cache_stats=cache_stats
    )


@performance_bp.route('/database/clear_stats', methods=['POST'])
@admin_required
def database_clear_stats():
    """Clear database query statistics"""
    performance_manager.database_optimizer.clear_stats()
    flash('Database query statistics cleared', 'success')
    return redirect(url_for('performance.database_optimization'))


@performance_bp.route('/database/add_rule', methods=['POST'])
@admin_required
def database_add_rule():
    """Add a database query optimization rule"""
    pattern = request.form.get('pattern', '')
    replacement = request.form.get('replacement', '')

    if not pattern or not replacement:
        flash('Pattern and replacement are required', 'error')
        return redirect(url_for('performance.database_optimization'))

    # Add the optimization rule
    performance_manager.database_optimizer.add_optimization_rule(
        pattern=pattern,
        replacement=replacement
    )

    flash('Database optimization rule added', 'success')
    return redirect(url_for('performance.database_optimization'))


@performance_bp.route('/database/clear_cache', methods=['POST'])
@admin_required
def database_clear_cache():
    """Clear the database query cache"""
    performance_manager.database_optimizer.cache.clear()
    flash('Database query cache cleared', 'success')
    return redirect(url_for('performance.database_optimization'))


@performance_bp.route('/details_view')
@admin_required
def details_view():
    """Detailed performance metrics view"""
    # Get performance metrics history
    metrics_history = performance_manager.metrics_history

    # Get current metrics
    current_metrics = performance_manager.current_metrics.to_dict()

    # Get system information
    system_info = {
        'cpu_count': multiprocessing.cpu_count(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'process_id': os.getpid(),
        'start_time': datetime.fromtimestamp(
            psutil.Process(os.getpid()).create_time()
        ).strftime('%Y-%m-%d %H:%M:%S')
    }

    # Get component profiles
    profiles = performance_manager.optimization_profiles

    return render_template(
        'performance/details.html',
        metrics_history=metrics_history,
        current_metrics=current_metrics,
        system_info=system_info,
        profiles=profiles
    )
