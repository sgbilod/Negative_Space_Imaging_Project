"""
Database Query Optimization Implementation
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

Implements advanced query optimization strategies for the Sovereign database system.
"""

import re
import logging
import time
from typing import Dict, List, Any, Callable, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sovereign.query_optimizer')


@dataclass
class QueryPattern:
    """Pattern for query optimization"""
    name: str
    description: str
    pattern: str
    replacement: str
    regex: bool = False
    condition: Optional[Callable[[str], bool]] = None
    savings_estimate: float = 0.0  # Estimated time savings in seconds
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH


class QueryOptimizationEngine:
    """Advanced engine for SQL query optimization"""

    def __init__(self):
        self.optimization_patterns: List[QueryPattern] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.automatic_mode: bool = False
        self.learning_mode: bool = True
        self.detected_patterns: Dict[str, Dict[str, Any]] = {}

        # Initialize with default optimization patterns
        self._load_default_patterns()

    def _load_default_patterns(self) -> None:
        """Load default optimization patterns"""
        # SELECT * optimization
        self.add_pattern(
            QueryPattern(
                name="select_columns",
                description="Replace SELECT * with specific columns",
                pattern="SELECT *",
                replacement="SELECT id, name, created_at",
                regex=False,
                condition=lambda q: "WHERE" in q and "GROUP BY" not in q,
                savings_estimate=0.15,
                risk_level="LOW"
            )
        )

        # Add index hint
        self.add_pattern(
            QueryPattern(
                name="add_index_hint",
                description="Add index hint for queries with WHERE clauses on indexed columns",
                pattern=r"FROM\s+(\w+)\s+WHERE\s+(\w+)\s*=",
                replacement=r"FROM \1 USE INDEX(\2_idx) WHERE \2 =",
                regex=True,
                condition=lambda q: "JOIN" not in q,
                savings_estimate=0.25,
                risk_level="MEDIUM"
            )
        )

        # Optimize LIKE queries
        self.add_pattern(
            QueryPattern(
                name="optimize_like",
                description="Optimize LIKE '%text%' to use full-text search",
                pattern=r"LIKE\s+['\"]%(.+?)%['\"]",
                replacement=r"MATCH AGAINST('\1' IN BOOLEAN MODE)",
                regex=True,
                condition=lambda q: "ORDER BY" in q,
                savings_estimate=0.5,
                risk_level="MEDIUM"
            )
        )

        # Add LIMIT to queries without it
        self.add_pattern(
            QueryPattern(
                name="add_limit",
                description="Add LIMIT to queries without it",
                pattern=r"(SELECT.+FROM.+)(?<!LIMIT\s+\d+)$",
                replacement=r"\1 LIMIT 1000",
                regex=True,
                condition=lambda q: "WHERE" in q and "LIMIT" not in q,
                savings_estimate=0.4,
                risk_level="LOW"
            )
        )

        # Optimize COUNT(*)
        self.add_pattern(
            QueryPattern(
                name="optimize_count",
                description="Optimize COUNT(*) queries",
                pattern="COUNT(*)",
                replacement="COUNT(1)",
                regex=False,
                condition=lambda q: "GROUP BY" not in q,
                savings_estimate=0.05,
                risk_level="LOW"
            )
        )

        # Use EXISTS instead of COUNT for existence check
        self.add_pattern(
            QueryPattern(
                name="use_exists",
                description="Use EXISTS instead of COUNT for existence checks",
                pattern=r"SELECT\s+COUNT\(\*\)\s+FROM\s+(\w+)\s+WHERE",
                replacement=r"SELECT EXISTS(SELECT 1 FROM \1 WHERE",
                regex=True,
                condition=lambda q: "HAVING" not in q and ">" not in q and "<" not in q,
                savings_estimate=0.3,
                risk_level="MEDIUM"
            )
        )

        # Optimize JOIN with USING
        self.add_pattern(
            QueryPattern(
                name="optimize_join",
                description="Optimize JOIN with USING clause",
                pattern=r"(\w+)\s+JOIN\s+(\w+)\s+ON\s+\1\.(\w+)\s*=\s*\2\.\3",
                replacement=r"\1 JOIN \2 USING(\3)",
                regex=True,
                condition=lambda q: "LEFT" not in q and "RIGHT" not in q,
                savings_estimate=0.1,
                risk_level="LOW"
            )
        )

        logger.info(f"Loaded {len(self.optimization_patterns)} default optimization patterns")

    def add_pattern(self, pattern: QueryPattern) -> None:
        """Add a new optimization pattern"""
        self.optimization_patterns.append(pattern)
        logger.debug(f"Added optimization pattern: {pattern.name}")

    def remove_pattern(self, pattern_name: str) -> bool:
        """Remove an optimization pattern by name"""
        initial_count = len(self.optimization_patterns)
        self.optimization_patterns = [p for p in self.optimization_patterns if p.name != pattern_name]

        removed = initial_count > len(self.optimization_patterns)
        if removed:
            logger.debug(f"Removed optimization pattern: {pattern_name}")

        return removed

    def optimize_query(self, query: str, max_optimizations: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Optimize a SQL query using registered patterns

        Args:
            query: The SQL query to optimize
            max_optimizations: Maximum number of optimizations to apply

        Returns:
            Tuple of (optimized query, list of applied optimizations)
        """
        if not query:
            return query, []

        original_query = query
        optimized_query = query
        applied_optimizations = []

        # Apply each pattern in order
        for pattern in self.optimization_patterns:
            # Skip if we've reached max optimizations
            if len(applied_optimizations) >= max_optimizations:
                break

            # Check condition if provided
            if pattern.condition and not pattern.condition(optimized_query):
                continue

            # Apply pattern
            if pattern.regex:
                # Use regex replacement
                try:
                    new_query = re.sub(pattern.pattern, pattern.replacement, optimized_query)
                    if new_query != optimized_query:
                        applied_optimizations.append({
                            'pattern_name': pattern.name,
                            'description': pattern.description,
                            'risk_level': pattern.risk_level,
                            'savings_estimate': pattern.savings_estimate
                        })
                        optimized_query = new_query
                except re.error as e:
                    logger.error(f"Regex error in pattern {pattern.name}: {e}")
            else:
                # Use simple string replacement
                if pattern.pattern in optimized_query:
                    new_query = optimized_query.replace(pattern.pattern, pattern.replacement)
                    applied_optimizations.append({
                        'pattern_name': pattern.name,
                        'description': pattern.description,
                        'risk_level': pattern.risk_level,
                        'savings_estimate': pattern.savings_estimate
                    })
                    optimized_query = new_query

        # Record optimization in history
        if original_query != optimized_query:
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'original_query': original_query,
                'optimized_query': optimized_query,
                'applied_optimizations': applied_optimizations,
                'estimated_savings': sum(opt['savings_estimate'] for opt in applied_optimizations)
            })

        return optimized_query, applied_optimizations

    def analyze_query_patterns(self, queries: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze a set of queries to detect optimization patterns

        Args:
            queries: Dictionary of query_id -> query_stats

        Returns:
            Dictionary of optimization recommendations by query_id
        """
        recommendations = {}

        for query_id, stats in queries.items():
            query_text = stats.get('query_text', '')
            if not query_text:
                continue

            # Get possible optimizations
            optimized_query, optimizations = self.optimize_query(query_text)

            if optimizations:
                recommendations[query_id] = {
                    'original_query': query_text,
                    'optimized_query': optimized_query,
                    'optimizations': optimizations,
                    'execution_count': stats.get('execution_count', 0),
                    'avg_time': stats.get('avg_time', 0),
                    'total_time': stats.get('total_time', 0),
                    'estimated_savings_per_query': sum(opt['savings_estimate'] for opt in optimizations),
                    'estimated_total_savings': sum(opt['savings_estimate'] for opt in optimizations) *
                                               stats.get('execution_count', 0)
                }

                # Learn from this query if in learning mode
                if self.learning_mode:
                    self._learn_from_query(query_text, stats)

        return recommendations

    def _learn_from_query(self, query_text: str, stats: Dict[str, Any]) -> None:
        """
        Learn optimization patterns from a slow query

        Args:
            query_text: The query text
            stats: Query statistics
        """
        # Only learn from slow queries
        if stats.get('avg_time', 0) < 0.5:  # Less than 500ms
            return

        # Detect common patterns in slow queries
        self._detect_missing_indexes(query_text, stats)
        self._detect_inefficient_joins(query_text, stats)
        self._detect_missing_limits(query_text, stats)

    def _detect_missing_indexes(self, query_text: str, stats: Dict[str, Any]) -> None:
        """Detect missing indexes in query"""
        # Simple pattern matching for potential index candidates
        where_pattern = r"WHERE\s+(\w+\.\w+|\w+)\s*[=><]"
        order_pattern = r"ORDER\s+BY\s+(\w+\.\w+|\w+)"

        where_matches = re.findall(where_pattern, query_text, re.IGNORECASE)
        order_matches = re.findall(order_pattern, query_text, re.IGNORECASE)

        # Combine potential index fields
        potential_indexes = set()
        for field in where_matches + order_matches:
            if '.' in field:
                table, column = field.split('.')
                potential_indexes.add((table, column))
            else:
                potential_indexes.add((None, field))

        # Record potential missing indexes
        if potential_indexes:
            # Generate a pattern key
            tables = re.findall(r"FROM\s+(\w+)|JOIN\s+(\w+)", query_text, re.IGNORECASE)
            tables_str = "_".join(sorted([t[0] or t[1] for t in tables if t[0] or t[1]]))
            pattern_key = f"missing_index_{tables_str}"

            if pattern_key not in self.detected_patterns:
                self.detected_patterns[pattern_key] = {
                    'type': 'missing_index',
                    'tables': tables_str,
                    'potential_indexes': [],
                    'query_count': 0,
                    'avg_time': 0,
                    'last_seen': None
                }

            # Update pattern data
            pattern = self.detected_patterns[pattern_key]
            pattern['query_count'] += 1
            pattern['avg_time'] = ((pattern['avg_time'] * (pattern['query_count'] - 1)) +
                                   stats.get('avg_time', 0)) / pattern['query_count']
            pattern['last_seen'] = datetime.now()

            # Add potential indexes
            for table, column in potential_indexes:
                index_entry = {'table': table, 'column': column}
                if index_entry not in pattern['potential_indexes']:
                    pattern['potential_indexes'].append(index_entry)

    def _detect_inefficient_joins(self, query_text: str, stats: Dict[str, Any]) -> None:
        """Detect inefficient joins in query"""
        # Check for cartesian joins (missing join condition)
        from_tables = re.findall(r"FROM\s+(\w+)", query_text, re.IGNORECASE)
        join_tables = re.findall(r"JOIN\s+(\w+)", query_text, re.IGNORECASE)
        join_conditions = re.findall(r"ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", query_text, re.IGNORECASE)

        all_tables = from_tables + join_tables

        # If we have multiple tables but no/few join conditions, it might be a cartesian join
        if len(all_tables) > 1 and len(join_conditions) < len(all_tables) - 1:
            pattern_key = "inefficient_join_" + "_".join(sorted(all_tables))

            if pattern_key not in self.detected_patterns:
                self.detected_patterns[pattern_key] = {
                    'type': 'inefficient_join',
                    'tables': all_tables,
                    'query_count': 0,
                    'avg_time': 0,
                    'last_seen': None,
                    'sample_query': query_text
                }

            # Update pattern data
            pattern = self.detected_patterns[pattern_key]
            pattern['query_count'] += 1
            pattern['avg_time'] = ((pattern['avg_time'] * (pattern['query_count'] - 1)) +
                                  stats.get('avg_time', 0)) / pattern['query_count']
            pattern['last_seen'] = datetime.now()

    def _detect_missing_limits(self, query_text: str, stats: Dict[str, Any]) -> None:
        """Detect missing LIMIT clause in query"""
        # Check if it's a SELECT query without LIMIT
        if query_text.strip().upper().startswith('SELECT') and 'LIMIT' not in query_text.upper():
            pattern_key = "missing_limit"

            if pattern_key not in self.detected_patterns:
                self.detected_patterns[pattern_key] = {
                    'type': 'missing_limit',
                    'query_count': 0,
                    'avg_time': 0,
                    'last_seen': None,
                    'queries': []
                }

            # Update pattern data
            pattern = self.detected_patterns[pattern_key]
            pattern['query_count'] += 1
            pattern['avg_time'] = ((pattern['avg_time'] * (pattern['query_count'] - 1)) +
                                  stats.get('avg_time', 0)) / pattern['query_count']
            pattern['last_seen'] = datetime.now()

            # Add sample query if we don't have many
            if len(pattern['queries']) < 5:
                pattern['queries'].append({
                    'query_text': query_text,
                    'avg_time': stats.get('avg_time', 0)
                })

    def get_detected_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get all detected optimization patterns"""
        return self.detected_patterns

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the optimization history"""
        return self.optimization_history

    def get_pattern_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations based on detected patterns"""
        recommendations = []

        # Process missing index patterns
        for pattern_key, pattern in self.detected_patterns.items():
            if pattern['type'] == 'missing_index' and pattern['query_count'] >= 3:
                for index in pattern['potential_indexes']:
                    table = index['table'] or pattern['tables'].split('_')[0]
                    column = index['column']

                    recommendations.append({
                        'type': 'create_index',
                        'priority': 'HIGH' if pattern['avg_time'] > 1.0 else 'MEDIUM',
                        'description': f"Create index on {table}.{column}",
                        'sql': f"CREATE INDEX idx_{table}_{column} ON {table}({column});",
                        'estimated_savings': pattern['avg_time'] * 0.5 * pattern['query_count'],
                        'affected_queries': pattern['query_count']
                    })

            elif pattern['type'] == 'inefficient_join' and pattern['query_count'] >= 2:
                recommendations.append({
                    'type': 'optimize_join',
                    'priority': 'HIGH' if pattern['avg_time'] > 1.0 else 'MEDIUM',
                    'description': f"Optimize inefficient join across {', '.join(pattern['tables'])}",
                    'sample_query': pattern['sample_query'],
                    'estimated_savings': pattern['avg_time'] * 0.7 * pattern['query_count'],
                    'affected_queries': pattern['query_count']
                })

            elif pattern['type'] == 'missing_limit' and pattern['query_count'] >= 5:
                recommendations.append({
                    'type': 'add_limit',
                    'priority': 'MEDIUM',
                    'description': "Add LIMIT clause to queries",
                    'samples': pattern['queries'][:3],
                    'estimated_savings': pattern['avg_time'] * 0.3 * pattern['query_count'],
                    'affected_queries': pattern['query_count']
                })

        # Sort by estimated savings
        recommendations.sort(key=lambda x: x['estimated_savings'], reverse=True)

        return recommendations

    def enable_automatic_mode(self, enabled: bool = True) -> None:
        """Enable or disable automatic optimization mode"""
        self.automatic_mode = enabled
        logger.info(f"Automatic optimization mode {'enabled' if enabled else 'disabled'}")

    def enable_learning_mode(self, enabled: bool = True) -> None:
        """Enable or disable learning mode"""
        self.learning_mode = enabled
        logger.info(f"Learning mode {'enabled' if enabled else 'disabled'}")

    def reset_learning(self) -> None:
        """Reset all learned patterns"""
        self.detected_patterns = {}
        logger.info("Reset all learned optimization patterns")


# Create singleton instance
query_optimization_engine = QueryOptimizationEngine()

def get_query_optimization_engine() -> QueryOptimizationEngine:
    """Get the singleton query optimization engine"""
    return query_optimization_engine
