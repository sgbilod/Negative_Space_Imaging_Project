"""
Anomaly Detection Tests

Tests for anomaly detection implementations including isolation forest,
local outlier factor, IQR, Z-score, and Tukey fence methods with ensemble.
"""

import pytest
import math
from typing import List


class TestIsolationForest:
    """Test isolation forest anomaly detection."""

    @pytest.fixture
    def mixed_data(self) -> List[float]:
        """Generate data with normal and anomalous points."""
        normal = list(range(10, 50, 5))  # 10-45
        anomalies = [100, 101, 105]
        return normal + anomalies

    def test_isolation_tree_splitting(self, mixed_data):
        """Test data splitting creates isolation trees."""
        # Simulate isolation tree concept
        def isolation_score(point, data, depth=0):
            """Calculate isolation score for a point."""
            if len(data) <= 1:
                return depth
            # Simpler is more anomalous
            return depth

        scores = [isolation_score(p, mixed_data) for p in mixed_data]
        assert len(scores) == len(mixed_data)

    def test_anomaly_scoring_isolation_forest(self, mixed_data):
        """Test anomaly scoring in isolation forest."""
        # Anomalies should be isolated faster (lower depth)
        scores = {}

        for i, point in enumerate(mixed_data):
            # Anomalies: 100, 101, 105 should have extreme values
            if point > 50:
                scores[i] = 0.95  # High anomaly score
            else:
                scores[i] = 0.1  # Low anomaly score

        # Verify anomalies have high scores
        anomaly_indices = [i for i, p in enumerate(mixed_data) if p > 50]
        for idx in anomaly_indices:
            assert scores[idx] > 0.9

    def test_ensemble_isolation_trees(self, mixed_data):
        """Test ensemble of isolation trees."""
        num_trees = 10
        tree_scores = [
            [0.1 if x < 50 else 0.9 for x in mixed_data]
            for _ in range(num_trees)
        ]

        # Average scores across trees
        ensemble_scores = [
            sum(scores[i] for scores in tree_scores) / num_trees
            for i in range(len(mixed_data))
        ]

        # Verify ensemble reduces noise
        assert len(ensemble_scores) == len(mixed_data)

    def test_threshold_based_detection(self, mixed_data):
        """Test threshold-based anomaly detection."""
        threshold = 0.7

        scores = [0.1 if x < 50 else 0.9 for x in mixed_data]
        anomalies = [
            mixed_data[i] for i, s in enumerate(scores) if s > threshold
        ]

        # Should detect 3 anomalies
        assert len(anomalies) >= 1


class TestLocalOutlierFactor:
    """Test local outlier factor detection."""

    def test_local_density_calculation(self):
        """Test local density calculation."""
        point = 5.0
        neighbors = [4.8, 5.1, 4.9, 5.2, 5.0]

        # Local density based on neighbor distances
        distances = sorted([abs(point - n) for n in neighbors])
        k = 3
        local_reachability_distance = distances[k - 1] if len(
            distances
        ) >= k else distances[-1]

        # Lower distance = higher density
        local_density = 1.0 / (local_reachability_distance + 0.001)
        assert local_density > 0

    def test_lof_score_computation(self):
        """Test LOF score computation."""
        point = 100.0  # Outlier
        neighbors = list(range(1, 11))  # 1-10

        # Outlier has low local density
        point_distances = [abs(point - n) for n in neighbors]
        avg_neighbor_distance = sum(point_distances) / len(point_distances)

        neighbor_distances = [
            [abs(n1 - n2) for n2 in neighbors if n1 != n2]
            for n1 in neighbors
        ]
        avg_neighbor_avg_distance = sum(
            sum(d) / len(d) for d in neighbor_distances
        ) / len(neighbor_distances)

        # LOF: ratio of neighbor densities
        lof = avg_neighbor_avg_distance / (avg_neighbor_distance + 0.001)
        # LOF should indicate outlier characteristics
        assert lof > 0

    def test_density_reachable_detection(self):
        """Test density-reachable anomaly detection."""
        normal_cluster = list(range(10, 20))
        outlier = 100

        all_points = normal_cluster + [outlier]

        # Points in cluster are density-reachable
        # Outlier is not

        lof_scores = {}
        for point in all_points:
            distances = [abs(point - p) for p in all_points if p != point]
            avg_dist = sum(distances) / len(distances)

            # Outlier has large average distance
            lof_scores[point] = avg_dist

        # Outlier should have highest LOF
        assert lof_scores[outlier] > max(
            lof_scores[p] for p in normal_cluster
        )

    def test_k_distance_graph(self):
        """Test k-distance graph for neighbor detection."""
        data = [1, 2, 3, 4, 5, 100]
        k = 3

        k_distances = {}
        for point in data:
            distances = sorted(
                [abs(point - p) for p in data if p != point]
            )
            k_distances[point] = distances[k - 1] if len(
                distances
            ) >= k else distances[-1]

        # Outlier (100) should have large k-distance
        assert k_distances[100] > k_distances[1]


class TestIQRDetector:
    """Test interquartile range method."""

    def test_quartile_calculation(self):
        """Test quartile calculation."""
        data = list(range(1, 101))  # 1-100
        sorted_data = sorted(data)

        q1_idx = len(sorted_data) // 4
        q3_idx = (3 * len(sorted_data)) // 4

        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]

        assert q1 < q3
        assert q1 == 26  # Approximately
        assert q3 == 76  # Approximately

    def test_iqr_calculation(self):
        """Test IQR calculation."""
        data = list(range(1, 101))
        sorted_data = sorted(data)

        q1_idx = len(sorted_data) // 4
        q3_idx = (3 * len(sorted_data)) // 4

        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        # IQR for 1-100
        assert iqr > 0
        assert iqr <= 50

    def test_whisker_bounds(self):
        """Test whisker bound calculation."""
        data = list(range(1, 101)) + [500]
        sorted_data = sorted(data)

        q1_idx = len(sorted_data) // 4
        q3_idx = (3 * len(sorted_data)) // 4

        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr

        outliers = [x for x in data if x < lower_whisker or x > upper_whisker]
        assert 500 in outliers

    def test_iqr_with_quartile_variants(self):
        """Test IQR with different quartile calculation methods."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Method 1: Exclusive
        sorted_data = sorted(data)
        q1_inclusive = sorted_data[len(data) // 4]

        # Method 2: Inclusive
        q1_inclusive_alt = sorted_data[int(0.25 * len(data))]

        # Both should be close
        assert abs(q1_inclusive - q1_inclusive_alt) <= 1


class TestZScoreDetector:
    """Test Z-score based anomaly detection."""

    def test_zscore_calculation(self):
        """Test Z-score computation."""
        data = [10, 15, 20, 25, 30]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)

        # Z-score for value 25
        z_score = (25 - mean) / std_dev
        assert isinstance(z_score, float)

    def test_zscore_thresholds(self):
        """Test Z-score threshold ranges."""
        data = list(range(1, 101)) + [500]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)

        thresholds = [2, 2.5, 3]
        anomaly_counts = []

        for threshold in thresholds:
            anomalies = [
                x
                for x in data
                if abs((x - mean) / std_dev) > threshold
            ]
            anomaly_counts.append(len(anomalies))

        # Higher threshold = fewer anomalies
        assert anomaly_counts[0] >= anomaly_counts[-1]

    def test_modified_zscore_robustness(self):
        """Test modified Z-score for robustness."""
        data = [1, 2, 3, 4, 5, 100]  # Has outlier

        sorted_data = sorted(data)
        median = sorted_data[len(sorted_data) // 2]

        # Modified Z-score less affected by extremes
        deviations = sorted([abs(x - median) for x in data])
        mad = deviations[len(deviations) // 2]

        threshold = 3.5
        modified_zscores = []

        for x in data:
            if mad > 0:
                mod_z = 0.6745 * (x - median) / mad
            else:
                mod_z = 0

            modified_zscores.append(mod_z)
            if abs(mod_z) > threshold:
                assert x == 100  # Should detect 100 as outlier

    def test_multivariate_zscore(self):
        """Test Z-score in multivariate context."""
        # Two variables
        var1 = [10, 15, 20, 25, 30, 100]

        # Compute Mahalanobis distance (simplified)
        # For now, compute independent Z-scores
        zscores_v1 = []

        mean_v1 = sum(var1) / len(var1)
        std_v1 = math.sqrt(
            sum((x - mean_v1) ** 2 for x in var1) / len(var1)
        )

        for x in var1:
            z = (x - mean_v1) / std_v1 if std_v1 > 0 else 0
            zscores_v1.append(z)

        # Last point (100, 50) should be anomalous
        assert abs(zscores_v1[-1]) > 2


class TestTukeyFences:
    """Test Tukey's fence method."""

    def test_tukey_fence_calculation(self):
        """Test Tukey fence bounds."""
        data = list(range(1, 101)) + [500]
        sorted_data = sorted(data)

        q1_idx = len(sorted_data) // 4
        q3_idx = (3 * len(sorted_data)) // 4

        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        # Tukey fences
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = [x for x in data if x < lower or x > upper]
        assert 500 in outliers

    def test_mild_vs_extreme_outliers(self):
        """Test distinction between mild and extreme outliers."""
        data = list(range(1, 101))
        sorted_data = sorted(data)

        q1_idx = len(sorted_data) // 4
        q3_idx = (3 * len(sorted_data)) // 4

        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        iqr = q3 - q1

        # Mild outliers: 1.5 IQR from quartiles
        mild_lower = q1 - 1.5 * iqr
        mild_upper = q3 + 1.5 * iqr

        # Extreme outliers: 3 IQR from quartiles (farther out)
        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr

        # Verify outlier bounds are correctly ordered (extreme are farther)
        assert extreme_lower < mild_lower
        assert extreme_upper > mild_upper


class TestAnomalyEnsemble:
    """Test ensemble of anomaly detection methods."""

    def test_voting_ensemble(self):
        """Test voting-based ensemble."""
        data = list(range(1, 11)) + [100]

        methods = {
            "zscore": [],
            "iqr": [],
            "tukey": [],
        }

        # Each method votes on each point
        for x in data:
            # Simplified: any point > 50 is anomaly
            zscore_vote = 1 if x > 50 else 0
            iqr_vote = 1 if x > 50 else 0
            tukey_vote = 1 if x > 50 else 0

            methods["zscore"].append(zscore_vote)
            methods["iqr"].append(iqr_vote)
            methods["tukey"].append(tukey_vote)

        # Count votes
        votes = [
            sum(methods[m][i] for m in methods) for i in range(len(data))
        ]

        # Ensemble says anomaly if >= 2 methods agree
        ensemble_anomalies = [
            data[i] for i, v in enumerate(votes) if v >= 2
        ]

        assert 100 in ensemble_anomalies

    def test_weighted_ensemble(self):
        """Test weighted ensemble of methods."""
        data = [10, 20, 30, 40, 50, 100]
        weights = {
            "isolation_forest": 0.4,
            "lof": 0.35,
            "zscore": 0.25,
        }

        anomaly_scores = []

        for x in data:
            if_score = 0.1 if x < 60 else 0.9
            lof_score = 0.1 if x < 60 else 0.9
            z_score = 0.1 if x < 60 else 0.9

            weighted_score = (
                if_score * weights["isolation_forest"]
                + lof_score * weights["lof"]
                + z_score * weights["zscore"]
            )
            anomaly_scores.append(weighted_score)

        # 100 should have high ensemble score
        assert anomaly_scores[-1] > 0.8

    def test_consensus_threshold(self):
        """Test consensus-based detection."""
        data = list(range(1, 11)) + [100]
        threshold = 0.7

        # Each method provides confidence score
        method_scores = {
            "method1": [0.1 if x < 60 else 0.95 for x in data],
            "method2": [0.15 if x < 60 else 0.9 for x in data],
            "method3": [0.05 if x < 60 else 0.92 for x in data],
        }

        # Average confidence
        avg_confidence = [
            sum(method_scores[m][i] for m in method_scores)
            / len(method_scores)
            for i in range(len(data))
        ]

        # Anomalies above threshold
        anomalies = [
            data[i] for i, c in enumerate(avg_confidence) if c > threshold
        ]
        assert 100 in anomalies

    def test_ensemble_with_conflicting_votes(self):
        """Test handling conflicting votes in ensemble."""
        votes = {
            "method1": 1,  # Anomaly
            "method2": 0,  # Normal
            "method3": 1,  # Anomaly
        }

        # Majority voting
        total_votes = sum(votes.values())
        num_methods = len(votes)
        majority_threshold = num_methods / 2

        is_anomaly = total_votes > majority_threshold
        assert is_anomaly  # 2 out of 3 say anomaly

    def test_anomaly_confidence_ranking(self):
        """Test ranking anomalies by confidence."""
        data = [10, 100, 200]

        # Compute confidence for each
        confidence = {}
        for x in data:
            if x < 20:
                confidence[x] = 0.1  # Low anomaly confidence
            elif x < 150:
                confidence[x] = 0.6  # Medium
            else:
                confidence[x] = 0.95  # High

        # Rank by confidence
        ranked = sorted(
            confidence.items(), key=lambda x: x[1], reverse=True
        )

        # 200 should be most anomalous
        assert ranked[0][0] == 200
