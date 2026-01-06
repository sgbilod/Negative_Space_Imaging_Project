#!/usr/bin/env python
"""
Federated Learning Simulation Suite
Comprehensive testing with metrics collection and comparison.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt

from federated.simulation import FederationSimulator
from federated.healthcare_astronomy_setup import HealthcareAstronomySimulation

logger = logging.getLogger(__name__)


class FederatedLearningBenchmark:
    """Comprehensive benchmark suite for federated learning."""

    def __init__(self, output_dir: str = "./simulation_results"):
        """
        Initialize benchmark.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.results: Dict[str, Any] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def benchmark_healthcare_astronomy(
        self,
        num_hospitals: int = 3,
        num_observatories: int = 2,
        num_rounds: int = 20,
    ) -> Dict[str, Any]:
        """
        Benchmark healthcare-astronomy federated learning.

        Args:
            num_hospitals: Number of hospitals
            num_observatories: Number of observatories
            num_rounds: Number of federation rounds

        Returns:
            Benchmark results
        """
        logger.info("Starting healthcare-astronomy benchmark...")

        sim = HealthcareAstronomySimulation(
            num_hospitals=num_hospitals,
            num_observatories=num_observatories,
        )

        sim.setup_institutions()
        sim.create_datasets()
        sim.create_clients()

        results = sim.run_simulation(num_rounds=num_rounds)
        evaluation = sim.evaluate_use_case()
        summary = sim.get_multi_institutional_summary()

        self.results["healthcare_astronomy"] = {
            "config": {
                "num_hospitals": num_hospitals,
                "num_observatories": num_observatories,
                "num_rounds": num_rounds,
            },
            "results": results,
            "evaluation": evaluation,
            "summary": summary,
        }

        return self.results["healthcare_astronomy"]

    def benchmark_privacy_utility_tradeoff(
        self,
        num_clients: int = 5,
        num_rounds: int = 10,
        epsilon_values: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark privacy-utility tradeoff.

        Args:
            num_clients: Number of clients
            num_rounds: Number of rounds per epsilon value
            epsilon_values: Privacy budgets to test

        Returns:
            Tradeoff analysis
        """
        if epsilon_values is None:
            epsilon_values = [0.5, 1.0, 2.0, 5.0]

        logger.info(f"Benchmarking privacy-utility tradeoff...")

        tradeoff_results = {}

        for epsilon in epsilon_values:
            logger.info(f"Testing epsilon={epsilon}...")

            sim = FederationSimulator(
                num_clients=num_clients,
                device="cpu",
            )

            # Configure clients for privacy testing
            from federated.simulation import ClientSimulationConfig
            client_configs = [
                ClientSimulationConfig(
                    client_id=f"client_{i}",
                    num_samples=100 + i * 20,
                    iid_level=0.5,
                    data_heterogeneity=1.0,
                    enable_dp=True,
                    dp_epsilon=epsilon,
                )
                for i in range(num_clients)
            ]

            sim.create_clients(client_configs)
            sim.run_simulation(num_rounds=num_rounds)
            results = sim.get_simulation_summary()

            tradeoff_results[f"epsilon_{epsilon}"] = {
                "epsilon": epsilon,
                "accuracy": results.get("final_accuracy", 0),
                "privacy_epsilon": results.get("min_epsilon_reached", epsilon),
                "communication": results.get("total_communication", 0),
            }

        self.results["privacy_utility"] = tradeoff_results

        return tradeoff_results

    def benchmark_scalability(
        self,
        client_counts: List[int] = None,
        num_rounds: int = 5,
    ) -> Dict[str, Any]:
        """
        Benchmark system scalability with varying client counts.

        Args:
            client_counts: Number of clients to test
            num_rounds: Federation rounds per test

        Returns:
            Scalability analysis
        """
        if client_counts is None:
            client_counts = [5, 10, 20]

        logger.info("Benchmarking scalability...")

        scalability_results = {}

        for num_clients in client_counts:
            logger.info(f"Testing {num_clients} clients...")

            sim = FederationSimulator(
                num_clients=num_clients,
                device="cpu",
            )

            from federated.simulation import ClientSimulationConfig
            client_configs = [
                ClientSimulationConfig(
                    client_id=f"client_{i}",
                    num_samples=50 + i * 10,
                    iid_level=0.4,
                    data_heterogeneity=1.0,
                    straggler_probability=0.1,
                    dropout_probability=0.05,
                )
                for i in range(num_clients)
            ]

            sim.create_clients(client_configs)
            sim.run_simulation(num_rounds=num_rounds)
            results = sim.get_simulation_summary()

            scalability_results[f"clients_{num_clients}"] = {
                "num_clients": num_clients,
                "final_accuracy": results.get("final_accuracy", 0),
                "total_communication": results.get("total_communication", 0),
                "communication_per_client": results.get("total_communication", 0) / (
                    num_clients + 1e-10
                ),
            }

        self.results["scalability"] = scalability_results

        return scalability_results

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.

        Returns:
            Report summary
        """
        logger.info("Generating report...")

        report = {
            "timestamp": self.timestamp,
            "benchmarks": self.results,
            "summary": self._generate_summary(),
        }

        # Save report to JSON
        report_file = f"{self.output_dir}/benchmark_report_{self.timestamp}.json"
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_file}")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")

        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all benchmarks."""
        summary = {}

        if "healthcare_astronomy" in self.results:
            ha_results = self.results["healthcare_astronomy"]
            summary["healthcare_astronomy"] = {
                "final_accuracy": ha_results["evaluation"]["overall"]["final_accuracy"],
                "converged": ha_results["evaluation"]["convergence"]["converged"],
                "avg_epsilon": ha_results["evaluation"]["privacy"]["avg_epsilon"],
            }

        if "privacy_utility" in self.results:
            pu_results = self.results["privacy_utility"]
            accuracies = [r["accuracy"] for r in pu_results.values()]
            summary["privacy_utility"] = {
                "avg_accuracy": np.mean(accuracies) if accuracies else 0,
                "accuracy_range": (
                    f"{np.min(accuracies):.3f}-{np.max(accuracies):.3f}"
                    if accuracies else "N/A"
                ),
            }

        if "scalability" in self.results:
            scale_results = self.results["scalability"]
            accuracies = [r["final_accuracy"] for r in scale_results.values()]
            communications = [r["total_communication"] for r in scale_results.values()]
            summary["scalability"] = {
                "avg_accuracy": np.mean(accuracies) if accuracies else 0,
                "total_communication_mb": sum(communications) / (1024 * 1024) if communications else 0,
            }

        return summary


def run_comprehensive_simulation():
    """Run comprehensive federated learning simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 80)
    logger.info("FEDERATED LEARNING COMPREHENSIVE SIMULATION")
    logger.info("=" * 80)

    benchmark = FederatedLearningBenchmark()

    # 1. Healthcare-Astronomy Use Case
    logger.info("\n1. HEALTHCARE-ASTRONOMY FEDERATED LEARNING")
    logger.info("-" * 80)
    ha_results = benchmark.benchmark_healthcare_astronomy(
        num_hospitals=3,
        num_observatories=2,
        num_rounds=15,
    )
    logger.info(f"Healthcare-Astronomy Results: {ha_results['evaluation']['overall']}")

    # 2. Privacy-Utility Tradeoff
    logger.info("\n2. PRIVACY-UTILITY TRADEOFF ANALYSIS")
    logger.info("-" * 80)
    pu_results = benchmark.benchmark_privacy_utility_tradeoff(
        num_clients=5,
        num_rounds=10,
        epsilon_values=[0.5, 1.0, 2.0, 5.0],
    )
    logger.info(f"Privacy-Utility Results:")
    for epsilon_key, results in pu_results.items():
        logger.info(f"  {epsilon_key}: Accuracy={results['accuracy']:.3f}, "
                   f"Communication={results['communication']/1024:.1f}KB")

    # 3. Scalability Analysis
    logger.info("\n3. SCALABILITY ANALYSIS")
    logger.info("-" * 80)
    scale_results = benchmark.benchmark_scalability(
        client_counts=[5, 10, 20],
        num_rounds=5,
    )
    logger.info(f"Scalability Results:")
    for clients_key, results in scale_results.items():
        logger.info(f"  {clients_key}: Accuracy={results['final_accuracy']:.3f}, "
                   f"Comm/Client={results['communication_per_client']/1024:.1f}KB")

    # 4. Generate Report
    logger.info("\n4. GENERATING REPORT")
    logger.info("-" * 80)
    report = benchmark.generate_report()

    logger.info("\n" + "=" * 80)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Report Summary: {report['summary']}")

    return report


if __name__ == "__main__":
    run_comprehensive_simulation()
