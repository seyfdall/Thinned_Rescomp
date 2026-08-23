from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import importlib

import numpy as np
import sys

from metrics import (
    calculate_diameters,
    calculate_diameters_weakly_connected,
    consistency_analysis_pearson,
    div_metric_tests,
    vpt_time,
)
from file_io import (
    generate_rescomp_means,
    load_parameters,
    update_datasets,
    get_bundle_dir,
    get_named_bundle_dir,
    save_exemplar_bundle,
)
from helper import create_network, get_orbit, remove_edges, scale_spect_rad


def _get_rescomp_module():
    """Import the local ResComp module from the repository sibling directory."""
    rescomp_path = Path(__file__).resolve().parents[2] / "rescomp" / "rescomp"
    if str(rescomp_path) not in sys.path:
        sys.path.insert(0, str(rescomp_path))
    return importlib.import_module("ResComp")


class ArtifactLevel(str, Enum):
    METRICS_ONLY = "metrics_only"
    PREDICTION = "prediction"
    FULL_STATES = "full_states"


@dataclass
class ReservoirRunResult:
    mean_attrs: Dict[str, float]
    datasets: Dict[str, list]
    artifacts: Dict[str, Any]


@dataclass
class ThinningRecoverySearchResult:
    best_result: Optional[ReservoirRunResult]
    summary: Dict[str, Any]


def _extract_vpt_scalar(run_result: ReservoirRunResult) -> float:
    """Extract a scalar vpt value from run result datasets."""
    vpt_values = run_result.datasets.get("vpt", [])
    if not vpt_values:
        return 0.0
    return float(vpt_values[0])


def _evaluate_reservoir_on_network(
    A,
    tol: float,
    t_train,
    t_test,
    U_train,
    U_test,
    network_type: str,
    rho: float,
    mean_degree: float,
    gamma: float,
    sigma: float,
    alpha: float,
    artifact_level: ArtifactLevel = ArtifactLevel.METRICS_ONLY,
) -> ReservoirRunResult:
    """Run the reservoir workflow on a supplied adjacency matrix."""

    ResComp = _get_rescomp_module()
    n = A.shape[0]

    res_thinned = ResComp.ResComp(
        A,
        res_sz=n,
        mean_degree=mean_degree,
        ridge_alpha=alpha,
        spect_rad=rho,
        sigma=sigma,
        gamma=gamma,
        map_initial="activ_f",
    )

    print("First Replica Run")
    r0_1 = np.random.uniform(-1.0, 1.0, n)
    states_1 = res_thinned.internal_state_response(t_train, U_train, r0_1)

    print("Second Replica Run")
    r0_2 = np.random.uniform(-1.0, 1.0, n)
    states_2 = res_thinned.internal_state_response(t_train, U_train, r0_2)

    cap = consistency_analysis_pearson(states_1.T, states_2.T)

    print("Train")
    res_thinned.train(t_train, U_train)

    print("Forecast and predict")
    U_pred, states_pred = res_thinned.predict(t_test, r0=res_thinned.r0, return_states=True)
    error = np.linalg.norm(U_test - U_pred, axis=1)
    vpt = vpt_time(t_test, U_test, U_pred, vpt_tol=tol)
    divs = div_metric_tests(res_thinned.states)

    datasets: Dict[str, list] = {}

    if network_type == "undirected_erdos":
        giant_diam, average_diam, giant_size = calculate_diameters(res_thinned.res)
        datasets = update_datasets(
            datasets,
            giant_diam=giant_diam,
            average_diam=average_diam,
            giant_size=giant_size,
        )
    elif network_type == "directed_erdos":
        giant_diam, average_diam, giant_size = calculate_diameters_weakly_connected(res_thinned.res)
        print(f"GIANT SIZE: {giant_size}")
        datasets = update_datasets(
            datasets,
            giant_diam=giant_diam,
            average_diam=average_diam,
            giant_size=giant_size,
        )

    print("Divs:", divs)
    update_datasets(
        datasets,
        div_pos=divs[0],
        div_der=divs[1],
        div_spect=divs[2],
        div_rank=divs[3],
        pred=U_pred,
        err=error,
        vpt=vpt,
        consistency_correlation=cap,
    )

    mean_attrs = generate_rescomp_means(datasets)
    print("Mean_attrs:", mean_attrs)

    artifacts: Dict[str, Any] = {}
    if artifact_level in (ArtifactLevel.PREDICTION, ArtifactLevel.FULL_STATES):
        artifacts.update(
            {
                "U_pred": U_pred,
                "error": error,
                "vpt": vpt,
                "r0": res_thinned.r0,
                "W_out": res_thinned.W_out,
            }
        )
    if artifact_level == ArtifactLevel.FULL_STATES:
        artifacts.update(
            {
                "A": A,
                "states_train": res_thinned.states,
                "states_pred": states_pred,
                "replica_states_1": states_1,
                "replica_states_2": states_2,
                "U_train": U_train,
                "U_test": U_test,
                "t_train": t_train,
                "t_test": t_test,
            }
        )

    return ReservoirRunResult(mean_attrs=mean_attrs, datasets=datasets, artifacts=artifacts)


def _thin_base_network(A_base, thin_fraction: float):
    """Remove a fraction of edges from a base network without mutating it."""
    if thin_fraction <= 0.0:
        return A_base.copy().tocsr()

    total_edges = int(getattr(A_base, "nnz", 0))
    if total_edges <= 1:
        return A_base.copy().tocsr()

    n_edges_to_remove = int(round(total_edges * thin_fraction))
    n_edges_to_remove = max(1, min(n_edges_to_remove, total_edges - 1))
    return remove_edges(A_base, n_edges_to_remove).tocsr()


def _paper_plots_bundle_dir(parameter_set_name: str) -> Path:
    """Return the paper_plots/data bundle path regardless of the current working directory."""
    return Path(__file__).resolve().parents[1] / "paper_plots" / "data" / f"bundle_{parameter_set_name}"


def run_single_reservoir_analysis(
    tol: float,
    t_train,
    t_test,
    U_train,
    U_test,
    network_type: str,
    rho: float,
    p_thin: float,
    param_set: Tuple[float, float, float, float, float],
    artifact_level: ArtifactLevel = ArtifactLevel.METRICS_ONLY,
) -> ReservoirRunResult:
    """Run one reservoir draw and return aggregated metrics plus optional artifacts."""

    print("param_set:", param_set)

    n, erdos_c, gamma, sigma, alpha = param_set

    mean_degree = erdos_c * (1 - p_thin)
    if mean_degree < 0.0:
        mean_degree = 0.0

    p = mean_degree / n
    A = create_network([n, p], network_type, rho)
    return _evaluate_reservoir_on_network(
        A=A,
        tol=tol,
        t_train=t_train,
        t_test=t_test,
        U_train=U_train,
        U_test=U_test,
        network_type=network_type,
        rho=rho,
        mean_degree=mean_degree,
        gamma=gamma,
        sigma=sigma,
        alpha=alpha,
        artifact_level=artifact_level,
    )


def search_best_reservoir(
    tol: float,
    t_train,
    t_test,
    U_train,
    U_test,
    network_type: str,
    rho: float,
    p_thin: float,
    param_set: Tuple[float, float, float, float, float],
    draw_count: int = 100,
    best_vpt_start: float = 0.0,
    vpt_upper_bound: Optional[float] = 3.5,
    artifact_level: ArtifactLevel = ArtifactLevel.FULL_STATES
) -> Tuple[Optional[ReservoirRunResult], float]:
    """Search multiple random reservoirs and keep the best run by vpt."""

    best_result: Optional[ReservoirRunResult] = None
    best_vpt = float(best_vpt_start)

    for _ in range(draw_count):
        run_result = run_single_reservoir_analysis(
            tol=tol,
            t_train=t_train,
            t_test=t_test,
            U_train=U_train,
            U_test=U_test,
            network_type=network_type,
            rho=rho,
            p_thin=p_thin,
            param_set=param_set,
            artifact_level=artifact_level,
        )

        vpt = _extract_vpt_scalar(run_result)
        if vpt_upper_bound is not None and vpt >= vpt_upper_bound:
            continue

        if vpt > best_vpt:
            best_vpt = vpt
            best_result = run_result

    return best_result, best_vpt


def build_and_save_best_reservoir(
    n,
    network_type,
    rho,
    mean_degree,
    alpha,
    gamma,
    sigma,
    tol,
    duration,
    switch,
    draw_count=100,
    vpt_upper_bound=3.5,
    skip_if_bundle_exists=False,
    override=False,
    parameter_set_name=None,
):
    """Search for the best reservoir and persist it as an exemplar bundle."""
    if parameter_set_name is not None:
        bundle_dir = get_named_bundle_dir(parameter_set_name)
    else:
        bundle_dir = get_bundle_dir(
            n, network_type, rho, mean_degree, alpha, gamma, sigma, tol, duration, switch
        )

    if skip_if_bundle_exists and (bundle_dir / "vpt.npy").is_file():
        print(f"Bundle already exists at {bundle_dir}; skipping reservoir search.")
        return

    t_train, U_train, t_test, U_test = get_orbit(duration=duration, system='lorenz', switch=switch)

    # For the notebook flow, erdos_c equals the pre-thinning mean degree.
    erdos_c = mean_degree
    p_thin = 0.0
    param_set = (n, erdos_c, gamma, sigma, alpha)

    best_vpt_start = 0.0
    existing_vpt_path = bundle_dir / "vpt.npy"
    if existing_vpt_path.is_file() and not override:
        best_vpt_start = float(np.load(existing_vpt_path))

    best_result, best_vpt = search_best_reservoir(
        tol=tol,
        t_train=t_train,
        t_test=t_test,
        U_train=U_train,
        U_test=U_test,
        network_type=network_type,
        rho=rho,
        p_thin=p_thin,
        param_set=param_set,
        draw_count=draw_count,
        best_vpt_start=best_vpt_start,
        vpt_upper_bound=vpt_upper_bound,
        artifact_level=ArtifactLevel.FULL_STATES
    )

    if best_result is None:
        print(f"No improved reservoir found. Current best vpt={best_vpt_start:.4f}")
        return

    save_exemplar_bundle(
        bundle_dir=bundle_dir,
        artifacts=best_result.artifacts,
        mean_attrs=best_result.mean_attrs,
        datasets=best_result.datasets,
        include_datasets=False,
    )

    print(f"Saved bundle to {bundle_dir}")
    print(f"Best vpt: {best_vpt:.4f}")


def search_thinning_recovery_reservoir(
    tol: float,
    t_train,
    t_test,
    U_train,
    U_test,
    network_type: str,
    low_rhos,
    high_rhos,
    thin_levels,
    param_set: Tuple[float, float, float, float, float],
    draw_count: int = 200,
    low_vpt_min: float = 2.0,
    high_vpt_max: float = 1.0,
    recovery_vpt_min: float = 2.0,
    artifact_level: ArtifactLevel = ArtifactLevel.FULL_STATES,
) -> ThinningRecoverySearchResult:
    """Search for a network that succeeds at low rho, fails at high rho, and recovers after thinning."""

    n, erdos_c, gamma, sigma, alpha = param_set
    p = erdos_c / n

    best_result: Optional[ReservoirRunResult] = None
    best_summary: Dict[str, Any] = {}
    best_score = (-np.inf, -np.inf, np.inf)

    for draw_idx in range(draw_count):
        print(f"Thinning recovery draw {draw_idx + 1}/{draw_count}")
        A_base = create_network([n, p], network_type, 1.0)

        low_results = []
        for rho in low_rhos:
            A_low = scale_spect_rad(A_base.copy(), rho)
            run_result = _evaluate_reservoir_on_network(
                A=A_low,
                tol=tol,
                t_train=t_train,
                t_test=t_test,
                U_train=U_train,
                U_test=U_test,
                network_type=network_type,
                rho=rho,
                mean_degree=erdos_c,
                gamma=gamma,
                sigma=sigma,
                alpha=alpha,
                artifact_level=ArtifactLevel.METRICS_ONLY,
            )
            low_results.append({"rho": float(rho), "vpt": _extract_vpt_scalar(run_result), "A": A_low})

        high_results = []
        for rho in high_rhos:
            A_high = scale_spect_rad(A_base.copy(), rho)
            run_result = _evaluate_reservoir_on_network(
                A=A_high,
                tol=tol,
                t_train=t_train,
                t_test=t_test,
                U_train=U_train,
                U_test=U_test,
                network_type=network_type,
                rho=rho,
                mean_degree=erdos_c,
                gamma=gamma,
                sigma=sigma,
                alpha=alpha,
                artifact_level=ArtifactLevel.METRICS_ONLY,
            )
            high_results.append({"rho": float(rho), "vpt": _extract_vpt_scalar(run_result), "A": A_high})

            recovery_results = []
            for thin_level in thin_levels:
                A_thin_base = _thin_base_network(A_base, thin_level)
                thinned_mean_degree = max(erdos_c * (1.0 - thin_level), 0.0)
                A_recovery = scale_spect_rad(A_thin_base.copy(), rho)
                run_result = _evaluate_reservoir_on_network(
                    A=A_recovery,
                    tol=tol,
                    t_train=t_train,
                    t_test=t_test,
                    U_train=U_train,
                    U_test=U_test,
                    network_type=network_type,
                    rho=rho,
                    mean_degree=thinned_mean_degree,
                    gamma=gamma,
                    sigma=sigma,
                    alpha=alpha,
                    artifact_level=ArtifactLevel.METRICS_ONLY,
                )
                recovery_results.append(
                    {
                        "rho": float(rho),
                        "thin_level": float(thin_level),
                        "vpt": _extract_vpt_scalar(run_result),
                        "A": A_recovery,
                    }
                )

        best_low = max(low_results, key=lambda item: item["vpt"])
        worst_high = max(high_results, key=lambda item: item["vpt"])
        best_recovery = max(recovery_results, key=lambda item: item["vpt"])

        print()
        print(f"best_low: {best_low}")
        print(f"worst_high: {worst_high}")
        print(f"best_recovery: {best_recovery}")
        print()

        meets_criteria = (
            best_low["vpt"] >= low_vpt_min
            and best_low["vpt"] <= low_vpt_min + 3.0
            and worst_high["vpt"] <= high_vpt_max
            and best_recovery["vpt"] >= recovery_vpt_min
            and best_recovery["vpt"] <= recovery_vpt_min + 4.0
        )

        if not meets_criteria:
            print(f"Does not meet criteria")
            continue

        score = (best_low["vpt"], best_recovery["vpt"], -worst_high["vpt"])
        # print(f"Score: {score}")
        # if score <= best_score:
        #     continue

        best_recovery_result = _evaluate_reservoir_on_network(
            A=best_recovery["A"],
            tol=tol,
            t_train=t_train,
            t_test=t_test,
            U_train=U_train,
            U_test=U_test,
            network_type=network_type,
            rho=best_recovery["rho"],
            mean_degree=max(erdos_c * (1.0 - best_recovery["thin_level"]), 0.0),
            gamma=gamma,
            sigma=sigma,
            alpha=alpha,
            artifact_level=artifact_level,
        )
        best_recovery_result.artifacts.update(
            {
                "A_base": A_base,
                "A_low": best_low["A"],
                "A_high": worst_high["A"],
                "selected_low_rho": best_low["rho"],
                "selected_high_rho": worst_high["rho"],
                "selected_recovery_rho": best_recovery["rho"],
                "selected_recovery_thin_level": best_recovery["thin_level"],
            }
        )

        best_summary = {
            "draw_index": draw_idx,
            "criteria_met": True,
            "criteria": {
                "low_vpt_min": low_vpt_min,
                "high_vpt_max": high_vpt_max,
                "recovery_vpt_min": recovery_vpt_min,
                "low_rhos": [float(rho) for rho in low_rhos],
                "high_rhos": [float(rho) for rho in high_rhos],
                "thin_levels": [float(thin_level) for thin_level in thin_levels],
            },
            "low_results": [{"rho": item["rho"], "vpt": item["vpt"]} for item in low_results],
            "high_results": [{"rho": item["rho"], "vpt": item["vpt"]} for item in high_results],
            "recovery_results": [
                {"rho": item["rho"], "thin_level": item["thin_level"], "vpt": item["vpt"]}
                for item in recovery_results
            ],
            "selected_low": {"rho": best_low["rho"], "vpt": best_low["vpt"]},
            "selected_high": {"rho": worst_high["rho"], "vpt": worst_high["vpt"]},
            "selected_recovery": {
                "rho": best_recovery["rho"],
                "thin_level": best_recovery["thin_level"],
                "vpt": best_recovery["vpt"],
            },
            "score": [score[0], score[1], score[2]],
        }
        best_result = best_recovery_result
        best_score = score

        break

    return ThinningRecoverySearchResult(best_result=best_result, summary=best_summary)


def build_and_save_thinning_recovery_reservoir(
    parameter_set_name: str,
    parameters_file=None,
    draw_count: Optional[int] = None,
    skip_if_bundle_exists: bool = True,
    override: bool = False,
) -> None:
    """Search for a thinning-recovery network and persist the winning bundle in paper_plots/data."""

    params = load_parameters(parameter_set_name, parameters_file=parameters_file)
    bundle_dir = _paper_plots_bundle_dir(parameter_set_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if skip_if_bundle_exists and (bundle_dir / "A.npy").is_file() and not override:
        print(f"Bundle already exists at {bundle_dir}; skipping thinning recovery search.")
        return

    t_train, U_train, t_test, U_test = get_orbit(
        duration=params["duration"],
        system="lorenz",
        switch=params["switch"],
    )

    search_draw_count = draw_count if draw_count is not None else params.get("draw_count", 200)
    search_result = search_thinning_recovery_reservoir(
        tol=params["tol"],
        t_train=t_train,
        t_test=t_test,
        U_train=U_train,
        U_test=U_test,
        network_type=params["network_type"],
        low_rhos=params["low_rhos"],
        high_rhos=params["high_rhos"],
        thin_levels=params["thin_levels"],
        param_set=(params["n"], params["mean_degree"], params["gamma"], params["sigma"], params["alpha"]),
        draw_count=search_draw_count,
        low_vpt_min=params["low_vpt_min"],
        high_vpt_max=params["high_vpt_max"],
        recovery_vpt_min=params["recovery_vpt_min"],
        artifact_level=ArtifactLevel.FULL_STATES,
    )

    summary_payload = {
        "parameter_set_name": parameter_set_name,
        "parameters": params,
        "best_summary": search_result.summary,
        "best_vpt": _extract_vpt_scalar(search_result.best_result) if search_result.best_result else None,
    }

    with open(bundle_dir / "search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    if search_result.best_result is None:
        print("No network satisfied the thinning recovery criteria.")
        return

    save_exemplar_bundle(
        bundle_dir=bundle_dir,
        artifacts=search_result.best_result.artifacts,
        mean_attrs=search_result.best_result.mean_attrs,
        datasets=search_result.best_result.datasets,
        include_datasets=False,
    )

    print(f"Saved thinning recovery bundle to {bundle_dir}")
    print(f"Best vpt: {_extract_vpt_scalar(search_result.best_result):.4f}")
