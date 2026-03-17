from dataclasses import dataclass
from enum import Enum
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
    update_datasets,
    get_bundle_dir,
    get_named_bundle_dir,
    save_exemplar_bundle,
)
from helper import create_network, get_orbit


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


def _extract_vpt_scalar(run_result: ReservoirRunResult) -> float:
    """Extract a scalar vpt value from run result datasets."""
    vpt_values = run_result.datasets.get("vpt", [])
    if not vpt_values:
        return 0.0
    return float(vpt_values[0])


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

    ResComp = _get_rescomp_module()

    print("param_set:", param_set)

    n, erdos_c, gamma, sigma, alpha = param_set

    mean_degree = erdos_c * (1 - p_thin)
    if mean_degree < 0.0:
        mean_degree = 0.0

    p = mean_degree / n
    A = create_network([n, p], network_type, rho)

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
