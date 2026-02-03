import sys
import os
sys.path.insert(0, os.path.abspath(f'{os.getcwd()}/utils/'))
import utils.helper as helper
import utils.driver as driver

from metrics import vpt_time, component_sizes, fraction_driving
from file_io import create_rescomp_datasets_template, generate_rescomp_means
from helper import get_network

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "rescomp", "rescomp")))
import ResComp

"""
Main Method to call for Structural Analysis
"""

def drive_structural_analysis(
        tol,
        t_train,
        t_test,
        U_train,
        U_test,
        rho,
        p_thin,
        param_set
    ):
    """Inner for loop work here - run a single reservoir and perform analysis"""

    print("param_set:", param_set)

    n, erdos_c, gamma, sigma, alpha = param_set

    # Template for datasets
    datasets = create_rescomp_datasets_template(attributes=['vpt', 'component_size', 'fraction_driving'])

    # Generate thinned networks
    mean_degree = erdos_c*(1-p_thin)
    if mean_degree < 0.0:
        mean_degree = 0.0
    
    res_thinned = ResComp.ResComp(res_sz=n, mean_degree=mean_degree, 
                                ridge_alpha=alpha, spect_rad=rho, sigma=sigma, 
                                gamma=gamma, map_initial='activ_f')

    adj_matrix = res_thinned.res

    print("Train")       
    res_thinned.train(t_train, U_train)

    print("Forecast and predict")
    U_pred = res_thinned.predict(t_test, r0=res_thinned.r0, return_states=True)[0]
    vpt = vpt_time(t_test, U_test, U_pred, vpt_tol=tol)

    #structural components
    G = get_network(adj_matrix)

    component_dist = component_sizes(G)
    frac_drive = fraction_driving(G)

    datasets['vpt'].append(vpt)
    datasets['component_size'].append(component_dist)
    datasets['fraction_driving'].append(frac_drive)

    mean_attrs = generate_rescomp_means(datasets)

    print("Mean_attrs:", mean_attrs)

    return mean_attrs, datasets

def main():
    rho_p_thin_set, param, param_name, param_set = helper.parse_arguments()

    rho_p_thin_prod, erdos_possible_combinations = helper.generate_params(
        rho_p_thin_set,
        param=param, 
        param_name=param_name,
        param_set=param_set
    )

    n, _ = rho_p_thin_prod.shape

    if n == 1:
        rho, p_thin = rho_p_thin_prod[0]
    else:
        job_id_number = int(os.getenv('ID_TO_PROCESS'))
        print(job_id_number)
        rho, p_thin = rho_p_thin_prod[job_id_number]
        
    home = os.path.expanduser("~")
    results_path = f'{home}/nobackup/autodelete/results/{param_name}/{param}/{param_set}/{rho_p_thin_set}/'

    driver.rescomp_parallel_uniform_gridsearch_h5(
        erdos_possible_combinations, 
        rho,
        p_thin,
        draw_count=100000, 
        hdf5_file_path=results_path, 
        tf=7200,
        structural_analysis=True
    )
    


if __name__ == "__main__":
    main()