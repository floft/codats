"""
Help speed up preprocessing of some datasets by running jobs on multiple cores
"""
import tqdm
import multiprocessing


def run_job_pool(func, argsList, desc=None, cores=None):
    """
    Processor pool to use multiple cores, with a progress bar

    func = function to execute
    argsList = array of tuples, each tuple is the arguments to pass to the function

    Combination of:
    https://stackoverflow.com/a/43714721/2698494
    https://stackoverflow.com/a/45652769/2698494

    Returns:
    an array of the outputs from the function

    Example:
    # Define a function that'll be run a bunch of times
    def f(a,b):
        return a+b

    # Array of arrays (or tuples) of the arguments for the function
    commands = [[1,2],[3,4],[5,6],[7,8]]
    results = run_job_pool(f, commands, desc="Addition")
    """
    if cores is None:
        p = multiprocessing.Pool(multiprocessing.cpu_count())
    else:
        p = multiprocessing.Pool(cores)
    processes = []
    results = []

    for args in argsList:
        processes.append(p.apply_async(func, args))

    with tqdm.tqdm(total=len(processes), desc=desc) as pbar:
        for process in processes:
            results.append(process.get())
            pbar.update()
    pbar.close()
    p.close()
    p.join()

    return results
