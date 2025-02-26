#!/usr/bin/env python
import os
import numpy as np
from datetime import date

# Import required carputils modules
from carputils import settings
from carputils import tools
from carputils import mesh
from carputils import testing

EXAMPLE_DESCRIPTIVE_NAME = 'Bi-Ventricular Model Based on MedalCare-XL'
EXAMPLE_AUTHOR = 'James McGreivy'
EXAMPLE_DIR = os.path.dirname(__file__)

# Define parameters exposed to the user on the commandline
def parser():
    parser = tools.standard_parser()
    parser.add_argument('--tend',
                        type=float, default=500.0,
                        help='Duration of simulation (ms).')
    parser.add_argument('--stim-strength',
                        type=float, default=-80.0,
                        help='Stimulus strength (mV).')
    parser.add_argument('--mesh-base',
                        type=str, default='./data/instance_001_lowres',
                        help='Directory + basename for path to mesh files.')
    return parser

def jobID(args):
    """
    Generate name of top level output directory.
    """
    today = date.today()
    return '{}_ventricle_model_{}ms_np{}'.format(today.isoformat(), args.tend, args.np)

@tools.carpexample(parser, jobID)
def run(args, job):
    # Define mesh paths based on provided directory
    mesh_base = args.mesh_base
    
    # Get basic command line, including solver options from external .par file
    cmd = tools.carp_cmd(os.path.join(EXAMPLE_DIR, 'simple.par'))
    
    # Override mesh paths from the parameter file
    cmd += ['-meshname', os.path.join(mesh_base, "heart_instance_001_lowres")]
    
    # Override parameter file settings as needed
    cmd += ['-simID', job.ID,
            '-tend', args.tend,
            '-stimulus[0].strength', args.stim_strength,
            '-stimulus[0].vtx_file', os.path.join(mesh_base, "fascicular_stim.vtx")]
    
    # Handle region files for fast endocardium and regular myocardium
    # The regions are already defined in the .elem file as explained
    # We'll use the existing tags where 0=regular myocardium, 1=fast conducting endocardium
        
    # Run simulation
    job.carp(cmd)
    
    # Do visualization
    if args.visualize and not settings.platform.BATCH:
        # Prepare file paths
        geom = os.path.join(job.ID, os.path.basename(mesh_base)+'_i')
        
        # Visualize transmembrane voltage
        data = os.path.join(job.ID, 'vm.igb.gz')
        
        # Alternatively, visualize activation times
        # data = os.path.join(job.ID, 'init_acts_activation-thresh.dat')
        
        # Call meshalyzer with default view
        job.meshalyzer(geom, data)
        
        # You can create a custom view file and use it like this:
        # view = tools.simfile_path(os.path.join(EXAMPLE_DIR, 'ventricle_view.mshz'))
        # job.meshalyzer(geom, data, view)

if __name__ == '__main__':
    run()