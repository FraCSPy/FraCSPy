import os


def write_SOFIrunScript(sofimaster, num_procs, sofi_param_file, outfile, write=True):
    ''' creates the string for the

    Parameters
    ----------
    sofimaster: path to the binary for sofi master
    num_procs: number of processes to be parallelised over
    sofi_param_file: json parameter file for the experiment

    Returns
    -------
    sh_script: text that is saved into outfile

    '''
    sh_script = '''#!/bin/bash

mkdir -p outputs
mkdir -p outputs/log
mkdir -p outputs/snap
mkdir -p outputs/su

sofipath=%s
mpirun -np %i ${sofipath} %s > sofi3D.jout

# Do the snap merge thing
snapmergepath=%s
${snapmergepath} %s

# Clean up the snap files (for memory purposes)
rm -rf ./outputs/snap/**.0*
rm -rf ./outputs/snap/**.1*

# Clean up the distributed models
rm -rf ./inputs/models/**.SOFI3D.**
''' % (os.path.join(sofimaster, 'sofi3D'),
       num_procs,
       sofi_param_file,
       os.path.join(sofimaster, 'snapmerge'),
       sofi_param_file,
       )



    # WRITE TO SHELL SCRIPT FOR FAST EXECUTION
    if write:
        text_file = open(outfile, "wt")
        text_file.writelines(sh_script)
        text_file.close()

    return sh_script


def write_SOFIrunScript_NoSnaps(sofimaster, num_procs, sofi_param_file, outfile, write=True):
    ''' creates the string for the

    Parameters
    ----------
    sofimaster: path to the binary for sofi master
    num_procs: number of processes to be parallelised over
    sofi_param_file: json parameter file for the experiment

    Returns
    -------
    sh_script: text that is saved into outfile

    '''
    sh_script = '''#!/bin/bash

mkdir -p outputs
mkdir -p outputs/log
mkdir -p outputs/snap
mkdir -p outputs/su

sofipath=%s
mpirun -np %i ${sofipath} %s > sofi3D.jout

''' % (os.path.join(sofimaster, 'sofi3D'),
       num_procs,
       sofi_param_file,
       )

    # WRITE TO SHELL SCRIPT FOR FAST EXECUTION
    if write:
        text_file = open(outfile, "wt")
        text_file.writelines(sh_script)
        text_file.close()

    return sh_script


def write_SOFIjsonParams(default_strs, monitor_strs,  t_str, mod_strs, src_str, bndry_str, outfile, write=True):

    pre_blurb = '''#-----------------------------------------------------------------
#      JSON PARAMETER FILE FOR SOFI3D
#-----------------------------------------------------------------
# description: example of json input file
# description/name of the model: homogeneous full space (hh.c)
#
'''

    string_list = [pre_blurb, '{',
                   default_strs,
                   monitor_strs,
                   t_str,
                   mod_strs,
                   src_str,
                   bndry_str,
                   '}']

    SOFIstring = '\n \n'.join(string_list)

    # WRITE TO JSON FILE
    if write:
        text_file = open(outfile, "wt")
        text_file.writelines(SOFIstring)
        text_file.close()

    return SOFIstring


def get_default_strings(rec_file='./inputs/receiver.dat'):
    chkpt_str = '''"Checkpoints" : "comment",
    "CHECKPTREAD" : "0",
    "CHECKPTWRITE" : "0",
    "CHECKPT_FILE" : "tmp/checkpoint_sofi3D",'''



    fdorder_str = '''"FD order" : "comment",
    "FDORDER" : "4",
    "FDORDER_TIME" : "2",
    "FDCOEFF" : "2",
    "fdcoeff values: Taylor=1, Holberg=2" : "comment",'''

    q_str = '''"Q-approximation" : "comment",
    "L" : "0",
    "FREF" : "5.0",
    "FL1" : "5.0",
    "TAU" : "0.00001",'''

    rec_str = '''"Receiver" : "comment",
    "SEISMO" : "4",
    "READREC" : "1",
    "REC_FILE" : "%s",
    "REFRECX, REFRECY, REFRECZ" : "0.0 , 0.0 , 0.0",
    "NGEOPH" : "1",
    "REC_ARRAY" : "0", # No array as it is read from the rec file'''%rec_file



    default_strs = '\n \n'.join([chkpt_str,
                                 fdorder_str,
                                 q_str,
                                 rec_str,
                                 ])

    return default_strs


def get_boundary_str(fs=True, npad=30, cpml=True, vppml=3500):
    if fs: fsnum=1
    else: fsnum=0
    if cpml: abstype=1
    else: abstype=2

    bndry_str = '''"Boundary Conditions" : "comment",
    "FREE_SURF" : "%i",
    "ABS_TYPE" : "%i",
    "FW" : "%i",
    "DAMPING" : "4.0",
    "FPML" : "20.0",
    "VPPML" : "%.1f",
    "NPOWER" : "4.0",
    "K_MAX_CPML" : "1.0",
    "BOUNDARY" : "0",'''%(fsnum, abstype, npad, vppml)

    return bndry_str


def get_monitor_str(tsnap_params, smgrm_dtfn, expname, sbsmp_xyz=4, snap=True):
    if snap:
        snap_str = f'''"Snapshots" : "comment",
        "SNAP" : "4",
        "TSNAP1" : "%.2e",
        "TSNAP2" : "%.2e",
        "TSNAPINC" : "%.2e",
        "IDX" : "{sbsmp_xyz}",
        "IDY" : "{sbsmp_xyz}",
        "IDZ" : "{sbsmp_xyz}",
        "SNAP_FORMAT" : "3",
        "SNAP_FILE" : "./outputs/snap/%s",
        "SNAP_PLANE" : "2",''' % (tsnap_params[0],
                                  tsnap_params[1],
                                  tsnap_params[2],
                                  expname)
    else:
        snap_str = f'''"Snapshots" : "comment",
        "SNAP" : "0",
        "TSNAP1" : "%.2e",
        "TSNAP2" : "%.2e",
        "TSNAPINC" : "%.2e",
        "IDX" : "{sbsmp_xyz}",
        "IDY" : "{sbsmp_xyz}",
        "IDZ" : "{sbsmp_xyz}",
        "SNAP_FORMAT" : "3",
        "SNAP_FILE" : "./outputs/snap/%s",
        "SNAP_PLANE" : "2",''' % (tsnap_params[0],
                                  tsnap_params[1],
                                  tsnap_params[2],
                                  expname)

    smgrm_str = '''"Seismograms" : "comment",
    "NDT, NDTSHIFT" : "%i, 0",
    "SEIS_FORMAT" : "2",
    "SEIS_FILE" : "./outputs/su/%s",''' % (smgrm_dtfn, expname)

    log_str = '''"Monitoring the simulation" : "comment",
    "LOG_FILE" : "./outputs/log/%s.log",
    "LOG" : "1",
    "OUT_SOURCE_WAVELET" : "1",
    "OUT_TIMESTEP_INFO" : "10",
    ''' % (expname)

    monitor_strs = '\n \n'.join([snap_str, smgrm_str, log_str])

    return monitor_strs


def get_time_str(dt, tdur):
    t_str = '''"Time Stepping" : "comment",
    "TIME" : "%.3f",
    "DT" : "%.2e",''' % (tdur, dt)

    return t_str


def get_subsurfmod_str(n_xzy,
                       d_xzy,
                       expname,
                       moddir="./inputs/model/",
                       np_xzy=[2,2,2]):
    mod_str1 = '''"3-D Grid" : "comment",
    "NX" : "%i",
    "NY" : "%i",
    "NZ" : "%i",
    "DX" : "%.4f",  # meters
    "DY" : "%.4f",  # meters
    "DZ" : "%.4f",  # meters
''' % (n_xzy[0], n_xzy[1], n_xzy[2],
       d_xzy[0], d_xzy[1], d_xzy[2],)

    mod_str2 = '''"Model" : "comment",
    "READMOD" : "1", # Read from file
    "MFILE" : "%s",
    "WRITE_MODELFILES" : "2",''' % (moddir)

    dom_decomp_str = '''"Domain Decomposition" : "comment",
    "NPROCX" : "%i",
    "NPROCY" : "%i",
    "NPROCZ" : "%i",
    '''% (np_xzy[0], np_xzy[1], np_xzy[2])

    mod_strs = '\n \n'.join([mod_str1, mod_str2, dom_decomp_str])
    return mod_strs


def get_source_str(sfile="./inputs/sources.dat", multisource=0):
    src_str = '''"Source" : "comment",
    "SOURCE_SHAPE" : "1",  # Ricker
    "SOURCE_TYPE" : "1",  # Explosive
    "SRCREC" : "1",  # Read from file
    "SOURCE_FILE" : "%s",
    "RUN_MULTIPLE_SHOTS" : "%i",
    "PLANE_WAVE_ANGLE" : "0.0",
    "TS" : "0.05",  # Duration of source-signal
    '''%(sfile, multisource)

    return src_str

def get_MTsource_str(strike, dip, rake, sfile="./inputs/sources.dat"):
    src_str = '''"Source" : "comment",
    "SOURCE_SHAPE" : "1",  # Ricker
    "SOURCE_TYPE" : "6",  # Explosive
    "SRCREC" : "1",  # Read from file
    "AMON" : "1",
    "STR" : "%.2f",  # Strike
    "DIP" : "%.2f",  # Dip
    "RAKE" : "%.2f",  # Rake
    "SOURCE_FILE" : "%s",
    "RUN_MULTIPLE_SHOTS" : "0",
    "PLANE_WAVE_ANGLE" : "0.0",
    "TS" : "0.05",  # Duration of source-signal
    '''%(strike, dip, rake, sfile)

    return src_str