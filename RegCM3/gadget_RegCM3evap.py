"""
Determine downscaled reference PM ETr evaporation from the RegCM3 future climate scenario forcing

usage:

    gadget_METDATAevap.py -I inifile [-S start][-E end][-l loglevel]

    -I inifile - ini file with settings which data to get
    -S start - start timestep (default is 1)
    -E end - end timestep default is last one defined in ini file (from date)
    -l loglevel - DEBUG, INFO, WARN, ERROR
"""
# from pcraster import *
import getopt, sys, os
# import pcraster as pcr
import datetime
from numpy import *
import numpy as np
import numexpr as ne
from RegCM3_utils import *
# import scikits.hydroclimpy
# import psutil
import time
from gadgetRegCM3_lib import *
#from memory_profiler import profile

nthreads = 2
start_time = time.time()
ne.set_num_threads(nthreads)


def usage(*args):
    """
    Print usage information

    -  *args: command line arguments given
    """
    sys.stdout = sys.stderr
    for msg in args: print msg
    print __doc__
    sys.exit(0)


def save_as_mapsstack(lat, lon, data, times, directory, prj, prefix="GADGET", oformat="Gtiff"):
    cnt = 0
    if not os.path.exists(directory):
        os.mkdir(directory)
    for a in times:
        mapname = getmapname(cnt, prefix)
        # print "saving map: " + os.path.join(directory,mapname)
        # writeMap(os.path.join(directory,mapname),oformat,lon,lat[::-1],flipud(data[cnt,:,:]),-999.0)
        writeMap(os.path.join(directory, mapname), oformat, lon, lat, data[cnt, :, :], prj, -999.0)
        cnt = cnt + 1


def save_as_gtiff(lat, lon, data, ncnt, directory, prefix, prj, oformat='GTiff'):
    if not os.path.exists(directory):
        os.mkdir(directory)
    mapname = prefix + '.tif'
    # print "saving map: " + os.path.join(directory,mapname)
    writeMap(os.path.join(directory, mapname), oformat, lon, lat[::-1], flipud(data[:, :]), prj, -999.0)


#### MAIN ####

# Setup variable and file names to read in from netCDF files
# load run settings from ini file
# loop over days to downscale variables and calculate resulting PM ref ET

# @profile
def main(argv=None):
    # Set all sorts of defaults.....
    serverroot = 'F:\\StatewideWater\\NLDAS'
    wrrsetroot = '\\NLDAS'

    # available variables with corresponding file names and standard_names as in NC files
    variables = ['TA', 'RHA', 'TAMINA', 'SWI', 'UA', 'VA']
    folders = ['Tavg', 'RH', 'Tmin', 'RadSW', 'WindU', 'WindV']
    # filenames = [os.path.join(fold, 'RegCM3_{}_'.format(var)) for var, fold in zip(variables, folders)]


    filenames = ["RegCM3Monthly_", "RegCM3Monthly_", "RegCM3Monthly_", "RegCM3Monthly_", "RegCM3Monthly_", "RegCM3Monthly_"]
    standard_names = ['TA', 'RHA', 'TAMINA', 'SWI', 'UA', 'VA']
    prefixes = ["Tmean", "uWind", "PSurf", "Qair",
                "Rainf", "SWdown", "Snowf", "vWind"]

    # tempdir

    # defaults in absence of ini file
    startyear = 1980
    endyear = 1980
    startmonth = 1
    endmonth = 12
    latmin = 51.25
    latmax = 51.75
    lonmin = 5.25
    lonmax = 5.75
    startday = 1
    endday = 1
    getDataForVar = True
    calculateEvap = True
    evapMethod = None
    downscaling = None
    resampling = None
    StartStep = 1
    EndStep = 0

    nrcalls = 0
    loglevel = logging.INFO

    if argv is None:
        argv = sys.argv[1:]
        if len(argv) == 0:
            usage()
            exit()
    try:
        opts, args = getopt.getopt(argv, 'I:l:S:E:')
    except getopt.error, msg:
        usage(msg)

    for o, a in opts:
        if o == '-I': inifile = a
        if o == '-E': EndStep = int(a)
        if o == '-S': StartStep = int(a)
        if o == '-l': exec "loglevel = logging." + a

    logger = setlogger("gadget_RegCM3evap.log", "gadget_METDATAevap", level=loglevel)
    # logger, ch = setlogger("e2o_getvar.log","e2o_getvar",level=loglevel)
    logger.info("Reading settings from ini: " + inifile)
    theconf = iniFileSetUp(inifile)

    # Read period from file
    startyear = int(configget(logger, theconf, "selection", "startyear", str(startyear)))
    endyear = int(configget(logger, theconf, "selection", "endyear", str(endyear)))
    endmonth = int(configget(logger, theconf, "selection", "endmonth", str(endmonth)))
    startmonth = int(configget(logger, theconf, "selection", "startmonth", str(startmonth)))
    endday = int(configget(logger, theconf, "selection", "endday", str(endday)))
    startday = int(configget(logger, theconf, "selection", "startday", str(startday)))
    start = datetime.datetime(startyear, startmonth, startday)
    end = datetime.datetime(endyear, endmonth, endday)

    # read remaining settings from in file
    lonmax = float(configget(logger, theconf, "selection", "lonmax", str(lonmax)))
    lonmin = float(configget(logger, theconf, "selection", "lonmin", str(lonmin)))
    latmax = float(configget(logger, theconf, "selection", "latmax", str(latmax)))
    latmin = float(configget(logger, theconf, "selection", "latmin", str(latmin)))
    BB = dict(
        lon=[lonmin, lonmax],
        lat=[latmin, latmax]
    )
    serverroot = configget(logger, theconf, "url", "serverroot", serverroot)
    wrrsetroot = configget(logger, theconf, "url", "wrrsetroot", wrrsetroot)
    oformat = configget(logger, theconf, "output", "format", "Gtiff")
    odir = configget(logger, theconf, "output", "directory", "output/")
    oprefix = configget(logger, theconf, "output", "prefix", "GADGET")
    radcordir = configget(logger, theconf, "downscaling", "radiationcordir", "output_rad")
    FNhighResDEM = configget(logger, theconf, "downscaling", "highResDEM", "downscaledem.map")
    FNlowResDEM = configget(logger, theconf, "downscaling", "lowResDEM", "origdem.map")
    saveAllData = int(configget(logger, theconf, "output", "saveall", "0"))

    # Set clone for DEM
    # pcr.setclone(FNhighResDEM)

    # Check whether downscaling should be applied
    resamplingtype = configget(logger, theconf, "selection", "resamplingtype", "linear")
    downscaling = configget(logger, theconf, "selection", "downscaling", downscaling)
    resampling = configget(logger, theconf, "selection", "resampling", resampling)

    if downscaling == 'True' or resampling == "True":
        # get grid info
        resX, resY, cols, rows, highResLon, highResLat, highResDEM, prj, FillVal = readMap(FNhighResDEM, 'GTiff', logger)
        LresX, LresY, Lcols, Lrows, lowResLon, lowResLat, lowResDEM, prj, FillVal = readMap(FNlowResDEM, 'GTiff', logger)
        # writeMap("DM.MAP","PCRaster",highResLon,highResLat,highResDEM,FillVal)
        # elevationCorrection, highResDEM, resLowResDEM = resampleDEM(FNhighResDEM,FNlowResDEM,logger)
        demmask = highResDEM != FillVal
        mismask = highResDEM == FillVal
        Ldemmask = lowResDEM != FillVal
        Lmismask = lowResDEM == FillVal
        # Fille gaps in high res DEM with Zeros for interpolation purposes
        lowResDEM[Lmismask] = 0.0
        resLowResDEM = resample_grid(lowResDEM, lowResLon, lowResLat, highResLon, highResLat, method=resamplingtype,
                                     FillVal=0.0)

        lowResDEM[Lmismask] = FillVal
        elevationCorrection = highResDEM - resLowResDEM
        # lons = highResLon
        # lats = highResLat
        #print('Elevation Correction', elevationCorrection.shape)
        #print('Lats sdfsdfsd', lats)
        #print('Lats sdfsdfsd', long)
        #save_as_gtiff(lats, lons, elevationCorrection, 0, odir, 'ElevCorrect', prj)


    # Check whether evaporation should be calculated
    calculateEvap = configget(logger, theconf, "selection", "calculateEvap", calculateEvap)

    if calculateEvap == 'True':
        evapMethod = configget(logger, theconf, "selection", "evapMethod", evapMethod)

    if evapMethod == 'PenmanMonteith':
        relevantVars = ['TA', 'RHA', 'TAMINA', 'SWI', 'UA', 'VA']

    currentdate = start
    ncnt = 0
    if EndStep == 0:
        EndStep = (end - start).days + 1

    logger.info(
        "Will start at step: " + str(StartStep) + " date/time: " + str(start + datetime.timedelta(days=StartStep)))
    logger.info("Will stop at step: " + str(EndStep) + " date/time: " + str(start + datetime.timedelta(days=EndStep)))

    while currentdate <= end:
        ncnt += 1

        if ncnt > 0 and ncnt >= StartStep and ncnt <= EndStep:
            downscale(ncnt, currentdate, filenames, variables, standard_names, serverroot, wrrsetroot, relevantVars,
                      elevationCorrection, BB, highResLon, highResLat, resLowResDEM, highResDEM, lowResLon, lowResLat,
                      lowResDEM, logger, radcordir, odir, oprefix, lonmax, lonmin, latmax, latmin, downscaling,
                      resamplingtype, oformat, saveAllData, prj, FillVal)

        gc.collect()
        currentdate += datetime.timedelta(days=1)

    logger.info("Done.")
    comptime = (time.time() - start_time)
    logger.info("Computation time : " + str(comptime) + ' seconds')


if __name__ == "__main__":
    main()
