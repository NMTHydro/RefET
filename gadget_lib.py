import getopt, sys, os
# import pcraster as pcr
import datetime
import numpy as np
import numexpr as ne
from e2o_utils import *
import gc
import time

# from memory_profiler import profile

nthreads = 4
ne.set_num_threads(nthreads)


def save_as_mapsstack_per_day(lat, lon, data, ncnt, date, directory, prefix="GADGET", oformat="Gtiff", FillVal=1E31,
                              gzip=True):
    import platform

    if not os.path.exists(directory):
        os.mkdir(directory)
    mapname = getmapname(ncnt, prefix, date)
    # print "saving map: " + os.path.join(directory,mapname)
    # writeMap(os.path.join(directory,mapname),oformat,lon,lat[::-1],flipud(data[:,:]),-999.0)
    writeMap(os.path.join(directory, mapname), oformat, lon, lat, data[:, :], FillVal)
    if gzip:
        if 'Linux' in platform.system():
            os.system('gzip ' + os.path.join(directory, mapname))

# @profile
def PenmanMonteith(currentdate, relevantDataFields, rtoa, es_mean, ea_mean, mismask, elev):
    """

    :param lat:
    :param currentdate:
    :param relevantDataFields:
    :param Tmax:
    :param Tmin:
    :return:

    relevantDataFields : ['MaxTemperature','MinTemperature','NearSurfaceSpecificHumidity',\
                         'SurfaceIncidentShortwaveRadiation','SurfaceWindSpeed','Pressure','CorrectedSpecificHumidity']
    """

    Tmax = relevantDataFields[0]
    Tmin = relevantDataFields[1] + 273.15
    # Q       =  relevantDataFields[2]
    Rsin    =  relevantDataFields[3]
    Wsp = relevantDataFields[4]
    Pres = relevantDataFields[5]
    # Q       =  relevantDataFields[6]

    Tmean = (relevantDataFields[0] + (relevantDataFields[1]+273.15)) / 2
    Tmean[mismask] = 0.0001

    """
    Computes Penman-Monteith reference evaporation
    Inputs:
        Rsin:        netCDF obj or NumPy array   -- 3D array (time, lat, lon) incoming shortwave radiation [W m-2]
        Rlin:        netCDF obj or NumPy array   -- 3D array (time, lat, lon) incoming longwave radiation [W m-2]
        Tmean:       netCDF obj or NumPy array   -- 3D array (time, lat, lon) daily mean temp [K]
        Tmax:        netCDF obj or NumPy array   -- 3D array (time, lat, lon) daily max. temp [K]
        Tmin:        netCDF obj or NumPy array   -- 3D array (time, lat, lon) daily min. temp [K]
        Wsp:         netCDF obj or NumPy array   -- 3D array (time, lat, lon) wind speed [m s-2]
        Q:           netCDF obj or NumPy array   -- 3D array (time, lat, lon) specific humididy [kg kg-1]
        Pres:        netCDF obj or NumPy array   -- 3D array (time, lat, lon) Surface air pressure [Pa]
        Pet:         netCDF obj or NumPy array   -- 3D array (time, lat, lon) for target data
    Outputs:
        trg_var:    netCDF obj or NumPy array   -- 3D array (time, lat, lon) for target data, updated with computed values
    """
    cp = 1013  # specific heat of air 1013 [J kg-1 K-1]
    TimeStepSecs = 86400  # timestep in seconds
    karman = 0.41  # von Karman constant [-]
    vegh = 0.50  # vegetation height [m]
    alpha = 0.23  # albedo, 0.23 [-]
    rs = 45  # surface resistance, 70 [s m-1]
    R = 287.058  # Universal gas constant [J kg-1 K-1]
    convmm = 1000 * TimeStepSecs  # conversion from meters to millimeters
    sigma = 4.903e-9  # stephan boltzmann [W m-2 K-4 day]
    eps = 0.622  # ratio of water vapour/dry air molecular weights [-]
    g = 9.81  # gravitational constant [m s-2]
    R_air = 8.3144621  # specific gas constant for dry air [J mol-1 K-1]
    Mo = 0.0289644  # molecular weight of gas [g / mol]
    lapse_rate = 0.0065  # lapse rate [K m-1]


    # clear sky solar radiation MJ d-1
    Rso = np.maximum(0.1, ne.evaluate("(0.75+(2*0.00005*elev)) * rtoa"))  # add elevation correction term elev with 5*10-5
    Rlnet_Watt = - sigma * (((Tmin ** 4 + Tmax ** 4) / 2) * (0.34 - 0.14 * np.sqrt(np.maximum(0, (ea_mean / 1000)))) \
                            * (1.35 * np.minimum(1, ((Rsin * 0.0864) / Rso)) - 0.35))  # ea_mean in Pa / 1000 to get kPa
    Rlnet_Watt /= 0.0864  # MJ d-1 to Watts

    Rnet = np.maximum(0, ((1 - alpha) * Rsin + Rlnet_Watt))

    #vapour pressure deficit
    vpd = np.maximum(es_mean - ea_mean, 0.)

    #Virtual temperature
    Tkv = Tmean * (1-0.378*(ea_mean/Pres))**-1
    # density of air [kg m-3]
    rho = Pres/(Tkv*R)

    # Latent heat [J kg-1]
    Lheat = (2.501-(0.002361*(Tmean-273.15)))*1e6

    # slope of vapour pressure [Pa C-1]
    deltop  = 4098. *(610.8*np.exp((17.27*(Tmean-273.15))/((Tmean-273.15)+237.3)))
    delbase = ((Tmean-273.15)+237.3)**2
    delta = deltop/delbase
    # print('delta ', delta)

    # psychrometric constant
    gamma = cp*Pres/(eps*Lheat)
    # print('gamma ', gamma)

    # aerodynamic resistance
    z = 10  # height of wind speed variable (10 meters above surface)
    Wsp_2 = Wsp * 4.87 / (np.log(67.8 * z - 5.42))  # Measured over 0.13 m full grass = [m s-1]
    # ra = 208./Wsp_2                                # 0.13 m short crop height = [s m-1]

    # Wsp_2 = Wsp*3.44/(np.log(16.3*z-5.42))         # Measured over 0.50 m tall crop height = [m s-1]
    ra = 110./Wsp_2                                  # 0.50 m tall crop height = [s m-1]


    PETmm = np.maximum(ne.evaluate("(delta * Rnet) + (TimeStepSecs *rho * cp * (vpd / ra))"), 1)
    PETmm /= np.maximum(ne.evaluate("(delta + gamma*(1 + rs/ra))"), 1)
    PETmm /= Lheat

    # PETag = pcr.numpy2pcr(Scalar, PETmm, 0.0)
    # aguila(PETag)

    if PETmm.any() == float("inf"):
        sys.exit("Value infinity found")
    else:
        pass

    return PETmm, delta, gamma, rho


def downscale(ncnt, currentdate, filenames, variables, standard_names, serverroot, wrrsetroot, relevantVars,
              elevationCorrection, BB, highResLon, highResLat, resLowResDEM, highResDEM,
              lowResLon, lowResLat, lowResDEM, logger, radcordir, odir, oprefix, lonmax, lonmin, latmax, latmin,
              downscaling, resamplingtype, oformat, saveAllData, FillVal):
    nrcalls = 0
    start_steptime = time.time()

    # Get all daily datafields needed and aad to list
    relevantDataFields = []

    year = str(currentdate.year)
    yearly_dir = os.path.join(odir, year)

    if not os.path.exists(yearly_dir):
        os.mkdir(yearly_dir)

    odir = yearly_dir

    # Get all data for this timestep
    mapname = os.path.join(odir, getmapname(ncnt, oprefix, currentdate))
    if os.path.exists(mapname) or os.path.exists(mapname + ".gz") or os.path.exists(mapname + ".zip"):
        logger.info("Skipping map: " + mapname)
    else:
        for i in range(0, len(variables)):
            if variables[i] in relevantVars:
                filename = filenames[i]
                logger.info("Getting data field: " + filename)
                standard_name = standard_names[i]
                logger.info("Get file list..")
                tlist, timelist = get_times_daily(currentdate, currentdate, serverroot, wrrsetroot, filename, logger)
                logger.info("Get dates..")

                ncstepobj = getstepdaily(tlist, BB, standard_name, logger)

                logger.info("Get data...: " + str(timelist))
                mstack = ncstepobj.getdates(timelist)
                mean_as_map = flipud(mstack.mean(axis=0)) #Time dimension is 3(2) instead of 1st in new data

                # if variables[i] == 'SurfaceIncidentShortwaveRadiation':
                #     save_as_mapsstack_per_day(lats, lons, mean_as_map, int(ncnt), currentdate, odir,
                #                               prefix='RTOT_pre_sample',oformat=oformat, FillVal=FillVal)

                logger.info("Get data body...")
                logger.info("Downscaling..." + variables[i])
                # print('Data dimensions{}'.format(mean_as_map.shape))
                # print('Lon shape{}'.format(ncstepobj.lon.shape))
                # print('Lat shape{}'.format(ncstepobj.lat.shape))

                
                # save_as_mapsstack_per_day(ncstepobj.lat,ncstepobj.lon,mean_as_map,int(ncnt),'temp',prefixes[i],oformat='GTiff')
                # mean_as_map = resample(FNhighResDEM,prefixes[i],int(ncnt),logger)
                mean_as_map = resample_grid(mean_as_map, ncstepobj.lon, ncstepobj.lat, highResLon, highResLat,
                                            method=resamplingtype, FillVal=FillVal)
                mismask = mean_as_map == FillVal
                mean_as_map = flipud(mean_as_map)
                mean_as_map[mismask] = FillVal
                if variables[i] == 'MaxTemperature':
                    mean_as_map, Tmax = correctTemp(mean_as_map, elevationCorrection, FillVal)
                if variables[i] == 'MinTemperature':
                    mean_as_map, Tmin = correctTemp(mean_as_map, elevationCorrection, FillVal)
                if variables[i] == 'SurfaceIncidentShortwaveRadiation':
                    mean_as_map, rtoa = correctRsin(mean_as_map, currentdate, radcordir, LATITUDE, logger, FillVal)
                mean_as_map[mismask] = FillVal

                relevantDataFields.append(mean_as_map)

                # only needed once to get vector of latitudes, needed to calculate Ra called by correctRsin function and PM (not needed, left over from before)
                if nrcalls == 0:
                    nrcalls = nrcalls + 1
                    latitude = ncstepobj.lat[:]
                    # assuming a resolution of 0.041665999999999315 degrees (4km lat)
                    factor = 1 / 0.041665999999999315  # ~24 instead of 8 for NLDAS (1/8 degree)
                    LATITUDE = np.ones(((factor * (latmax - latmin)), (factor * (lonmax - lonmin))))
                    for i in range(0, int((factor * (lonmax - lonmin)))):
                        LATITUDE[:, i] = LATITUDE[:, i] * latitude
                    if downscaling == 'True' or resampling == "True":
                        # save_as_mapsstack_per_day(ncstepobj.lat,ncstepobj.lon,LATITUDE,int(ncnt),'temp','lat',oformat=oformat)
                        # LATITUDE = resample(FNhighResDEM,'lat',int(ncnt),logger)
                        LATITUDE = zeros_like(highResDEM)
                        for i in range(0, LATITUDE.shape[1]):
                            LATITUDE[:, i] = highResLat

                    # assign longitudes and lattitudes grids
                    if downscaling == 'True' or resampling == "True":
                        lons = highResLon
                        lats = highResLat
                    else:
                        lons = ncstepobj.lon
                        lats = ncstepobj.lat

        # Correct Pressure separately since no data in METDATA netCDF file
        mean_as_map = correctPres(relevantDataFields, highResDEM, resLowResDEM, FillVal=FillVal)
        # mismask = mean_as_map == FillVal
        # mean_as_map[mismask] = FillVal
        relevantDataFields.append(mean_as_map)

        # Correct RH by keeping constant at lapsed temperature and adjust pressure with elevation
        es_mean, ea_mean, rh_corr, rh_org = correctQ_RH(relevantDataFields, Tmax, Tmin, highResDEM, resLowResDEM, mismask,
                                       FillVal=FillVal)
        # mean_as_map[mismask] = FillVal
        # relevantDataFields.append(mean_as_map)
        ea_org = ea_mean
        # ea_org.clip(-999, 10000, out=ea_org)

        # Apply aridity correction
        logger.info("Applying aridity correction...")
        ea_mean, Tdew_diff = arid_cor(relevantDataFields[1], ea_mean, logger)
        mean_as_map[mismask] = FillVal
        ea_arid = ea_mean
        # ea_arid.clip(-999, 10000, out=ea_arid)

        PETmm, delta, gamma, rho = PenmanMonteith(currentdate, relevantDataFields, rtoa, es_mean, ea_mean, mismask, highResDEM)
        # FIll out unrealistic values
        PETmm[mismask] = FillVal
        PETmm[isinf(PETmm)] = FillVal
        PETmm.clip(0, 50, out=PETmm)

        logger.info("Saving PM PET data for: " + str(currentdate))
        save_as_mapsstack_per_day(lats, lons, PETmm, int(ncnt), currentdate, odir, prefix=oprefix, oformat=oformat,
                                  FillVal=FillVal)
        save_as_mapsstack_per_day(lats, lons, relevantDataFields[3], int(ncnt), currentdate, odir, prefix='RTOT',
                                   oformat=oformat, FillVal=FillVal)
        save_as_mapsstack_per_day(lats, lons, relevantDataFields[1], int(ncnt), currentdate, odir, prefix='TMIN',
                                  oformat=oformat, FillVal=FillVal)
        save_as_mapsstack_per_day(lats, lons, relevantDataFields[0], int(ncnt), currentdate, odir, prefix='TMAX',
                                  oformat=oformat, FillVal=FillVal)
        save_as_mapsstack_per_day(lats,lons,delta,int(ncnt),currentdate,odir,prefix='delta',oformat=oformat,FillVal=FillVal)
        save_as_mapsstack_per_day(lats,lons,gamma,int(ncnt),currentdate,odir,prefix='gamma',oformat=oformat,FillVal=FillVal)
        save_as_mapsstack_per_day(lats,lons,rho, int(ncnt), currentdate, odir, prefix='Rho',oformat=oformat, FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,Rnet,int(ncnt),currentdate,odir,prefix='RNET',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,Rlnet_Watt,int(ncnt),currentdate,odir,prefix='RLIN',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,Ra,int(ncnt),currentdate,odir,prefix='Ra',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,Tdew_diff,int(ncnt),currentdate,odir,prefix='Tdewcor',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,ea_org,int(ncnt),currentdate,odir,prefix='ea_org',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,ea_arid,int(ncnt),currentdate,odir,prefix='ea_arid',oformat=oformat,FillVal=FillVal)

        if saveAllData:
            save_as_mapsstack_per_day(lats, lons, relevantDataFields[5], int(ncnt), currentdate, odir, prefix='PRESS',
                                      oformat=oformat, FillVal=FillVal)
            save_as_mapsstack_per_day(lats, lons, relevantDataFields[3], int(ncnt), currentdate, odir, prefix='RSIN',
                                      oformat=oformat, FillVal=FillVal)
            save_as_mapsstack_per_day(lats, lons, relevantDataFields[4], int(ncnt), currentdate, odir, prefix='WIN',
                                      oformat=oformat, FillVal=FillVal)
            save_as_mapsstack_per_day(lats, lons, relevantDataFields[2], int(ncnt), currentdate, odir, prefix='Q',
                                      oformat=oformat, FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,relevantDataFields[6],int(ncnt),currentdate,odir,prefix='Qcorr',oformat=oformat,FillVal=FillVal)
            save_as_mapsstack_per_day(lats,lons,rh_corr,int(ncnt),currentdate,odir,prefix='RHcorr',oformat=oformat,FillVal=FillVal)
            save_as_mapsstack_per_day(lats,lons,rh_org,int(ncnt),currentdate,odir,prefix='RHorg',oformat=oformat,FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,Ra,int(ncnt),currentdate,odir,prefix='Ra',oformat=oformat,FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,Tmax,int(ncnt),currentdate,odir,prefix='Tmaxraw',oformat=oformat,FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,Tmin,int(ncnt),currentdate,odir,prefix='Tminraw',oformat=oformat,FillVal=FillVal)

            # relevantDataFields : ['MaxTemperature','MinTemperature','NearSurfaceSpecificHumidity',\
            # 'SurfaceIncidentShortwaveRadiation','SurfaceWindSpeed','Pressure','CorrectedQ']

        # Empty calculated arrays
        compsteptime = (time.time() - start_steptime)
        print(str(currentdate) + ' Computation time: ' + str(compsteptime) + ' seconds' )
        #a = open("gadgetevap_comptime.txt", "a")
        #a.write(str(currentdate) + ' Computation time: ' + str(compsteptime) + ' seconds' + '\n')
        #a.close()

    pass


def correctTemp(Temp, elevationCorrection, FillVal):
    """
    Elevation based correction of temperature

    inputs:
    Temperature             = daily min or max temperature (degrees Celcius)
    Elevation correction    = difference between high resolution and low resolution (4 km) DEM  [m]

    constants:
    lapse_rate = 0.0065 # [ K m-1 ]
    """

    # apply elevation correction
    lapse_rate = 0.0065  # [ K m-1 ]

    Temp[Temp < 0.0] = FillVal
    Temp[Temp == 0.0] = FillVal
    Temp[isinf(Temp)] = FillVal

    Temp_cor = ne.evaluate("Temp - lapse_rate * elevationCorrection")
    # Tempag = pcr.numpy2pcr(Scalar, Temp, 0.0)
    # aguila(Tempag)
    # Elag = pcr.numpy2pcr(Scalar, elevationCorrection, 0.0)
    # aguila(Elag)
    Temp_cor[Temp_cor < 0.0] = FillVal
    Temp_cor[Temp_cor == 0.0] = FillVal
    Temp_cor[isinf(Temp_cor)] = FillVal

    return Temp_cor, Temp


def correctRsin(Rsin, currentdate, radiationCorDir, lat, logger, FillVal):
    """
    Calculates daily topographically corrected incoming radiation using clear-sky global, beam and diffuse solar maps outputted from r.sun

    :param Rsin:
    :param currentdate:
    :param radiationCorDir:
    :param logger:
    :return:  corrected incoming radiation (Rsin) and extraterrestrial radiation (Ra) 
    """

    path = radiationCorDir
    day = currentdate
    flatdir = 'FlatRad_DEMRes'
    topoDEM = 'RadWm2'

    # Get daily Extraterrestrial radiation
    rtoa, sunset = Ra_daily(currentdate, lat)

    rtoa /= 0.0864  # Convert from MJ day-1 to W m-2 day

    # Calculate clearness index (kt) using Ra, adjust with optical path if Kt >= 0.65
    kt = np.maximum(np.minimum(ne.evaluate("Rsin/rtoa"), 1), 0.0001)

    # Use Ruiz-Ariaz, 2010b elevation function instead of Opcorr? Or saved Opcorr from hourly algorithm?
    #np.select([kt >= 0.65, kt < 0.65], [Rsin * Opcorr, Rsin])

    # Partition radiation into beam and diffuse components for flat surface following Ruiz-Ariaz, 2010b
    # (for hourly not daily solar radiation)
    # kd = np.maximum(0.952 - (1.041 * np.exp(-1 * np.exp(2.300 - 4.702 * kt))), 0.0001)

    # Partition daily global radiation into beam and diffuse components for flat surface following Erbs et al., 1982
    #Use different coefficients between Summer, Spring, Fall vs Winter based on sunset hour angle (radians)

    # print('sunset angle radians', sunset)
    kd = np.zeros_like(Rsin)
    #Winter
    kd = np.where((sunset < 1.4208) & (kt < 0.715),
                  1.0 - 0.2727 * kt + 2.4495 * np.power(kt, 2) - 11.9514 * np.power(kt, 3) + 9.3879 * np.power(kt, 4), kd)
    kd = np.where((sunset < 1.4208) & (kt >= 0.715),
                  0.143, kd)

    #Summer
    kd = np.where((sunset >= 1.4208) & (kt < 0.722),
                  1.0 + 0.2832 * kt - 2.5557 * np.power(kt, 2) + 0.8448 * np.power(kt, 3), kd)
    kd = np.where((sunset >= 1.4208) & (kt >= 0.722),
                  0.175, kd)

    Rd_rsky = kd * Rsin
    Rb_rsky = (1-kd) * Rsin

    #Calculate clearsky beam and diffuse indices based on gridded data resolution r.sun clearsky beam and diffuse maps
    Rd_clrflat = 'Rdflat_' + day.strftime('%j') + '.tif'
    Rb_clrflat = 'Rbflat_' + day.strftime('%j') + '.tif'

    resX, resY, cols, rows, LinkeLong, LinkeLat, Rd_flat, FillVal = readMap((os.path.join(path, flatdir, Rd_clrflat)), 'Gtiff', logger)
    resX, resY, cols, rows, LinkeLong, LinkeLat, Rb_flat, FillVal = readMap((os.path.join(path, flatdir, Rb_clrflat)), 'Gtiff', logger)

    # Rb_flat = flipud(Rb_flat.mean(axis=0))
    # Rd_flat = flipud(Rb_flat.mean(axis=0))
    #
    # Rb_flat = resample_grid(Rb_flat, ncstepobj.lon, ncstepobj.lat, highResLon, highResLat, method="linear", FillVal=FillVal)
    # Rd_flat = resample_grid(Rb_flat, ncstepobj.lon, ncstepobj.lat, highResLon, highResLat, method="linear", FillVal=FillVal)

    #Calculate Beam and Diffuse clear-sky indices for flat radiation
    Kdif = np.maximum(np.minimum(Rd_rsky / Rd_flat,1),0.0001)
    Kbeam = np.maximum(np.minimum(Rb_rsky / Rb_flat,1),0.0001)

    #Read in topographically-adjusted DEM scale clear-sky beam and diffuse radiation
    Rd_clrDEM = 'Rd_' + day.strftime('%j') + '.tif'
    Rb_clrDEM = 'Rb_' + day.strftime('%j') + '.tif'

    resX, resY, cols, rows, LinkeLong, LinkeLat, Rd_DEM, FillVal = readMap((os.path.join(path, topoDEM, Rd_clrDEM)), 'Gtiff', logger)
    resX, resY, cols, rows, LinkeLong, LinkeLat, Rb_DEM, FillVal = readMap((os.path.join(path, topoDEM, Rb_clrDEM)), 'Gtiff', logger)

    logger.info("Reading clear-sky daily solar radiation:")

    #Sum and Scale topographically-adjusted clear-sky beam and diffuse components using Kbeam, Kdif clear-sky indices

    Rg_DEM = Rd_DEM * Kdif + Rb_DEM * Kbeam

    FillVal = 0
    Rg_DEM[Rsin < 0.0] = FillVal
    Rg_DEM[Rsin == 0.0] = 0.00001
    Rg_DEM[isinf(Rsin)] = FillVal

    return Rg_DEM, rtoa


def correctPres(relevantDataFields, highResDEM, resLowResDEM, FillVal=1E31):
    """
    Correction of air pressure for DEM based altitude correction based on barometric formula

    :param relevantDataFields:
    :param Pressure:
    :param highResDEM:
    :param resLowResDEM:
    :return: corrected pressure

    relevantDataFields : ['Temperature','DownwellingLongWaveRadiation','SurfaceAtmosphericPressure',\
                    'NearSurfaceSpecificHumidity','SurfaceIncidentShortwaveRadiation','NearSurfaceWindSpeed']
    """

    Tmax = relevantDataFields[0]
    Tmin = relevantDataFields[1]
    Tmean = (Tmin + Tmax) / 2

    g = 9.801  # gravitational constant [m s-2]
    R_air = 8.3144621  # specific gas constant for dry air [J mol-1 K-1]
    # R            = 287           # gas constant per kg air  [J kg-1 K-1]
    Mo = 0.0289644  # molecular weight of gas [g / mol]
    lapse_rate = 0.0065  # lapse rate [K m-1]
    Pressure = 101300  # Atmospheric pressure at sea-level [Pa]

    # Why is this, switched off for now...
    # highResDEM  = np.maximum(0,highResDEM)

    # Pressag = pcr.numpy2pcr(Scalar, Pressure, 0.0)
    # aguila(Pressag)

    Pres_corr = np.zeros_like(highResDEM)
    # Incorrect without pressure at METDATA elevation
    # Pres_corr    = ne.evaluate("Pressure *( (Tmean/ ( Tmean + lapse_rate * (highResDEM))) ** (g * Mo / (R_air * lapse_rate)))")
    Pres_corr = ne.evaluate("101300 *( (293.0 - lapse_rate * (highResDEM)) / 293.0) ** (5.26)")

    Pres_corr[isnan(Pres_corr)] = FillVal
    Pres_corr[isinf(Pres_corr)] = FillVal
    Pres_corr[Pres_corr > 150000] = FillVal

    return Pres_corr


# @profile
def correctQ_RH(relevantDataFields, Tmax, Tmin, highResDEM, resLowResDEM, mismask, FillVal=1E31):
    """
    Constant Relative Humidity with elevation using datum specific humidity and temperature

    inputs:
    Temperature             = daily mean, min or max temperature (degrees Celcius)
    Elevation correction    = difference between high resolution and low resolution (4 km) DEM  [m]

   relevantDataFields : ['MaxTemperature','MinTemperature','NearSurfaceSpecificHumidity',\
                         'SurfaceIncidentShortwaveRadiation','SurfaceWindSpeed','Pressure','CorrectedSpecificHumidity']

    """

    # constants:
    g = 9.81  # gravitational constant [m s-2]
    R_air = 8.3144621  # specific gas constant for dry air [J mol-1 K-1]
    Mo = 0.0289644  # molecular weight of gas [g / mol]
    lapse_rate = 0.006  # lapse rate [K m-1]
    eps = 0.622  # ratio of water vapour/dry air molecular weights [-]
    FillVal = 1E31
    R = 287.058  # Specific gas constant for dry air [J kg-1 K-1]
    rv = 461  # Specific gas constant for water vapor[J kg-1 K-1]
    eps = 0.622  # ratio of water vapour/dry air molecular weights (R / rv) [-]

    Temp_corr = (relevantDataFields[0] + relevantDataFields[1]) / 2
    Pres_corr = relevantDataFields[5]
    Q = relevantDataFields[2]
    # p_mb = relevantDataFields[5] / 1000
    Tmean = (Tmax + Tmin) / 2  # Original Tmean, Tmax without elevation lapse adjustment

    # saturation vapour pressure [Pa]
    es_ref = ne.evaluate("610.8*exp((17.27*(Tmean-273.15))/((Tmean-273.15)+237.3))")
    es_elev = ne.evaluate("610.8*exp((17.27*(Temp_corr-273.15))/((Temp_corr-273.15)+237.3))")
    es_elev[isinf(es_elev)] = FillVal
    ##    tag = pcr.numpy2pcr(Scalar, es_ref, FillVal)
    ##    aguila(tag)
    ##    tags = pcr.numpy2pcr(Scalar, es_elev, FillVal)
    ##    aguila(tags)

    # actual vapour pressure [Pa]
    ea_ref = ne.evaluate("-(Q*Pres_corr)/((eps-1)*Q-eps)")
    ea_ref[mismask] = 0.0001
    ea_corr = ne.evaluate("(ea_ref / es_ref) * es_elev")  # Set actual vapor pressure equal to reference RH
    ea_corr[isinf(ea_corr)] = 0.0001
    ea_corr[isnan(ea_corr)] = 0.0001
    ea_corr[ea_corr <= 0] = 0.0001
    ea_corr[mismask] = 0.0001

    rh_corr = ne.evaluate("ea_corr / es_elev")
    rh_corr[rh_corr < 0.0] = FillVal
    rh_org = ea_ref / es_ref

    return es_elev, ea_corr, rh_corr, rh_org


def arid_cor(Tmin, ea, logger):
    # Calculate dew point after correcting for constant RH at elevation for aridity correction

    ea_kPa = ea / 1000.0
    Tdew = (np.log(ea_kPa) + 0.49299) / (0.0707 - 0.00421 * np.log(ea_kPa))  # Shuttleworth, 2012 in degrees C; ea in kPA
    Tmin -= 273.15  # Convert from K to degrees C to compare

    # Check daily max difference between Tmin minus Tdew
    # Apply aridity correction where Tdew is > 2 degrees C less than Tmin and make Tdew equal to Tmin - 2

    # Read in NLCD agricultural areas (NLCD LC 82), NLCD 81 (Hay / Pasture), 90 (Woody Wetlands), 95 (Emergent Herbaceous Wetlands )
    # nlcd_nm_wgs84_AgWetlands_sm.tif  (Includes 81, 82, 90, 95), 255 = no class
    path = 'DEM/'
    # NLCDAg_file = 'nlcd_nm_wgs84_Agfields_sm.tif'
    NLCDAg_file = 'nlcd_nm_wgs84_AgWetlands_sm.tif'
    resX, resY, cols, rows, LinkeLong, LinkeLat, NLCDAg, FillVal = readMap((os.path.join(path, NLCDAg_file)), 'Gtiff',
                                                                           logger)
    Tmindif = Tmin - Tdew

    Tdew_cor = Tdew
    Tdew_cor = where((NLCDAg != 255) & (Tmindif > 2),
                     Tmin - 2.0, Tdew)

    ea_kPa_cor = 0.6108 * np.exp(17.27 * Tdew_cor / (Tdew_cor + 237.3))  # (ASCE, 2005): ea in kPa, ASCE in degrees C
    ea_cor = ea_kPa * 1000  # Convert kPa to Pa

    # Save Tdew correction difference to verify working
    Tdew_diff = Tmin - Tdew_cor

    return ea_cor, Tdew_diff


def Ra_daily(currentdate, lat):
    # CALCULATE EXTRATERRESTRIAL RADIATION
    # get day of year
    tt = currentdate.timetuple()
    JULDAY = tt.tm_yday
    #    #Latitude radians
    LatRad = ne.evaluate("lat*pi/180.0")
    # declination (rad)
    declin = ne.evaluate("0.4093*(sin(((2.0*pi*JULDAY)/365.0)-1.39))")

    # sunset hour angle
    # arccosInput = ne.evaluate("-1*(tan(LatRad))*(tan(declin))")
    arccosInput = -1 * (np.tan(LatRad)) * np.tan(declin)
    arccosInput = np.minimum(1, arccosInput)
    arccosInput = np.maximum(-1, arccosInput)
    sunangle = ne.evaluate("arccos(arccosInput)") #Sunset hour angle (rads)
    #    # distance of earth to sun
    distsun = ne.evaluate("1+0.033*(cos((2*pi*JULDAY)/365.0))")
    # Ra = water equivalent extra terrestiral radiation in MJ day-1
    rtoa = ne.evaluate("((24 * 60 * 0.082) / 3.14) * distsun * (sunangle*(sin(LatRad))*(sin(declin))+(cos(LatRad))*(cos(declin))*(sin(sunangle)))")
    rtoa[rtoa < 0] = 0
    # Raag = numpy2pcr(Scalar, Ra, 0.0)
    # aguila(Raag)

    return rtoa, sunangle
