import getopt, sys, os
# import pcraster as pcr
import datetime
import numpy as np
import numexpr as ne
from RegCM3_utils import *
import gc
import time

# from memory_profiler import profile

nthreads = 2
ne.set_num_threads(nthreads)


def save_as_mapsstack_per_day(lat, lon, data, ncnt, date, directory, prj, prefix="GADGET", oformat="Gtiff",  FillVal=1E31,
                              gzip=True):
    import platform

    if not os.path.exists(directory):
        os.mkdir(directory)
    mapname = getmapname(ncnt, prefix, date)
    # print "saving map: " + os.path.join(directory,mapname)
    # writeMap(os.path.join(directory,mapname),oformat,lon,lat[::-1],flipud(data[:,:]),-999.0)
    writeMap(os.path.join(directory, mapname), oformat, lon, lat, data[:, :], prj, FillVal)
    if gzip:
        if 'Linux' in platform.system():
            os.system('gzip ' + os.path.join(directory, mapname))

# @profile
def PenmanMonteith(currentdate, relevantDataFields, rtoa, es_mean, ea_mean, Pres, elev):

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

    Tmean = relevantDataFields[0]
    Ua = relevantDataFields[4]
    Uv = relevantDataFields[5]
    #Tmax = relevantDataFields[0]
    #Tmin = relevantDataFields[1]
    # Q       =  relevantDataFields[2]
    Rsin    =  relevantDataFields[3]
    #Pres = relevantDataFields[5]
    # Q       =  relevantDataFields[6]

    #Calculate resultant wind speed by taking sqrt of north and west vectors
    Wsp = np.sqrt(Ua**2 + Uv**2)

    # Convert C to K for Rnl, delta, and RefET calculation
    Tmean += 273.16

    # Convert kPa to Pa for consistent units
    ea_mean *= 1000.0
    es_mean *= 1000.0

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
    rs = 45  # surface resistance, 70 [s m-1] for grass reference
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

    #Tmean in K, ea_mean in Pa (/ 1000 to get kPa), rsin in W/m^2 (*0.0864 for MJ d-1)
    Rlnet_Watt = - sigma * ((Tmean ** 4) * (0.34 - 0.14 * np.sqrt(np.maximum(0, (ea_mean/1000))))
                            * (1.35 * np.minimum(1, ((Rsin * 0.0864) / Rso)) - 0.35))
    Rlnet_Watt /= 0.0864  # MJ d-1 to Watts

    Rnet = np.maximum(0, ((1-alpha) * Rsin + Rlnet_Watt))

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

    PETmm = np.maximum(ne.evaluate("(delta * Rnet) + (rho * cp * (vpd / ra))"), 1)
    PETmm /= np.maximum(ne.evaluate("(delta + gamma*(1 + rs/ra))"), 1)
    PETmm *= (TimeStepSecs/Lheat)
    # PETag = pcr.numpy2pcr(Scalar, PETmm, 0.0)
    # aguila(PETag)

    if PETmm.any() == float("inf"):
        sys.exit("Value infinity found")
    else:
        pass

    return PETmm, Wsp, delta, rho, vpd, Rlnet_Watt


def downscale(ncnt, currentdate, filenames, variables, standard_names, serverroot, wrrsetroot, relevantVars,
              elevationCorrection, BB, highResLon, highResLat, resLowResDEM, highResDEM,
              lowResLon, lowResLat, lowResDEM, logger, radcordir, odir, oprefix, lonmax, lonmin, latmax, latmin,
              downscaling, resamplingtype, oformat, saveAllData, prj, FillVal):
    nrcalls = 0
    start_steptime = time.time()

    #Need to read in actual lat / long of high-resolution grid
    #Change data highResDEM to highResDEMlatlong so doesn't overwrite LCC elevation values
    #Change prj to prj_WGS84 so radiation, refet rasters have correct LCC project not WGS84
    highresDEMwgs84 = 'DEM/NMbufferMatchLCCasWGS84.tif'
    resX, resY, cols, rows, highResLondeg, highResLatdeg, highResDEMlatlong, prj_WGS84, Fills = readMap(highresDEMwgs84, 'GTiff', logger)

    # lons = highResLon
    # lats = highResLat

    # elevationCorrection = ne.evaluate("highResDEM - resLowResDEM")
    # print('Elevation Correction', elevationCorrection.shape)
    # print('Elevation Correction', elevationCorrection)

    # Get all daily datafields needed and aad to list
    relevantDataFields = []

    year = str(currentdate.year)

    pm_year = 'PM{}'.format(year)
    rad_year = 'rad{}'.format(year)

    odirPM = os.path.join(odir, pm_year)
    if not os.path.exists(odirPM):
        os.mkdir(odirPM)
    odirrad = os.path.join(odir, rad_year)
    if not os.path.exists(odirrad):
        os.mkdir(odirrad)

    # Get all data for this timestep
    mapname = os.path.join(odirPM, getmapname(ncnt, oprefix, currentdate))
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
                mean_as_map = (mstack.mean(axis=0))  #Time dimension is 3(2) instead of 1st in new data

                # save_as_mapsstack_per_day(ncstepobj.lat, ncstepobj.lat, mean_as_map, int(ncnt), currentdate, odir, prj,
                #                           prefix='Tmeanpre_resample',
                #                           oformat=oformat, FillVal=FillVal)

                logger.info("Get data body...")
                logger.info("Downscaling..." + variables[i])
                #print('Data dimensions{}'.format(mean_as_map.shape))
                #print('Lon shape{}'.format(ncstepobj.lon.shape))
                #print('Lat shape{}'.format(ncstepobj.lat.shape))

                # save_as_mapsstack_per_day(ncstepobj.lat,ncstepobj.lon,mean_as_map,int(ncnt),'temp',prefixes[i],oformat='GTiff')
                # mean_as_map = resample(FNhighResDEM,prefixes[i],int(ncnt),logger)
                mean_as_map = resample_grid(mean_as_map, ncstepobj.lon, ncstepobj.lat, highResLon, highResLat,
                                            method=resamplingtype, FillVal=-999)
                mismask = mean_as_map == FillVal
                mean_as_map = flipud(mean_as_map)
                # FillVal = 0
                mean_as_map[mismask] = FillVal
                if variables[i] == 'TA':
                    # save_as_mapsstack_per_day(lats, lons, mean_as_map, int(ncnt), currentdate, odir, prj,
                    #                           prefix='Tmean_early',
                    #                           oformat=oformat, FillVal=FillVal)
                    mean_as_map = correctTemp(mean_as_map, elevationCorrection, FillVal)
                    # print('Corrected average temperature', tempavg)
                    # print('Data dimensions{}'.format(mean_as_map.shape))
                if variables[i] == 'TAMINA':
                    mean_as_map = correctTemp(mean_as_map, elevationCorrection, FillVal)
                # if variables[i] == 'RHA':
                    # relhum = mean_as_map
                    # print('average RH mean', relevantDataFields[2])
                    # print('average RH', relhum)
                if variables[i] == 'SWI':
                    mean_as_map, rtoa = correctRsin(mean_as_map, currentdate, radcordir, LATITUDE, logger, FillVal)
                mean_as_map[mismask] = FillVal

                relevantDataFields.append(mean_as_map)

                # # only needed once to get vector of latitudes
                if nrcalls == 0:
                    nrcalls = nrcalls + 1
                    latitude = ncstepobj.latdeg[:]
                    # print('sfasdfs', latitude)
                    latmin = 31.125
                    latmax = 37.125
                    lonmin = -109.375
                    lonmax = -102.75
                    # assuming a resolution of 0.041665999999999315 degrees (4km lat)
                    # RegCM3 Lambert Conformal Conic projection pixel size is 15 km
                    latfactor = 1 / 0.136363636
                    lonfactor = 1 / 0.154069767
                    #latfactor = 1/15 #X coordinate factor for LCC units = 15 km
                    #lonfactor = 1/15 #Y coordinate factor for LCC units = 15 km

                    LATITUDE = np.ones(((latfactor * (latmax - latmin)), (lonfactor * (lonmax - lonmin))))

                    numlats =int((latfactor * (lonmax - lonmin)))
                    numlongs =int((lonfactor * (latmax - latmin)))
                    # print('number of lats', numlats)
                    # print('number of longs', numlongs)
                    #print('latitudes', latitude.shape)

                    for i in range(0, int((lonfactor * (lonmax - lonmin)))):
                        LATITUDE[:, i] = LATITUDE[:, i] * latitude
                    if downscaling == 'True' or resampling == "True":
                        # save_as_mapsstack_per_day(ncstepobj.lat,ncstepobj.lon,LATITUDE,int(ncnt),'temp','lat',oformat=oformat)
                        # LATITUDE = resample(FNhighResDEM,'lat',int(ncnt),logger)
                        LATITUDE = zeros_like(highResDEM)
                        for i in range(0, LATITUDE.shape[1]):
                            LATITUDE[:, i] = highResLatdeg

                        # assign longitudes and lattitudes grids
                    if downscaling == 'True' or resampling == "True":
                        lons = highResLon
                        lats = highResLat
                    else:
                        lons = ncstepobj.lon
                        lats = ncstepobj.lat

        Pres = correctPres(relevantDataFields, highResDEM, resLowResDEM, FillVal=FillVal)
        # mismask = mean_as_map == FillVal
        # mean_as_map[mismask] = FillVal
        #relevantDataFields.append(mean_as_map)

        # Correct RH by keeping constant at lapsed temperature and adjust pressure with elevation
        # es_mean, ea_mean = correctQ_RH(tempavg, relhum, highResDEM, resLowResDEM,
        #                                FillVal=FillVal)
        # mean_as_map[mismask] = FillVal
        # relevantDataFields.append(mean_as_map)
        # ea_org = ea_mean
        # ea_org.clip(0, 10000, out=ea_org)

        # Apply aridity correction
        logger.info("Applying aridity correction...")
        ea_mean, es_mean, rh_org = arid_cor(relevantDataFields, logger)
        ea_mean[isinf(ea_mean)] = FillVal
        es_mean[isinf(es_mean)] = FillVal
        rh_cor = ea_mean / es_mean
        # ea_arid = ea_mean
        # ea_arid.clip(0, 10000, out=ea_arid)

        # print ('Length of relevantDataFields', len(relevantDataFields))
        # print '\n'.join(str(p) for p in relevantDataFields)
        PETmm, Wsp, delta, rho, vpd, Rlnet_Watt = PenmanMonteith(currentdate, relevantDataFields, rtoa, es_mean, ea_mean, Pres, highResDEM)
        # FIll out unrealistic values
        # PETmm[mismask] = FillVal
        PETmm[isinf(PETmm)] = FillVal
        # PETmm.clip(0, 50, out=PETmm)

        logger.info("Saving PM PET data for: " + str(currentdate))
        save_as_mapsstack_per_day(lats, lons, PETmm, int(ncnt), currentdate, odirPM, prj, prefix=oprefix, oformat=oformat,
                                  FillVal=FillVal)
        save_as_mapsstack_per_day(lats, lons, relevantDataFields[3], int(ncnt), currentdate, odirrad, prj, prefix='RTOT',
                                   oformat=oformat, FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,vpd,int(ncnt),currentdate,odir,prj,prefix='VPD',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,Rlnet_Watt,int(ncnt),currentdate,odir,prj,prefix='RLIN',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,rho,int(ncnt),currentdate,odir,prj,prefix='RHO',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,rh_cor,int(ncnt),currentdate,odir,prj,prefix='rh_cor',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,rh_org,int(ncnt),currentdate,odir,prj,prefix='rh_org',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,ea_mean,int(ncnt),currentdate,odir,prj,prefix='ea_mean',oformat=oformat,FillVal=FillVal)
        # save_as_mapsstack_per_day(lats,lons,rtoa,int(ncnt),currentdate,odir,prj,prefix='rtoa',oformat=oformat,FillVal=FillVal)

        if saveAllData:
            save_as_mapsstack_per_day(lats, lons, relevantDataFields[0], int(ncnt), currentdate, odir, prj, prefix='Tmean',
                                      oformat=oformat, FillVal=FillVal)
            save_as_mapsstack_per_day(lats, lons, delta, int(ncnt), currentdate, odir, prj, prefix='delta',
                                      oformat=oformat, FillVal=FillVal)
            # save_as_mapsstack_per_day(lats, lons, relevantDataFields[1], int(ncnt), currentdate, odir, prj, prefix='RH',
            #                           oformat=oformat, FillVal=FillVal)
            # save_as_mapsstack_per_day(lats, lons, rh_cor, int(ncnt), currentdate, odir, prj, prefix='RH_cor',
            #                           oformat=oformat, FillVal=FillVal)
            # save_as_mapsstack_per_day(lats, lons, Rlnet_Watt, int(ncnt), currentdate, odir, prj, prefix='Rlnet',
            #                           oformat=oformat, FillVal=FillVal)
            # save_as_mapsstack_per_day(lats, lons, Pres, int(ncnt), currentdate, odir, prj, prefix='PRESS',
            #                            oformat=oformat, FillVal=FillVal)
            # save_as_mapsstack_per_day(lats, lons, Ra, int(ncnt), currentdate, odir, prj, prefix='Ra', oformat=oformat,
            #                           FillVal=FillVal)
            # save_as_mapsstack_per_day(lats, lons, Wsp, int(ncnt), currentdate, odir, prj, prefix='Wsp', oformat=oformat,
            #                           FillVal=FillVal)
            # save_as_mapsstack_per_day(lats, lons, ea_mean, int(ncnt), currentdate, odir, prj, prefix='ea',
            #                           oformat=oformat, FillVal=FillVal)
            # save_as_mapsstack_per_day(lats, lons, es_mean, int(ncnt), currentdate, odir, prj, prefix='es',
            #                           oformat=oformat, FillVal=FillVal)
            #save_as_mapsstack_per_day(lats, lons, relevantDataFields[2], int(ncnt), currentdate, odir, prefix='Q',
                                      # oformat=oformat, FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,relevantDataFields[6],int(ncnt),currentdate,odir,prefix='Qcorr',oformat=oformat,FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,RHref,int(ncnt),currentdate,odir,prefix='RHref',oformat=oformat,FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,RHcorr,int(ncnt),currentdate,odir,prefix='RHcorr',oformat=oformat,FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,Ra,int(ncnt),currentdate,odir,prefix='Ra',oformat=oformat,FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,Tmax,int(ncnt),currentdate,odir,prefix='Tmaxraw',oformat=oformat,FillVal=FillVal)
            # save_as_mapsstack_per_day(lats,lons,Tmin,int(ncnt),currentdate,odir,prefix='Tminraw',oformat=oformat,FillVal=FillVal)

            # relevantDataFields : ['MaxTemperature','MinTemperature','NearSurfaceSpecificHumidity',\
            # 'SurfaceIncidentShortwaveRadiation','SurfaceWindSpeed','Pressure','CorrectedQ']

        # Empty calculated arrays
        compsteptime = (time.time() - start_steptime)
        print(str(currentdate) + ' Computation time: ' + str(compsteptime) + ' seconds' + '\n')
        #a = open("gadgetevap_comptime.txt", "a")
        #a.write(str(currentdate) + ' Computation time: ' + str(compsteptime) + ' seconds' + '\n')
        #a.close()

    pass


def correctTemp(Temp, elevationCorrection, FillVal):
    """
    Elevation based correction of temperature

    inputs:
    Temperature             = daily min or max temperature (degrees Celcius)
    Elevation correction    = difference between high resolution and low resolution (15 km) DEM  [m]

    constants:
    lapse_rate = 0.0065 # [ K m-1 ]
    """

    # apply elevation correction
    lapse_rate = 0.0065  # [ K m-1 ]

    # Temp[Temp < 0.0] = FillVal
    # Temp[Temp == 0.0] = FillVal
    # Temp[isinf(Temp)] = FillVal

    Temp_cor = ne.evaluate("Temp - lapse_rate * elevationCorrection")
    # Tempag = pcr.numpy2pcr(Scalar, Temp, 0.0)
    # aguila(Tempag)
    # Elag = pcr.numpy2pcr(Scalar, elevationCorrection, 0.0)
    # aguila(Elag)
    # Temp_cor[Temp_cor < 0.0] = FillVal
    # Temp_cor[Temp_cor == 0.0] = FillVal
    # Temp_cor[isinf(Temp_cor)] = FillVal

    return Temp_cor


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

    rtoa /= 0.0864 #Convert from MJ day-1 to W m-2 day

    # Calculate clearness index (kt) using Ra, adjust with optical path if Kt >= 0.65
    kt = np.maximum(np.minimum(ne.evaluate("Rsin/rtoa"), 1), 0.0001)

    # Use Ruiz-Ariaz, 2010b elevation function instead of Opcorr? Or saved Opcorr from hourly algorithm?
    #np.select([kt >= 0.65, kt < 0.65], [Rsin * Opcorr, Rsin])

    # Partition radiation into bbeam and diffuse components for flat surface following Ruiz-Ariaz, 2010b
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

    resX, resY, cols, rows, LinkeLong, LinkeLat, Rd_flat, prj, FillVal = readMap((os.path.join(path, flatdir, Rd_clrflat)), 'Gtiff', logger)
    resX, resY, cols, rows, LinkeLong, LinkeLat, Rb_flat, prj, FillVal = readMap((os.path.join(path, flatdir, Rb_clrflat)), 'Gtiff', logger)

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

    resX, resY, cols, rows, LinkeLong, LinkeLat, Rd_DEM, prj, FillVal = readMap((os.path.join(path, topoDEM, Rd_clrDEM)), 'Gtiff', logger)
    resX, resY, cols, rows, LinkeLong, LinkeLat, Rb_DEM, prj, FillVal = readMap((os.path.join(path, topoDEM, Rb_clrDEM)), 'Gtiff', logger)

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

    #Tmax = relevantDataFields[0]
    #Tmin = relevantDataFields[1]
    #Tmean = (Tmin + Tmax) / 2

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
    # Pres_corr    = ne.evaluate("Pressure *( (Tmean/ ( Tmean + lapse_rate * (highResDEM-resLowResDEM))) ** (g * Mo / (R_air * lapse_rate)))")
    Pres_corr = ne.evaluate("101300 *( (293.0 - lapse_rate * (highResDEM)) / 293.0) ** (5.26)")

    Pres_corr[isnan(Pres_corr)] = FillVal
    Pres_corr[isinf(Pres_corr)] = FillVal
    Pres_corr[Pres_corr > 150000] = FillVal

    return Pres_corr

def arid_cor(relevantDataFields, logger):
    # Calculate dew point after correcting for constant RH at elevation for aridity correction

    Tmean = relevantDataFields[0]
    relhumid = relevantDataFields[1]
    Tmin = relevantDataFields[2]

    es = ne.evaluate("0.6108*exp((17.27*Tmean)/(Tmean+237.3))") #saturation vapor pressure in kPa, Tmean in degrees C
    # es = ne.evaluate("610.8*exp((17.27*(Tmean-273.15))/((Tmean-273.15)+237.3))")  #For degrees K, and Pa
    ea_mean = (es * relhumid)  # No /100, since RH is out of 1 not 100 so don't need to divide by 100

    # print('Tmean ljklkjgsd', Tmean)
    # print('Tmin ljklkjgsd', Tmin)
    # print('RH safasdf', relhumid)
    # print('es_ref safasdf', es)
    # print('ea_ref fjghgfh', ea_mean)

    Tdew = (np.log(ea_mean) + 0.49299) / (0.0707 - 0.00421 * np.log(ea_mean))  # Shuttleworth, 2012 in degrees C; ea in kPA
    # print('Tdew asdfasdfa', Tdew)

    # Tmin -= 273.15  # Convert from K to degrees C to compare (from METDATA units)

    # Read in NLCD agricultural areas (NLCD LC 82) in raster file
    # Use NLCD 81 (Hay / Pasture), 90 (Woody Wetlands), 95 (Emergent Herbaceous Wetlands )?
    # nlcd_nm_wgs84_AgWetlands_sm.tif  (Includes 81, 82, 90, 95), 255 = no class
    path = 'DEM/'
    # NLCDAg_file = 'nlcd_nm_wgs84_Agfields_sm.tif'
    NLCDAg_file = 'nlcd_nm_wgs84_AgWetlands_LCC.tif'
    resX, resY, cols, rows, LinkeLong, LinkeLat, NLCDAg, prj, FillVal = readMap((os.path.join(path, NLCDAg_file)), 'Gtiff',
                                                                           logger)
    Tmindif = Tmin - Tdew

    # Check daily max difference between Tmin minus Tdew
    # Apply aridity correction where Tdew is > 2 degrees C less than Tmin to make Tdew equal to Tmin - 2
    Tdew_cor = Tdew
    Tdew_cor = where((NLCDAg != 255) & (Tmindif > 2), Tmin - 2.0, Tdew)
    ea_cor = 0.6108 * np.exp(17.27 * Tdew_cor / (Tdew_cor + 237.3))  # (ASCE, 2005): ea in kPa, ASCE in degrees C
    # ea_cor *= 1000  # Convert kPa to Pa

    es[es < 0.0] = FillVal
    es[es == 0.0] = 0.00001
    es[isinf(es)] = FillVal

    ##    tag = pcr.numpy2pcr(Scalar, Temp_corr, FillVal)
    ##    aguila(tag)
    ##
    ##    tags = pcr.numpy2pcr(Scalar, Tmean, FillVal)
    ##    aguila(tags)

    # Save Tdew correction difference
    # Tdew_diff = Tdew_cor - Tdew
    # Tdew_diff = np.where(Tmindif > 1, Tmindif, 0)
    # Tdew_diff = Tmin - Tdew_cor
    # Tdew_diff = Tmindif

    return ea_cor, es, relhumid


def Ra_daily(currentdate, lat):
    # print('LATITUDE PASSED TO Ra', lat)

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
    # print(arccosInput)

    arccosInput = np.minimum(1, arccosInput)
    arccosInput = np.maximum(-1, arccosInput)
    sunangle = ne.evaluate("arccos(arccosInput)")
    #    # distance of earth to sun
    distsun = ne.evaluate("1+0.033*(cos((2*pi*JULDAY)/365.0))")
    # Ra = water equivalent extra terrestiral radiation in MJ day-1
    Ra = ne.evaluate("((24 * 60 * 0.082) / 3.14) * distsun * (sunangle*(sin(LatRad))*(sin(declin))+(cos(LatRad))*(cos(declin))*(sin(sunangle)))")
    # Ra[Ra < 0] = 0
    # Raag = numpy2pcr(Scalar, Ra, 0.0)
    # aguila(Raag)

    return Ra, sunangle
