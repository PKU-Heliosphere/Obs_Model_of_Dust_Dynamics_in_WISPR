import datetime

import spiceypy as spice
import sys
import json
import base64
import requests

spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_v300.tf')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/'
             'PSP_kernels/spp_2018_224_2025_243_RO5_00_nocontact.alp.bc')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_001.tf')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_dyn_v201.tf')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_wispr_v002.ti')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_sclk_0865.tsc')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20181008_20190120_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20190120_20190416_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20190416_20190914_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20200101_20200301_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20200802_20201016_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20201016_20210101_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20210101_20210226_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20210723_20210904_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20210904_20211104_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20211104_20211217_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20211217_20220329_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20220329_20220620_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20220620_20220725_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/PSP_kernels/spp_recon_20220725_20220923_v001.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/Normal_kernels/pck00010.tpc')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/Normal_kernels/naif0012.tls')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/Normal_kernels/earthstns_itrf93_201023.bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/SmallBody_kernels/'
             'phaethon_2003200(2018-07-10_2022-07-18).bsp')
spice.furnsh('D://Microsoft Download/Formal Files/data file/Kernels/Normal_kernels/de440.bsp')


def get_spk_from_horizons(spkid, start_time, stop_time):
    # Define API URL and SPK filename:
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api'
    spk_filename = 'spk_file.bsp'

    # # Define the time span:
    # start_time = '2021-10-10'
    # stop_time = '2022-10-10'

    # # # Get the requested SPK-ID from the command-line:
    # # if (len(sys.argv)) == 1:
    # #     print("please specify SPK-ID on the command-line");
    # #     sys.exit(2)
    # # spkid = sys.argv[1]
    # Define the spkid
    # spkid = 2163693

    # Build the appropriate URL for this API request:
    # IMPORTANT: You must encode the "=" as "%3D" and the ";" as "%3B" in the
    #            Horizons COMMAND parameter specification.
    url += "?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
    url += "&COMMAND='DES%3D{}%3B'&START_TIME='{}'&STOP_TIME='{}'".format(spkid, start_time, stop_time)

    # Submit the API request and decode the JSON-response:
    response = requests.get(url)
    try:
        data = json.loads(response.text)
    except ValueError:
        print("Unable to decode JSON results")

    # If the request was valid...
    if (response.status_code == 200):
        #
        # If the SPK file was generated, decode it and write it to the output file:
        if "spk" in data:
            #
            # If a suggested SPK file basename was provided, use it:
            if "spk_file_id" in data:
                spk_filename = 'D://Desktop/'+data["spk_file_id"] +'('+start_time+'_'+stop_time+')'+ ".bsp"
                # spk_filename = str(spkid) +'('+start_time+'_'+stop_time+')'+ ".bsp"
            try:
                f = open(spk_filename, "wb")
            except OSError as err:
                print("Unable to open SPK file '{0}': {1}".format(spk_filename, err))
            #
            # Decode and write the binary SPK file content:
            f.write(base64.b64decode(data["spk"]))
            f.close()
            print("wrote SPK content to {0}".format(spk_filename))
            return data["spk_file_id"]
            sys.exit()
        #
        # Otherwise, the SPK file was not generated so output an error:
        print("ERROR: SPK file not generated")
        if "result" in data:
            print(data["result"])
        else:
            print(response.text)
        sys.exit(1)

    # If the request was invalid, extract error content and display it:
    if (response.status_code == 400):
        data = json.loads(response.text)
        if "message" in data:
            print("MESSAGE: {}".format(data["message"]))
        else:
            print(json.dumps(data, indent=2))

    # Otherwise, some other error occurred:
    print("response code: {0}".format(response.status_code))
    sys.exit(2)


# the_time = spice.datetime2et(datetime.datetime.strptime('2021-01-11T03:00:16', '%Y-%m-%dT%H:%M:%S'))
# pos = spice.spkpos('2003200', the_time, 'SPP_HCI', 'None', 'SUN')
# print(pos)
# phaethon_id = 2003200
# start_time = '2018-07-10'
# end_time = '2022-07-18'
# get_spk_from_horizons(phaethon_id, start_time, end_time)

