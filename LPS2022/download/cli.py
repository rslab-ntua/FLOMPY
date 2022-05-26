import os
from os import write
import sys
import optparse
from downloader import downloadAPI

def main():
    """CLI for sentinelsat python package.
    todo:
        * Work with other Sentinel programs i.e S1
    """
    # parse command line arguments
    if len(sys.argv) == 1:
        prog = os.path.basename(sys.argv[0])
        print('python3 ' + sys.argv[0] + ' [options]')
        print("python3 ", prog, " --help")
        print("python3 ", prog, " -h")
        sys.exit(-1)
    else:
        usage = "usage: prog [options] "

        parser = optparse.OptionParser()

        parser.add_option("-d", "--start_date", dest="start_date", action="store", type="string",
                        help="start date, fmt('20181123').", default=None)

        parser.add_option("-e", "--end_date", dest="end_date", action="store", type="string",
                        help="end date, fmt('20181124').", default=None)
        
        parser.add_option("-l", "--level", dest="level", action="store", type="string",
                        help="Refers to Sentinel 2 L1C and L2A", default="L1C")
        
        parser.add_option("-n", "--no_download", dest="no_download", action="store_true",
                        help="Do not download products.", default=False)
        
        parser.add_option("-m", "--max_cloud", dest="max_cloud", action="store", type="int",
                        help="Do not download products with more cloud percentage ", default=100)
        
        parser.add_option("-w", "--write_dir", dest="write_dir", action="store", type="string",
                        help="Path where the products should be downloaded", default='.')
        
        parser.add_option("-s", "--sentinel", dest="sentinel", action="store", type="string",
                        help="Sentinel mission considered", default='S2')
        
        parser.add_option("-t", "--tiles", dest="tiles", action="append",
                        help="Sentinel-2 Tiles numbers", default=None)
        
        parser.add_option("-f", "--filename", dest="filename", action="store", type="string",
                        help="Filename with stored APIHub credentials", default=None)
    
    (options, args) = parser.parse_args()

    # -t
    if (options.tiles) == None:
        print ("Provide at least one tile (-t) to download.")
        sys.exit(-1)
    else:
        tiles = options.tiles
        
        # -f
        if options.filename == None:
            print("Provide at least a password file (-f).")
            sys.exit(-2)
        else:
            # -d
            # if no start date is provided search 7 days back
            if options.start_date == None:
                start = "NOW-7DAYS"
            else:
                start = options.start_date
            # -e
            # until today
            if options.end_date == None:    
                end = "NOW"
            else:
                end = options.end_date
            # -f
            # Reading password file
            try:
                f = open(options.filename)
                (account, passwd) = f.readline().split(' ')
                if passwd.endswith('\n'):
                    passwd = passwd[:-1]
                f.close()
                #print ("Username: {}".format(account))
                #print ("Password: {}".format(passwd))
            except IOError:
                print("Error with the password file (-f).")
                sys.exit(-2)
            # -s
            if options.sentinel != 'S2':
                print ('Currently works only for Sentinel 2...')
                sys.exit(-2)
            platform = 'Sentinel-2'
            # -l
            if options.level == 'L2A':
                product = 'S2MSI2A'
            elif options.level == 'L1C':
                product = 'S2MSI1C'
            else:
                print ('Currently works only for Sentinel 2 and levels L2A and L1C.')
                sys.exit(-2)
            # -n
            if options.no_download == True:
                download = False
            else:
                download  = True

            # -w
            write_dir = options.write_dir
 
            # -m
            if options.max_cloud == 100:
                downloadAPI(account, passwd, tiles, platform, product, start, end, download = download, write_dir = write_dir)
            else:
                cloudcoverage = options.max_cloud
                downloadAPI(account, passwd, tiles, platform, product, start, end, download = download, write_dir = write_dir, cloudcoverage = cloudcoverage)

if __name__ == "__main__":
    main()   