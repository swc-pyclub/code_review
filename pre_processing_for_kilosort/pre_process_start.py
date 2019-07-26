import sys
import cav_from_arti_sub as cav
import remove_artifact_large_data as arti

if __name__ == '__main__':
    arti.arti_remove_and_save(*sys.argv[1:])
    cav.cav_and_save(*sys.argv[1:])
