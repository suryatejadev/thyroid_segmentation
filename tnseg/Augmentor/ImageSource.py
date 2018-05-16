from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import os
import glob


class ImageSource(object):
    """
    The ImageSource class is used to search for and contain paths to images for augmentation.
    """
    def __init__(self, source_directory, recursive_scan=False):
        source_directory = os.path.abspath(source_directory)
        self.image_list = self.scan_directory(source_directory, recursive_scan)

        self.largest_file_dimensions = (800, 600)

    def scan_directory(self, source_directory, recusrive_scan=False):
        # TODO: Make this a static member somewhere later
        file_types = ['*.jpg', '*.bmp', '*.jpeg', '*.gif', '*.img', '*.png']
        file_types.extend([str.upper(x) for x in file_types])

        list_of_files = []

        for file_type in file_types:
            list_of_files.extend(glob.glob(os.path.join(os.path.abspath(source_directory), file_type)))

        return list_of_files
