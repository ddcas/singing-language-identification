import numpy as np
import argparse

import unittest

# test if resulting preprocessed segments are present and
# have the adequate length
class Testing(unittest.TestCase):
    def test_preprocessing_length(self, args):
        """
        This test assumes the preprocessed audio segments are
        contained in json files as text written from numpy arrays.
        The dict containing each list of arrays can be accessed
        with the key specified key contained in the variable 'segs_wav'
        """

        # construct the path to the data
        file_path = '{}/{}/{}'.format(
            args.parent_dir,
            args.folder_name,
            args.filename
        )
        # open file containing test data
        with open(file_path, 'r') as input_f:
            for i, line in enumerate(input_f):
                json_line = json.loads(line)
                song_segments = np.array(json_line['segs_wav'])
                # tests
                self.assertEqual(
                    type(song_segments[0]),
                    type(np.array([1., 2., 3.], dtype=np.float32))
                )
                self.assertEqual(
                    len(song_segments),
                    args.num_seg
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test preprocessing step')
    parser.add_argument('--parent_dir', type=str, default='datasets/size{ds_size}',
                        help='dataset parent directory (default: \'\')')
    parser.add_argument('--folder_name', type=str, default='',
                        help='folder containing the data (default: \'\')')
    parser.add_argument('--filename', type=str, default='',
                        help='filename/s to test (default: \'\')')
    parser.add_argument('--num_seg', type=int, default=5,
                        help='number of song segments extracted (default: 5)')

    cmd_args = parser.parse_args()
    unittest.main(cmd_args)
