from text_analytics.loader import ExternalFileLoader
from text_analytics.settings import *
from pandas.core.frame import DataFrame
from gensim.models.phrases import FrozenPhrases
from unittest import mock
import unittest
import shutil


class ExternalFilePath(unittest.TestCase):
    def setUp(self):
        self.loader = ExternalFileLoader()
        shutil.copy('tests/test_file.csv', 'states/test_file2.csv')

    def test_get_dir(self):
        directory = 'test'
        self.loader._get_dir(directory)
        self.assertTrue(os.path.exists(directory))
        os.removedirs(directory)

    @mock.patch('os.mkdir')
    def test_get_dir_permission_error(self, mock_open):
        mock_open.side_effect = PermissionError
        directory = 'test'
        with self.assertRaises(PermissionError):
            self.loader._get_dir(directory)

    def test_get_file_path_data(self):
        filename = 'test.csv'
        path = self.loader.get_file_path(filename, 'data')
        self.assertEqual(path, os.path.join(DATA_DIR, filename))

    def test_get_file_path_state(self):
        filename = 'test.csv'
        path = self.loader.get_file_path(filename, 'state')
        self.assertEqual(path, os.path.join(STATES_DIR, filename))

    def test_get_corpus(self):
        filename = 'corruption.csv'
        data = self.loader.get_corpus(filename)
        self.assertIsInstance(data, DataFrame)
        data = self.loader.get_corpus(filename)
        self.assertIsInstance(data, DataFrame)

    def test_get_state(self):
        filename = 'AI.State.Combined.phrases.pickle'
        data = self.loader.get_state(filename)
        self.assertIsInstance(data, FrozenPhrases)


    def tearDown(self):
        shutil.rmtree('states')
        shutil.rmtree('data')
        pass
