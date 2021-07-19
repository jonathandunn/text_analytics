from tqdm import tqdm
from urllib.parse import urlparse
import pandas as pd
import requests
import os
import boto3
import hashlib

try:
    from settings import Settings
except:
    from .settings import Settings
    
try:
    from serializers import SERIALIZERS
    from serializers import *
except:
    from .serializers import SERIALIZERS
    from .serializers import *
    

#Initialize settings module
settings = Settings()

class ExternalFileLoader(object):
    """
    Manages external Files for text_analytics.
    If project is running outside Jupyter Notebook gathers files on demand.
    """
    base_url = None
    data_dir = None

    def __init__(self, **kwargs):
        self.data_url = settings.DATA_BUCKET
        self.state_url = settings.STATE_BUCKET
        self.data_dir = self._get_dir(kwargs.get('data_dir') if kwargs.get('data_dir') else DATA_DIR)
        self.state_dir = self._get_dir(kwargs.get('states_dir') if kwargs.get('states_dir') else STATES_DIR)

        self.checksum_base_url = settings.CHECKSUM_BASE_URL

    @staticmethod
    def _get_dir(directory):
        """
        Creates data dir
        :param directory:
        :return:
        """
        if not os.path.exists(directory):
            # print('{} Dir does not exist. Creating it...'.format(directory))
            try:
                os.mkdir(directory)
            except PermissionError:
                # print('Something went wrong, when creating dir {}, check privileges.'.format(directory))
                raise
        return directory

    def build_url(self, external_file_name, file_type):  # pragma: no cover
        """
        Gets file link for s3 given a file type
        :param external_file_name:
        :param file_type:
        :return:
        """
        return '/'.join((getattr(self, "{}_url".format(file_type)), external_file_name))

    def _load_state(self, filename, file_type=None, state_type=None):
        """
        Reads File from path and serializes it based on state_type
        :param filename:
        :param file_type:
        :param state_type:
        :return:
        """
        path = self.get_file_path(filename, file_type=file_type)
        with open(path, "rb") as fo:
            data = fo.read()
        return SERIALIZERS[state_type](data).deserealize()

    def _load_data(self, filename, file_type=None, **kwargs):
        """
        Loads pandas csv file from disk.
        :param filename:
        :param file_type:
        :param kwargs:
        :return:
        """
        path = self.get_file_path(filename, file_type=file_type)
        df = pd.read_csv(path, index_col=0)
        return df

    def _get_file(self, filename, file_type=None, state_type=None):
        """
        Get file wrapper for states and data.
        :param filename:
        :param file_type:
        :param state_type:
        :return:
        """
        if not self.file_exists(filename, file_type=file_type):
            self.download_file(filename, file_type=file_type)
        return getattr(self, '_load_{}'.format(file_type))(filename, file_type=file_type, state_type=state_type)

    def get_corpus(self, filename):
        """
        Wrapper for _get_file , just for copus files.
        :param filename:
        :return:
        """
        return self._get_file(filename, file_type='data')

    def get_state(self, filename, state_type=None):
        """
        Wrapper for _get_file for state files, state_type must be a valid serializer class type.
        :param filename:
        :param state_type:
        :return:
        """
        if not state_type:
            raise Exception('Please provide one of the following types: {}'.format([key for key in SERIALIZERS.keys()]))
        return self._get_file(filename, file_type='state', state_type=state_type)

    def download_file(self, external_file_name, file_type=None):
        """
        Attempts to download file from S3 bucket.
        :param external_file_name:
        :param file_type:
        :return:
        """
        resp = requests.get(self.build_url(external_file_name, file_type), stream=True)
        if 'AccessDenied' in resp.text:
            raise Exception('File not found')
        total = int(resp.headers.get('content-length', 0))
        file_path = self.get_file_path(external_file_name, file_type=file_type)
        with open(file_path, 'wb') as file, tqdm(
                desc=external_file_name,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        if not self._validate_file(external_file_name, file_type=file_type):
            os.remove(file_path)
            raise Exception('Could not load file.')

    def get_file_path(self, local_file_path, file_type=None):
        """
        Gets local file path for given file type
        :param local_file_path:
        :param file_type:
        :return:
        """
        path = getattr(self, "{}_dir".format(file_type))
        return os.path.join(path, local_file_path)

    def file_exists(self, filename, file_type=None):
        """
         Checks if file exists in filesystem
        :param filename:
        :param file_type:
        :return:
        """
        return os.path.isfile(self.get_file_path(filename, file_type=file_type))

    def save_file(self, filename, file_contents, file_type=None):
        """Saves file to disk"""
        path = self.get_file_path(filename, file_type=file_type)
        with open(path, "wb") as f:
            f.write(file_contents)

    def _get_file_checksum(self, filename, file_type=None):
        """
        Gets file checksum for later integrity checks.
        :param filename:
        :param file_type:
        :return:
        """
        path = self.get_file_path(filename, file_type=file_type)
        with open(path, 'rb') as file_to_check:
            data = file_to_check.read()
            checksum = hashlib.md5(data).hexdigest()
        return checksum

    def _validate_file(self, filename, file_type=None):
        """
        Checks file integrity through aws-lambda function
        :param filename:
        :param file_type:
        :return:
        """
        if file_type == 'data':
            return True

        checksum = self._get_file_checksum(filename, file_type=file_type)
        return self._validate_checksum(filename, checksum)

    def _validate_checksum(self, filename, checksum):
        """
        Validates integrity
        :param filename:
        :param checksum:
        :return:
        """
        try:
            self._checksum_request(filename, checksum, request_type='check_hmac')
        except Exception:
            return False
        return True

    def upload_data(self, filename):
        """
        Wrapper for _upload_file for data files.
        :param filename:
        :return:
        """
        return self._upload_file(filename, file_type='data')

    def upload_state(self, filename):
        """
        Wrapper for _upload_file for state files.
        :param filename:
        :return:
        """
        return self._upload_file(filename, file_type='state', encrypt=True)

    def _get_checksum_url(self, action):
        """
        Gets checksum url for given action
        :param action:
        :return:
        """
        return '/'.join((self.checksum_base_url, action))

    def _checksum_request(self, filename, checksum, request_type=None):
        """
        Makes aws-lambda request.
        :param filename:
        :param checksum:
        :param request_type:
        :return:
        """
        payload = {"filename": filename,
                   "hmac": checksum}
        req = requests.post(self._get_checksum_url(request_type), json=payload)
        if req.status_code == 200:
            return req.json()
        else:
            raise Exception('Unable to validate file encryption')

    def _get_bucket(self, file_type):
        """
        Gets bucket name based on file type
        :param file_type:
        :return:
        """
        url = getattr(self, '{}_url'.format(file_type))
        bucket = urlparse(url, allow_fragments=False).netloc.split('.')[0]
        return bucket

    def _upload_file(self, filename, file_type=None, encrypt=False):
        """
        Upload files to S3 bucket. Encripts and saves checksum in case of a state type file.
        :param filename:
        :param file_type:
        :param encrypt:
        :return:
        """
        if encrypt:
            checksum = self._get_file_checksum(filename, file_type=file_type)
            if not checksum:
                raise Exception('Could not get CHECKSUM for file {}'.format(filename))
            self._checksum_request(filename, checksum, 'create_hmac')

        client = boto3.client('s3')

        response = client.put_object(
            Body=self.get_file_path(filename, file_type),
            Bucket=self._get_bucket(file_type),
            Key=filename,
        )
        return response
