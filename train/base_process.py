import logging


class BaseProcess:
    '''Base process for geneation/trainining/evaluation processes.'''

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def set_logger(cls, log_file: str):
        '''Set the logger to log info in terminal and file `log_path`.

        Args:
            log_path(str): Path to save log file.
        '''

        logger = logging.getLogger()  # return a logger which is the root logger of the hierarchy
        logger.setLevel(logging.INFO)  # sets the tracking threshold for this logger to `logging.INFO`

        if not logger.handlers:  # No defined Handlers. According to the above code, satisfy this condition.
            # Logging to a file
            file_handler = logging.FileHandler(
                log_file)  # instantiate `FileHandler` class. The specific file `log_file` is opened and used as the stream for logging
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s:%(levelname)s: %(message)s'))  # determine the formatting of logging message for final output to logs
            logger.addHandler(file_handler)

            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)
        logging.info('**** Start logging ****')

    def start(self):
        raise NotImplementedError('Method not implemented!')
#zmq
