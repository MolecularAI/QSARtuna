class NoRetrainingDataConvention(Exception):
	"""
	Raised if a file in input-directory does not follow the %Y-%m-%d convention
	"""
	def __init__(self, task, message="input-directory file [{0}] does not contain a date format %Y-%m-%d"):
		self.task = task
		self.message = message.format(self.task)
		super().__init__(self.message)

	pass


class NoNewRetrainingData(Exception):
	"""
	Raised if no new retraining data is available
	"""

	pass


class NoDifferingRetrainingData(Exception):
	"""
	Raised if no different retraining data is available between previous & current time bins
	"""

	pass


class RetrainingHeadersIssue(Exception):
	"""
	Raised when issue with retraining headers in a file (columns unknown)
	"""

	pass


class RetrainingIsAlreadyProcessed(Exception):
	"""
	Raised when retraining is processed
	"""

	def __init__(self, task, message="Retraining[{0}] already processed"):
		self.task = task
		self.message = message.format(self.task)
		super().__init__(self.message)


class RetrainingIsLocked(Exception):
	"""
	Raised when retraining is locked
	"""

	def __init__(
		self,
		task,
		message="Retraining[{0}] is locked",
	):
		self.task = task
		self.message = message.format(self.task)
		super().__init__(self.message)


class TemporalPredsPredicted(Exception):
	"""
	Raised when a temporal prediction is already predicted.
	"""

	def __init__(self, task, message="Retraining[{0}] code is predicted"):
		self.task = task
		self.message = message.format(self.task)
		super().__init__(self.message)


class NoPreviousModel(Exception):
	"""
	Raised when no previous model exists for a retraining point
	"""

	def __init__(self, prev_model_name, message="No previous model found for [{0}]"):
		self.prev_model_name = prev_model_name
		self.message = message.format(self.prev_model_name)
		super().__init__(self.message)


class SamePreviousModel(Exception):
	"""
	Raised when a temporal prediction would be for the same (identical) model training
	"""

	def __init__(self, task, message="Retraining[{0}] already processed"):
		self.task = task
		self.message = message.format(self.task)
		super().__init__(self.message)


class TimepointSkipped(Exception):
	"""
	Raised when a timepoint should be skipped
	"""

	def __init__(self, task, message="Retraining[{0}] set to be skipped"):
		self.task = task
		self.message = message.format(self.task)
		super().__init__(self.message)


class SlurmNoLog(Exception):
	"""
	Raised when a SLURM job file is not present for submitted itcode jobs
	"""

	pass


class SlurmTimeLimitExceeded(Exception):
	"""
	Raised when a past SLURM job time was exceeded
	"""

	pass


class SlurmMemoryExceeded(Exception):
	"""
	Raised when a past SLURM memory was exceeded
	"""

	pass


class SlurmJobSkip(Exception):
	"""
	Raised when an itcode is no longer trialed with QSARtuna (i.e. due to incompatibility)
	"""

	pass


class SlurmParseError(Exception):
	"""
	Raised when a .sh SLURM job modification (for resubmission) is not possible
	"""

	pass