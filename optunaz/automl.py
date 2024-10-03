import copy
import os
import re
import shutil

import numpy as np
import glob
import pandas as pd
import time
import logging
import logging.config
import argparse
from datetime import datetime, timedelta
import dataclasses
from pid.decorator import pidfile
import subprocess
from joblib import Parallel, delayed, effective_n_jobs
from optunaz.utils.retraining import *
from typing import Dict, List, Any
import pickle
import json
import pathlib

from optunaz.config import LOG_CONFIG

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelAutoML:
    """
    Prepares the data ready for the model training with ModelDispatcher.
    The ModelAutoML will also store activity for new tasks pending enough data.
    """

    def __init__(
        self,
        output_path: str = None,
        input_data: str = None,
        n_cores: int = -1,
        email: str = None,
        user_name: str = None,
        smiles_col: str = None,
        activity_col: str = None,
        task_col: str = None,
        dry_run: bool = False,
        timestr: str = time.strftime("%Y%m%d-%H%M%S"),
    ):
        self.retrain_timepoint = None
        self.new_data = None
        self.output_path = output_path
        self.input_data = input_data
        self.email = email
        self.user_name = user_name
        self.smiles_col = smiles_col
        self.activity_col = activity_col
        self.task_col = task_col
        self.dry_run = dry_run
        self.timestr = timestr
        self.n_cores = effective_n_jobs(n_cores)
        self.headers = [self.smiles_col, self.activity_col, self.task_col]

    @property
    def first_run(self) -> bool:
        if not os.path.exists(self.output_path):
            logging.debug(f"{self.output_path} does not exist, creating it")
            os.mkdir(self.output_path)
        if os.path.exists(f"{self.output_path}/processed_timepoints.json"):
            logging.debug(f"{self.output_path}/processed_timepoints.json exists")
            return False
        else:
            logging.debug(f"{self.output_path}/processed_timepoints.json not set")
            return True

    @property
    def processed_timepoints(self) -> Dict | List[None]:
        try:
            return json.load(open(f"{self.output_path}/processed_timepoints.json", "r"))
        except FileNotFoundError:
            return []

    @property
    def last_timepoint(self) -> str | List[None]:
        try:
            return self.processed_timepoints[-1]
        except IndexError:
            return []

    def getAllRetrainingData(self) -> Dict[datetime, str]:
        """
        Returns a dict of the wilcard data with converted datetime as the keys
        """
        fs = dict()
        glob_fs = glob.glob(self.input_data)
        if "*" in self.input_data:
            for glob_f in glob_fs:
                potential_dates = pathlib.Path(glob_f).stem.split(".")
                for potential_date in potential_dates:
                    try:
                        d = datetime.strptime(potential_date, "%Y-%m-%d")
                        fs[d] = glob_f
                        continue
                    except ValueError:
                        pass
                if glob_f not in fs.values():
                    raise NoRetrainingDataConvention(potential_dates)
        else:
            fs[
                datetime.fromtimestamp(os.path.getmtime(self.input_data))
            ] = self.input_data
        return fs

    def getRetrainingData(self) -> tuple[pd.DataFrame, str]:
        """
        Get data for the latest unprocessed date bucket or raise NoNewRetrainingData if none
        """
        fs = self.getAllRetrainingData()
        for ybin, thisf in sorted(fs.items()):
            process_ybin = datetime.strftime(ybin, "%y_%m_%d")
            if process_ybin in self.processed_timepoints:
                logging.debug(f"{process_ybin} is in processed_timepoints.json")
                continue
            try:
                task_data = pd.read_csv(
                    thisf,
                    low_memory=False,
                    encoding="latin",
                    on_bad_lines="skip",
                    usecols=self.headers,
                ).dropna()
            except PermissionError:
                logging.warning(f"{thisf} has PermissionError")
                continue
            except ValueError:
                avail_cols = pd.read_csv(thisf, nrows=0).columns
                miss_headers = [col for col in self.headers if col not in avail_cols]
                raise RetrainingHeadersIssue(process_ybin, miss_headers)
            task_data[self.activity_col] = pd.to_numeric(
                task_data[self.activity_col]
                .astype(str)
                .str.replace(">", "")
                .str.replace("<", ""),
                errors="coerce",
            ).astype(float)
            if len(task_data) == 0:
                logging.debug(f"{process_ybin} has no valid datapoints")
            return task_data.dropna(), process_ybin
        raise NoNewRetrainingData

    def setRetrainingData(self):
        """
        Sets the newest data bucket and timepoint for latest available data
        """
        new_data, retrain_timepoint = self.getRetrainingData()
        self.new_data = new_data
        self.retrain_timepoint = retrain_timepoint

    def initProcessedTimepoints(self):
        """
        Initialise the JSON containing timepoints for a first run
        """
        with open(f"{self.output_path}/processed_timepoints.json", "wt") as newf:
            json.dump([], newf, indent=4)
            logging.debug(
                f"Init first processed timepoint to: {self.output_path}/processed_timepoints.json"
            )

    def setProcessedTimepoints(self, problem=None):
        """
        Set the processed timepoints and the currently processing timepoint to JSON
        """
        if problem is not None:
            new_processed = list(self.processed_timepoints) + [problem]
            logging.debug(
                f"Appended problem timepoint {problem} to: {self.output_path}/processed_timepoints.json"
            )
        else:
            new_processed = list(self.processed_timepoints) + [self.retrain_timepoint]
            logging.debug(
                f"Appended processed timepoint {self.retrain_timepoint} to {self.output_path}/processed_timepoints.json"
            )
        with open(f"{self.output_path}/processed_timepoints.json", "wt") as newf:
            json.dump(new_processed, newf, indent=4)


@dataclasses.dataclass
class ModelDispatcher:
    """
    Use ModelAutoML config as a basis to prepare QSARtuna jobs, dispatching to SLURM.
    ModelDispatcher always needs a quorum to prepare the model
    """

    def __init__(
        self,
        quorum: int = None,
        cfg: ModelAutoML = None,
        last_timepoint: str = None,
        initial_template: str = None,
        retrain_template: str = None,
        slurm_template: str = None,
        slurm_req_cores: int = 1,
        slurm_req_partition: str = None,
        slurm_req_mem: int = None,
        slurm_al_pool: str = None,
        slurm_al_smiles: str = None,
        slurm_job_prefix: str = None,
        slurm_partition: str = None,
        save_previous_models: bool = None,
        log_conf: dict = None,
    ):
        self.taskcode = None
        self.taskcode_base = None
        self.taskcode_file = None
        self.temporal_preds = None
        self.temporal_file = None
        self.skip_file = None
        self.lock_file = None
        self.al_file = None
        self.latest_model = None
        self.meta_file = None
        self.prev_model_name = None
        self.json_name = None
        self.dataset_file = None
        self.quorum = quorum
        self.slurm_retry = None
        self.slurm_log = None
        self.slurm_name = None
        self.slurm_template = slurm_template
        self.slurm_job_prefix = slurm_job_prefix
        self.slurm_partition = slurm_partition
        self.slurm_req_cores = slurm_req_cores
        self.slurm_req_partition = slurm_req_partition
        self.slurm_req_mem = slurm_req_mem
        self.slurm_al_pool = slurm_al_pool
        self.slurm_al_smiles = slurm_al_smiles
        self.last_timepoint = last_timepoint
        self.initial_template = initial_template
        self.retrain_template = retrain_template
        self.save_previous_models = save_previous_models
        self.cfg = cfg
        self._pretrained_model = None
        self.log_conf = log_conf
        if log_conf is not None:
            logging.config.dictConfig(log_conf)

    @property
    def pretrained_model(self) -> Any:
        """
        Load a pretrained model
        """
        if not self._pretrained_model:
            try:
                with open(self.prev_model_name, "rb") as newf:
                    self._pretrained_model = pickle.load(newf)
                return self._pretrained_model
            except FileNotFoundError:
                raise NoPreviousModel(self.prev_model_name)
        return self._pretrained_model

    def checkIfRetrainingProcessed(self, taskcode):
        """
        Checks if this timepoint has already been predicted (and therefore processed).
        Timepoints to be skipped with data but no model quorum will also be in .skipped dirs.
        """
        if os.path.isfile(self.al_file):
            logging.debug(
                f"{self.cfg.retrain_timepoint}: Retraining [{taskcode}] is processed"
            )
            raise RetrainingIsAlreadyProcessed(taskcode)
        if self.checkSkipped():
            logging.debug(
                f"{self.cfg.retrain_timepoint}: Retraining [{taskcode}] is set to skipped"
            )
            raise TimepointSkipped(taskcode)

    def checkisLocked(self, taskcode):
        """
        Checks if this timepoint is locked for a given taskcode.
        Locks occur if QSARtuna is unable to run multiple retrain script instances run.
        """
        if os.path.isfile(self.lock_file):
            logging.debug(
                f"{self.cfg.retrain_timepoint}: Lockfile [{self.lock_file}] locks the taskcode [{taskcode}]"
            )
            raise RetrainingIsLocked(taskcode)
        else:
            logging.debug(
                f"{self.cfg.retrain_timepoint}: Lockfile [{self.lock_file}] not set; no lock for taskcode [{taskcode}]"
            )

    def checkRunningSlurmJobs(self) -> List[str]:
        if self.cfg.dry_run:
            logging.debug(f"Dry run of /usr/bin/squeue")
            return []
        command = f"/usr/bin/squeue --Format=name".split()
        p = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout.read()
        running_jobs = [
            i.split()[0][len(self.slurm_job_prefix) + 1 :]
            for i in str(p).split("\\n")[1:-1]
            if i[: len(self.slurm_job_prefix)] == self.slurm_job_prefix
        ]
        if len(running_jobs) >= 1:
            logging.info(f"Active/queued SLURM jobs are: {running_jobs}")
        else:
            logging.debug(f"No Active/queued SLURM jobs")
        return running_jobs

    @staticmethod
    def calcSlurmMem(len_file) -> int:
        """
        Dynamic resource allocation for memory from query
        """
        bins = [-np.inf, -1, 0] + list(np.arange(60000, 200000, 20000)) + [np.inf]
        req_mem = pd.cut([len_file], bins, right=True).codes.astype(int) * 30
        return req_mem[0]

    def setDispatcherVariables(self, taskcode):
        """
        Sets environment variables on a per taskcode level
        """
        taskcode = str(taskcode)
        self.taskcode = f"{taskcode}"
        self.taskcode_base = f"{self.cfg.output_path}/data/{taskcode}"
        self.taskcode_file = f"{self.taskcode_base}/{taskcode}"
        self.dataset_file = f"{self.taskcode_file}.csv"
        self.slurm_name = f"{self.taskcode_file}.sh"
        self.slurm_log = f"{self.taskcode_file}.out"
        self.slurm_retry = f"{self.taskcode_base}/.retry"
        self.json_name = f"{self.taskcode_file}.json"
        self.prev_model_name = f"{self.taskcode_file}.pkl"
        self.meta_file = f"{self.taskcode_file}_{self.cfg.retrain_timepoint}.meta"
        self.latest_model = f"{self.taskcode_base}/latest.pkl"
        self.al_file = f"{self.taskcode_file}_{self.cfg.retrain_timepoint}.al"
        self.lock_file = f"{self.taskcode_base}/.{self.cfg.retrain_timepoint}"
        self.skip_file = f"{self.taskcode_base}/.skip"
        self.temporal_file = "TEMPORALFILE"
        self.temporal_preds = "TEMPORALPREDS"

    def setJobLocked(self):
        """
        Creates lock file to ensure future runs do not overwrite pending jobs
        """
        if os.path.isfile(self.lock_file):
            logging.debug(
                f"{self.cfg.retrain_timepoint}: Lockfile [{self.lock_file}] is already locked"
            )
        else:
            pathlib.Path(f"{self.lock_file}").touch()
            logging.debug(f"lock_file for {self.lock_file} was set")

    def processTrain(self, _taskcode_df) -> pd.DataFrame:
        """
        Opens existing training if possible, formats data and attributes set for prev data
        If no retrain, create directory, returns new smiles & y for train
        """
        if os.path.exists(self.taskcode_base):
            try:
                this_df = pd.read_csv(self.dataset_file).dropna()
                this_df["automl_predefined_split"] = -1
                _taskcode_df["automl_predefined_split"] = 1
                this_df = pd.concat((this_df, _taskcode_df)).drop_duplicates(
                    subset=[self.cfg.smiles_col, self.cfg.activity_col], keep="first"
                )
                len_new = len(this_df.query("automl_predefined_split == 1"))
                if len_new == 0:
                    raise NoDifferingRetrainingData
                logging.debug(f"{self.taskcode}: {len_new} new data points found")
                return this_df
            except FileNotFoundError:
                pass
        _taskcode_df["automl_predefined_split"] = -1
        os.makedirs(self.taskcode_base, mode=0o777)
        return _taskcode_df

    def processQuorum(self, _input_df) -> bool:
        """
        Evaluates quorum & formats retraining data
        """
        means = (
            _input_df[[self.cfg.smiles_col, self.cfg.activity_col]]
            .groupby(self.cfg.smiles_col)
            .mean()
        )
        mad = (
            (means[self.cfg.activity_col] - means[self.cfg.activity_col].mean())
            .abs()
            .mean()
        )
        quorum = (len(means) >= self.quorum) and (mad > 0)
        return quorum

    def isTrained(self) -> bool:
        if os.path.exists(self.meta_file):
            logging.debug(f"{self.meta_file} exists")
            return True
        else:
            logging.debug(f"{self.meta_file} not present")
            return False

    def checkSaveTemporalModel(self):
        if self.save_previous_models:
            save_n = (
                f"{self.taskcode_file}_{self.pretrained_model.metadata['name']}.pkl"
            )
            shutil.copyfile(self.prev_model_name, save_n)
            logging.debug(f"Saved pretrained {self.taskcode_file} model to {save_n}")

    def doTemporalPredictions(self, new_data):
        """
        Start/check temporal (pseudo-prospective) predictions with an old QSARtuna model vs. newest data
        """
        if self.cfg.retrain_timepoint == self.pretrained_model.metadata["name"]:
            raise SamePreviousModel(self.taskcode)
        self.temporal_file = (
            f"{self.taskcode_file}_{self.pretrained_model.metadata['name']}__"
            f"{self.cfg.retrain_timepoint}.csv"
        )
        self.temporal_preds = f"{self.temporal_file}.preds"
        if os.path.exists(self.temporal_preds):
            raise TemporalPredsPredicted(self.taskcode)
        self.setJobLocked()
        new_data[[self.cfg.smiles_col, self.cfg.activity_col]].groupby(
            self.cfg.smiles_col
        ).median().to_csv(self.temporal_file)
        self.writeSlurm()
        if self.submitJob() != 0:
            logging.warning(f"Could not submit temporal SLURM job for {self.taskcode}")
        logging.debug(
            f"{self.taskcode}: {self.pretrained_model.metadata['name']} model used to "
            f"predict {len(new_data)} {self.cfg.retrain_timepoint} datapoints"
        )
        self.checkSaveTemporalModel()
        return

    def writeSlurm(self):
        """
        Writes a slurm job for a QSARtuna run for a given taskcode
        """
        with open(self.slurm_name, "w") as fileobj:
            with open(f"{self.slurm_template}", "r") as openFile:
                fileobj.write(
                    openFile.read()
                    .replace("NAME", f"{self.slurm_job_prefix}_{self.taskcode}")
                    .replace("TASK_FILE", f"{self.taskcode_file}")
                    .replace("METAFILE", f"{self.meta_file}")
                    .replace("AL_FILE", f"{self.al_file}")
                    .replace("EMAIL", f"{self.cfg.email}")
                    .replace("LOCK", f"{self.lock_file}")
                    .replace("RETRY", f"{self.slurm_retry}")
                    .replace("LATEST", f"{self.latest_model}")
                    .replace("MEM", f"{self.slurm_req_mem}")
                    .replace("CORES", f"{self.slurm_req_cores}")
                    .replace("PARTITION", f"{self.slurm_req_partition}")
                    .replace("AL_POOL", f"{self.slurm_al_pool}")
                    .replace("AL_SMILES", f"{self.slurm_al_smiles}")
                    .replace("SMILES", f"{self.cfg.smiles_col}")
                    .replace("TEMPORALFILE", f"{self.temporal_file}")
                    .replace("TEMPORALPREDS", f"{self.temporal_preds}")
                )
        logging.debug(f"wrote slurm to {self.slurm_name}")
        return

    def writeJson(self):
        """
        Writes a QSARtuna json for a given taskcode
        """
        if os.path.exists(f"{self.latest_model}"):
            template = f"{self.retrain_template}"
        else:
            template = f"{self.initial_template}"
        with open(f"{self.json_name}", "w") as fileobj:
            with open(template, "r") as openFile:
                fileobj.write(
                    openFile.read()
                    .replace("NAME", f"{self.cfg.retrain_timepoint}")
                    .replace("DATASET_FILE", self.dataset_file)
                    .replace("LATEST", self.latest_model)
                    .replace("SMILES", self.cfg.smiles_col)
                    .replace("ACTIVITY", self.cfg.activity_col)
                )
        logging.debug(f"wrote json to {self.json_name}")
        return

    def writeDataset(self, out_df):
        """
        Writes the training datapoints to file
        """
        out_df.to_csv(self.dataset_file, index=False)
        logging.debug(f"wrote dataset to {self.dataset_file}")

    def setSkippedTimepoint(self):
        """
        Annotate the timepoint as not eligable for a taskcode
        """
        try:
            skipped_timepoints = json.load(open(f"{self.skip_file}", "r"))
        except FileNotFoundError:
            skipped_timepoints = []
            logging.debug(f"skip file {self.skip_file} will be created")

        with open(f"{self.skip_file}", "wt") as newf:
            json.dump(skipped_timepoints + [self.cfg.retrain_timepoint], newf, indent=4)
            logging.debug(f"{self.cfg.retrain_timepoint} added to {self.skip_file}")

    def checkSkipped(self):
        try:
            skipped_timepoints = json.load(open(f"{self.skip_file}", "r"))
        except FileNotFoundError:
            logging.debug(f"{self.skip_file} not present")
            return False
        except json.decoder.JSONDecodeError:
            logging.debug(f"{self.skip_file} error")
        is_skipped = self.cfg.retrain_timepoint in skipped_timepoints
        if is_skipped:
            logging.debug(
                f"Timepoint {self.cfg.retrain_timepoint} is in {self.skip_file}"
            )
        return is_skipped

    def submitJob(self):
        if not self.cfg.dry_run:
            sbatch = subprocess.run(
                f"/usr/bin/sbatch {self.slurm_name}",
                shell=True,
                stdout=subprocess.PIPE,
            )
            logging.debug(f"SLURM output: {sbatch}")
            if sbatch.returncode == 0:
                logging.debug(f"{self.slurm_name} submitted")
            return sbatch.returncode
        else:
            logging.debug(f"Dry run of /usr/bin/sbatch {self.slurm_name}")
            return 0

    def checkSlurmStatusAndNextProcedure(self):
        """
        Check a SLURM job completed with no cancellations
        """
        try:
            slurm_log = open(self.slurm_log).read()
            if "DUE TO TIME LIMIT" in slurm_log:
                logging.debug(f"{self.slurm_name} time limit was reached")
                raise SlurmTimeLimitExceeded
            elif any(
                [
                    err in slurm_log
                    for err in [
                        "DUE TO MEMORY",
                        "Bus error",
                        "Unable to allocate",
                        "oom_kill",
                        "OOM Killed",
                    ]
                ]
            ):
                logging.debug(f"{self.slurm_name} memory was reached")
                raise SlurmMemoryExceeded
            elif "func_code.py" in slurm_log:
                logging.debug(f"{self.slurm_name} had func_code.py error")
            elif "numpy.ComplexWarning" in slurm_log:
                logging.debug(f"{self.slurm_name} had numpy.ComplexWarning")
            elif "ValueError: Exiting since no trials returned values" in slurm_log:
                logging.debug(f"{self.slurm_name} had no valid trials")
                raise SlurmJobSkip
            elif "Adjust any of the aforementioned parameters" in slurm_log:
                logging.debug(f"{self.slurm_name} had splitting error")
                raise SlurmJobSkip
            elif "qptuna.predict.UncertaintyError" in slurm_log:
                logging.debug(f"{self.slurm_name} does not support uncertainty estimation")
                raise SlurmJobSkip
        except FileNotFoundError:
            raise SlurmNoLog

    def increaseJobTime(self, minutes):
        """
        Increase SLURM model time
        """
        job_sh = open(self.slurm_name).read().splitlines()
        mins = timedelta(minutes=minutes)
        for l_idx, line in enumerate(job_sh):
            if "--time" in line:
                try:
                    line = line.split("=")
                    old_time = line[-1]
                    job_time = datetime.strptime(line[-1], "%d-%H:%M")
                    line[-1] = (job_time + mins).strftime("%d-%H:%M")
                    if line[-1][0] == "0":
                        line[-1] = line[-1][1:]
                    logging.info(
                        f"{self.slurm_name} increased time by [{minutes}] from [{old_time}] to [{line[-1]}]"
                    )
                    job_sh[l_idx] = "=".join(line)
                except (ValueError, TypeError):
                    logging.warning(f"Unable to increase [{self.slurm_name}] job time")
                    raise SlurmParseError
        with open(self.slurm_name, "w") as fileobj:
            for line in job_sh:
                fileobj.write(f"{line}\n")

    def increaseJobMem(self, mem, max_mem=200):
        """
        Increase SLURM model memory
        """
        job_sh = open(self.slurm_name).read().splitlines()
        for l_idx, line in enumerate(job_sh):
            if "--mem" in line:
                line = re.split("(\d+)", line)
                try:
                    old_mem = line[1]
                    new_mem = int(line[1]) + mem
                    if new_mem >= max_mem:
                        logging.warning(
                            f"{self.slurm_name}] new mem [{new_mem}]G  >= max: [{max_mem}]G"
                        )
                        raise SlurmParseError
                    line[1] = str(new_mem)
                    logging.info(
                        f"{self.slurm_name} increasing mem by [{mem}G] from [{old_mem}G] to [{new_mem}G]"
                    )
                    job_sh[l_idx] = "".join(line)
                except (ValueError, TypeError) as e:
                    logging.warning(
                        f"Unable to increase [{self.slurm_name}] memory: {e}"
                    )
                    raise SlurmParseError
        with open(self.slurm_name, "w") as fileobj:
            for line in job_sh:
                fileobj.write(f"{line}\n")

    def increaseJobCpu(self, cpu, max_cpu=20):
        """
        Increase SLURM model cpu
        """
        job_sh = open(self.slurm_name).read().splitlines()
        for l_idx, line in enumerate(job_sh):
            if "#SBATCH -c " in line:
                line = re.split("(\d+)", line)
                try:
                    old_cpu = line[1]
                    new_cpu = int(line[1]) + cpu
                    if new_cpu >= max_cpu:
                        logging.warning(
                            f"{self.slurm_name}] new cpu [{new_cpu}]  >= max: [{max_cpu}]"
                        )
                        return
                    line[1] = str(new_cpu)
                    logging.info(
                        f"{self.slurm_name} increasing cpu by [{cpu}] from [{old_cpu}] to [{new_cpu}]"
                    )
                    job_sh[l_idx] = "".join(line)
                except (ValueError, TypeError) as e:
                    logging.warning(f"Unable to increase [{self.slurm_name}] cpu: {e}")
                    raise SlurmParseError
        with open(self.slurm_name, "w") as fileobj:
            for line in job_sh:
                fileobj.write(f"{line}\n")

    def addSlurmRetry(self):
        try:
            (pd.read_csv(f"{self.slurm_retry}") + 1).to_csv(
                f"{self.slurm_retry}", index=False
            )
        except FileNotFoundError:
            pd.DataFrame(data=[{"retry": 1}]).to_csv(f"{self.slurm_retry}", index=False)

    def getSlurmRetry(self):
        try:
            return pd.read_csv(f"{self.slurm_retry}").loc[0][0]
        except FileNotFoundError:
            return 0

    def resubmitAnyFailedJobs(
        self,
        locked_jobs,
        minutes=720,
        mem=20,
        cpu=4,
        max_retries=5,
        max_mem=200,
        max_cpu=20,
    ):
        """
        Resubmit failed jobs, according to reason for failure
        """
        running_jobs = self.checkRunningSlurmJobs()
        resubmitted = []
        failed_submission = []
        for job in locked_jobs:
            if job not in running_jobs:
                self.setDispatcherVariables(job)
                retrys = self.getSlurmRetry()
                if retrys > max_retries:
                    logging.warning(f"{self.slurm_name} had too many retries {retrys}")
                    self.setSkippedTimepoint()
                    continue
                try:
                    try:
                        self.checkSlurmStatusAndNextProcedure()
                    # Problematic jobs that will always fail are skipped
                    except SlurmJobSkip:
                        self.setSkippedTimepoint()
                        continue
                    # Time limited jobs are extended (inc. memory since swap may slow job)
                    except SlurmTimeLimitExceeded:
                        self.increaseJobTime(minutes)
                        self.increaseJobMem(mem, max_mem=max_mem)
                        self.increaseJobCpu(cpu, max_cpu=max_cpu)
                    # Memory limited jobs are increased
                    except SlurmMemoryExceeded:
                        self.increaseJobTime(minutes)
                        self.increaseJobMem(mem, max_mem=max_mem)
                    # Submit jobs that failed submission (maybe a sbatch glitch?)
                    except SlurmNoLog:
                        logging.warning(
                            f"{self.slurm_name} never ran, so will be resubmit"
                        )
                    # If no detectable reason, then a requeue is still issued
                    else:
                        logging.warning(
                            f"{self.slurm_name} had log but no detected abort/failure reason"
                        )
                        self.increaseJobTime(minutes)
                        self.increaseJobMem(mem, max_mem=max_mem)
                        self.increaseJobCpu(cpu, max_cpu=max_cpu)
                    self.addSlurmRetry()
                    if self.submitJob() == 0:
                        resubmitted.append(job)
                    else:
                        raise SlurmParseError
                    logging.info(f"{self.slurm_name} resubmit ({retrys} retrys)")
                except SlurmParseError:
                    self.addSlurmRetry()
                    failed_submission.append(job)
                    logging.warning(
                        f"{self.slurm_name} failed resubmission ({retrys} retrys)"
                    )
            else:
                logging.debug(f"{job} still running/queued")
        if len(resubmitted) >= 1:
            logging.info(f"Some jobs were resubmitted: {resubmitted}")
        if len(failed_submission) >= 1:
            logging.info(f"Some jobs failed resubmission: {failed_submission}")

    def processRetraining(self, taskcode):
        """
        Enumerates through new data, creating the latest files and models
        """
        out_df = self.cfg.new_data.loc[
            self.cfg.new_data[self.cfg.task_col] == taskcode
        ].dropna()
        # set variable names each iteration
        self.setDispatcherVariables(taskcode)
        # do basic checks
        try:
            self.checkIfRetrainingProcessed(taskcode)
            self.checkisLocked(taskcode)
        except (RetrainingIsAlreadyProcessed, TimepointSkipped):
            return {}
        except RetrainingIsLocked:
            return {"Locked": taskcode}

        # add preexisiting bioactivites if possible, and process training
        try:
            out_df = self.processTrain(out_df)
        # skip taskcode if latest dataset adds no new data (could be duplicated csv)
        except NoDifferingRetrainingData:
            # handle here that there appears to be no differing data for first timepoint
            if not self.last_timepoint:
                logging.debug(f"{self.taskcode}: Fist timepoint")
                return {}
            if self.last_timepoint != self.cfg.retrain_timepoint:
                if self.isTrained():
                    logging.debug(f"{self.taskcode}: Retraining trained")
                else:
                    logging.debug(f"{self.taskcode}: No new data (or all duplicates)")
                    self.setSkippedTimepoint()
                return {}

        if self.last_timepoint == self.cfg.retrain_timepoint:
            logging.warning(
                f"{self.taskcode}: Something went wrong: {self.last_timepoint} == {self.cfg.retrain_timepoint}"
            )
            return {}
        if self.processQuorum(out_df):
            if self.slurm_req_mem is None:
                self.slurm_req_mem = self.calcSlurmMem(len(out_df))  # query sets mem
                logging.debug(
                    f"{self.taskcode}: Dynamic resource allocation mem: {self.slurm_req_mem}G"
                )
            else:
                logging.debug(
                    f"{self.taskcode}: Manual resource allocation mem: {self.slurm_req_mem}G"
                )
            try:
                # generate (and write) predictions of old model on new data
                self.doTemporalPredictions(out_df.query("automl_predefined_split == 1"))
                return {"Working": taskcode}
            except (NoPreviousModel, TemporalPredsPredicted, SamePreviousModel) as e:
                logging.debug(
                    f"{self.taskcode}: {self.cfg.retrain_timepoint}: No temporal predictions since [{e}]"
                )
            # write files for dispatch, lock & dispatch to slurm
            self.writeDataset(out_df)
            self.writeSlurm()
            self.writeJson()
            self.setJobLocked()
            if self.submitJob() != 0:
                logging.warning(
                    f"{self.taskcode}: Could not submit SLURM job for {taskcode}"
                )
            return {"Working": taskcode}
        # not quorum so write pred lock
        else:
            logging.debug(f"{self.taskcode}: {self.cfg.retrain_timepoint}: Not quorum")
            self.writeDataset(out_df)
            self.setSkippedTimepoint()
        return {}


def process_retraining_task(taskcode, dispatcher):
    _dispatcher = copy.deepcopy(dispatcher)
    return _dispatcher.processRetraining(taskcode)


def dispatcher_process(global_cfg, args, dispatcher):
    work = sorted(global_cfg.new_data[args.input_task_csv_column].unique())
    if global_cfg.n_cores > 1:
        results = Parallel(n_jobs=global_cfg.n_cores * 2, backend="threading")(
            delayed(process_retraining_task)(w, dispatcher) for w in work
        )
    else:
        # Process tasks sequentially
        results = [copy.deepcopy(dispatcher.processRetraining(w)) for w in work]
    return results


@pidfile(piddir="./")
def meta():
    """
    Tracks temporal performance of QSARtuna models by writing the metadata to JSON files
    """
    parser = argparse.ArgumentParser(
        description="AutoML output performance of temporal models"
    )

    # fmt: off
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--pkl", type=str, help="Path to the output QSARtuna PKL file", required=True)
    requiredNamed.add_argument("--meta", type=str, help="Path to the output metadata file", required=True)
    # fmt: off
    args, leftovers = parser.parse_known_args()

    with open(args.pkl, "rb") as fid:
        prev_model = pickle.load(fid)
        metadata = prev_model.metadata

    with open(args.meta, "wt") as f:
        json.dump(metadata, f, indent=4)


def validate_args(args):
    assert os.path.isfile(
        args.slurm_al_pool
    ), f"AL pool {args.slurm_al_pool} provide '--slurm-al-pool' with a valid file"
    assert os.path.isfile(
        args.input_initial_template
    ), f"Initial template {args.input_initial_template} provide '--initial-template' with a valid file"
    assert os.path.isfile(
        args.input_retrain_template
    ), f"Retraining template {args.input_retrain_template} provide '--retrain-template' with a valid file"
    assert args.quorum >= 25, f"Quorum should be >=25, got {args.quorum}"


def prepare_dispatcher(global_cfg, args, log_conf):
    dispatcher = ModelDispatcher(
        cfg=global_cfg,
        quorum=args.quorum,
        slurm_job_prefix=args.slurm_job_prefix,
        last_timepoint=global_cfg.last_timepoint,
        initial_template=args.input_initial_template,
        retrain_template=args.input_retrain_template,
        slurm_template=args.input_slurm_template,
        slurm_req_mem=args.slurm_req_mem,
        slurm_req_partition=args.slurm_req_partition,
        slurm_req_cores=args.slurm_req_cores,
        slurm_al_pool=args.slurm_al_pool,
        slurm_al_smiles=args.slurm_al_smiles_csv_column,
        save_previous_models=args.save_previous_models,
        log_conf=log_conf,
    )
    return dispatcher


@pidfile(piddir="./")
def main():
    start = time.time()
    parser = argparse.ArgumentParser(
        description="AutoML scheduling for temporal automatic retraining of QSARtuna models"
    )

    # fmt: off
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--output-path", type=str, help="Path to the output AutoML directory", required=True)
    requiredNamed.add_argument("--email", type=str, help="Email for SLURM job notifications", required=True)
    requiredNamed.add_argument("--user_name", type=str, help="HPC 'username' for the AutoML user", required=True)

    # Input file variables
    requiredNamed.add_argument("--input-data", type=str, help="Name of the input file[s]. For multiple files use '*' in wildcard expression", required=True)
    requiredNamed.add_argument("--input-smiles-csv-column", type=str, help="Column name of SMILES column in csv file", required=True)
    requiredNamed.add_argument("--input-activity-csv-column", type=str, help="Column name of activity column in data file", required=True)
    requiredNamed.add_argument("--input-task-csv-column", type=str, help="Column name of task column in data file", required=True)
    requiredNamed.add_argument('--input-initial-template', type=str, required=True)
    requiredNamed.add_argument('--input-retrain-template', type=str, required=True)
    requiredNamed.add_argument('--input-slurm-template', type=str, required=True)

    # Non-required AutoML variables
    parser.add_argument("--quorum", type=int, default=25)
    parser.add_argument("--n-cores", type=int, default=-1) # No. cores for this pipeline, not SLURM
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument('-v', '--verbose', action='count', default=0)

    # SLURM global variables
    parser.add_argument('--slurm-req-cores', type=int, default=12)
    parser.add_argument('--slurm-req-mem', type=int, default=None) # By default, None = dynamic mem resource allocation
    requiredNamed.add_argument('--slurm-req-partition', type=str, required=True)
    requiredNamed.add_argument('--slurm-al-pool', type=str, required=True)
    requiredNamed.add_argument('--slurm-al-smiles-csv-column', type=str, required=True)

    # dispatcher variables
    requiredNamed.add_argument('--slurm-job-prefix', type=str, required=True)
    parser.add_argument('--slurm-failure-cores-increment', type=int, default=4)
    parser.add_argument('--slurm-failure-mem-increment', type=int, default=20)
    parser.add_argument('--slurm-failure-mins-increment', type=int, default=720)
    parser.add_argument('--slurm-failure-max-retries', type=int, default=5)
    parser.add_argument('--slurm-failure-max-mem', type=int, default=200)
    parser.add_argument('--slurm-failure-max-cpu', type=int, default=20)
    parser.add_argument('--save-previous-models', action="store_true")
    # fmt: on

    args, leftovers = parser.parse_known_args()
    log_conf = LOG_CONFIG
    match args.verbose:
        case 0:
            stdout = logging.WARNING
            stderr = logging.CRITICAL
        case 1:
            stdout = logging.INFO
            stderr = logging.WARNING
        case _:
            stdout = logging.DEBUG
            stderr = logging.WARNING
    log_conf["handlers"]["stdout_handler"]["level"] = stdout
    log_conf["handlers"]["stderr_handler"]["level"] = stderr
    logging.config.dictConfig(log_conf)

    logging.info(args)

    validate_args(args)

    global_cfg = ModelAutoML(
        output_path=args.output_path,
        input_data=args.input_data,
        email=args.email,
        user_name=args.user_name,
        n_cores=args.n_cores,
        dry_run=args.dry_run,
        smiles_col=args.input_smiles_csv_column,
        activity_col=args.input_activity_csv_column,
        task_col=args.input_task_csv_column,
    )

    while True:
        try:
            global_cfg.setRetrainingData()
        except NoNewRetrainingData:
            logging.debug("NoNewRetrainingData, so exiting")
            return
        except RetrainingHeadersIssue as e:
            logging.warning(
                f"Work not possible for timepoint {e.args[0]} due missing header[s] {e.args[1]}"
            )
            global_cfg.setProcessedTimepoints(problem=e.args[0])
            continue

        logging.debug(f"Processing timepoint {global_cfg.retrain_timepoint}")
        dispatcher = prepare_dispatcher(global_cfg, args, log_conf)
        if global_cfg.first_run:
            global_cfg.initProcessedTimepoints()
        results = pd.DataFrame(dispatcher_process(global_cfg, args, dispatcher))
        if "Working" in results.columns:
            logging.info("Exiting at this timepoint since there is work to do")
            logging.debug(f"Work: {results.Working.dropna().tolist()}")
            end = time.time()
            logging.info(f"AutoML script took [{end - start:.08}] seconds.")
            return
        if "Locked" in results.columns:
            dispatcher = prepare_dispatcher(global_cfg, args, log_conf)
            dispatcher.resubmitAnyFailedJobs(
                results.Locked.dropna().tolist(),
                mem=args.slurm_failure_mem_increment,
                minutes=args.slurm_failure_mins_increment,
                cpu=args.slurm_failure_cores_increment,
                max_retries=args.slurm_failure_max_retries,
                max_mem=args.slurm_failure_max_mem,
                max_cpu=args.slurm_failure_max_cpu,
            )
            logging.info(
                f"Exiting: {global_cfg.retrain_timepoint} lock(s) indicate(s) work ongoing"
            )
            end = time.time()
            logging.info(f"AutoML script took [{end - start:.08}] seconds.")
            return
        else:
            logging.info(
                f"Work appears complete for timepoint {global_cfg.retrain_timepoint}"
            )
            global_cfg.setProcessedTimepoints()
