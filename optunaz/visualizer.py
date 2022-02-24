import os

import optuna
import warnings
from optuna import storages
from copy import deepcopy

from optuna.trial import TrialState, FrozenTrial

from optunaz.config.optconfig import OptimizationConfig
from optunaz.utils import mkdict
from optunaz.utils.enums.optimization_configuration_enum import OptimizationConfigurationEnum
from optunaz.utils.enums.visualization_enum import VisualizationEnum


class Visualizer:
    """Class to visualize various aspects of the optimization / building process."""

    def __init__(self):
        # initialize Enums
        self._OE = OptimizationConfigurationEnum()
        self._VE = VisualizationEnum()

    def plot_by_configuration(self, conf: OptimizationConfig, study: optuna.Study):
        vis_dict = mkdict(conf)[self._VE.VISUALIZATION]

        output_folder = vis_dict[self._VE.VISUALIZATION_OUTPUT_FOLDER]
        self._make_folder(output_folder)
        if self._VE.VISUALIZATION_PLOTS_HISTORY in vis_dict[self._VE.VISUALIZATION_PLOTS].keys() and \
           vis_dict[self._VE.VISUALIZATION_PLOTS][self._VE.VISUALIZATION_PLOTS_HISTORY] is True:
            file_path = os.path.join(output_folder, '.'.join(["history",
                                                              vis_dict[self._VE.VISUALIZATION_FILE_FORMAT]]))
            self.plot_history(file_path=file_path,
                              study=study)
        if self._VE.VISUALIZATION_PLOTS_CONTOUR in vis_dict[self._VE.VISUALIZATION_PLOTS].keys() and \
           vis_dict[self._VE.VISUALIZATION_PLOTS][self._VE.VISUALIZATION_PLOTS_CONTOUR] is True:
            contour_folder = os.path.join(output_folder, "contour")
            self._make_folder(contour_folder)
            self.plot_contour(folder_path=contour_folder,
                              study=study,
                              file_format=vis_dict[self._VE.VISUALIZATION_FILE_FORMAT])
        if self._VE.VISUALIZATION_PLOTS_PARALLEL_COORDINATE in vis_dict[self._VE.VISUALIZATION_PLOTS].keys() and \
           vis_dict[self._VE.VISUALIZATION_PLOTS][self._VE.VISUALIZATION_PLOTS_PARALLEL_COORDINATE] is True:
            para_coord_folder = os.path.join(output_folder, "parallel_coordinates")
            self._make_folder(para_coord_folder)
            self.plot_parallel_coordinate(folder_path=para_coord_folder,
                                          study=study,
                                          file_format=vis_dict[self._VE.VISUALIZATION_FILE_FORMAT])
        if self._VE.VISUALIZATION_PLOTS_SLICE in vis_dict[self._VE.VISUALIZATION_PLOTS].keys() and \
           vis_dict[self._VE.VISUALIZATION_PLOTS][self._VE.VISUALIZATION_PLOTS_SLICE] is True:
            slice_folder = os.path.join(output_folder, "slice")
            self._make_folder(slice_folder)
            self.plot_slice(folder_path=slice_folder,
                            study=study,
                            file_format=vis_dict[self._VE.VISUALIZATION_FILE_FORMAT])

    def plot_slice(self, folder_path: str, study: optuna.Study, file_format="png"):
        # formats "png" and "jpeg" are handled inside the "write_image()" function of "plotly" / "orca"
        try:
            studies_list = self._split_study_by_algorithm(study=study)
            for sub_study in studies_list:
                file_path = os.path.join(folder_path, "".join([sub_study.study_name, '.', file_format]))
                fig = optuna.visualization._get_slice_plot(study=sub_study)
                fig.update_layout(title_text=sub_study.study_name)
                fig.write_image(file_path,
                                scale=3.25,
                                width=None,
                                height=None)
        except:
            warnings.warn("Orca could not find an X11 interface, plotting disabled.")

    def plot_parallel_coordinate(self, folder_path: str, study: optuna.Study, file_format="png"):
        try:
            studies_list = self._split_study_by_algorithm(study=study)
            for sub_study in studies_list:
                file_path = os.path.join(folder_path, "".join([sub_study.study_name, '.', file_format]))
                fig = optuna.visualization._get_parallel_coordinate_plot(study=sub_study)
                fig.update_layout(title_text=sub_study.study_name)
                fig.write_image(file_path,
                                scale=6.75,
                                width=None,
                                height=None)
        except:
            warnings.warn("Orca could not find an X11 interface, plotting disabled.")

    def plot_contour(self, folder_path: str, study: optuna.Study, file_format="png"):
        try:
            studies_list = self._split_study_by_algorithm(study=study)
            for sub_study in studies_list:
                # as this is a two-dimensional plot, disable it for all algorithms that have less than 2 hyper-parameters
                # note, that "study_type" has been removed by "_split_study_by_algorithm()", so only 'real' hyper-parameters
                # remain at this stage
                if len(sub_study.trials[0].params) < 2:
                    continue

                file_path = os.path.join(folder_path, "".join([sub_study.study_name, '.', file_format]))
                fig = optuna.visualization._get_contour_plot(study=sub_study)
                fig.update_layout(title_text=sub_study.study_name)
                fig.write_image(file_path,
                                scale=6.75,
                                width=None,
                                height=None)
        except:
            warnings.warn("Orca could not find an X11 interface, plotting disabled.")

    @staticmethod
    def plot_history(file_path: str, study: optuna.Study):
        try:
            fig = optuna.visualization._get_optimization_history_plot(study=study)
            fig.write_image(file_path,
                            scale=3.25,
                            width=None,
                            height=None)
        except:
            warnings.warn("Orca could not find an X11 interface, plotting disabled.")

    def _split_study_by_algorithm(self, study: optuna.Study) -> list:
        # the general idea is to make a copy of the "Study" object and remove all trials that do not belong to a given
        # algorithm, i.e. return a list of "Study" objects, one for each algorithm used
        # note, that internally "optuna" only uses the trials and optimization direction to do the plots
        studies_list = []

        # 1) get whether it is a regression or classification and, since algorithms are just another hyper-parameter in
        #    "Optuna_AZ", build a list of the algorithms used
        study_type = self._get_study_type(study=study)
        names_algorithms = list(dict.fromkeys([trial.params[study_type] for trial in study.trials]))

        # 2) loop over algorithms and remove trials that are not using the current algorithm; also set "best" attributes
        #    to "None" to avoid undesirable side-effects
        for algorithm in names_algorithms:
            # an unique study name is necessary for internal reasons
            storage = storages.InMemoryStorage()

            # remove the algorithm as "hyper-parameter" and renumber the trials to make sure they are plotted properly
            trials = [trial for trial in deepcopy(study.trials) if trial.params[study_type] == algorithm and
                                                                   trial.state == TrialState.COMPLETE]
            if len(trials) == 0:
                continue
            for number, trial in enumerate(trials):
                del trial.params[study_type]
                del trial.distributions[study_type]
                trial_updated = FrozenTrial(number=number,
                                            state=TrialState.COMPLETE,
                                            value=trial.value,
                                            datetime_start=trial.datetime_start,
                                            datetime_complete=trial.datetime_complete,
                                            params=trial.params,
                                            distributions=trial.distributions,
                                            user_attrs=trial.user_attrs,
                                            system_attrs=trial.system_attrs,
                                            intermediate_values=trial.intermediate_values,
                                            trial_id=number)
                trials[number] = trial_updated

            storage.trials = trials
            storage.study_name = algorithm
            sub_study = optuna.Study(study_name=algorithm, storage=storage)
            studies_list.append(sub_study)

        return studies_list

    def _get_study_type(self, study: optuna.Study) -> str:
        if self._VE.VISUALIZATION_REGRESSOR in study.best_trial.distributions.keys():
            return self._VE.VISUALIZATION_REGRESSOR
        elif self._VE.VISUALIZATION_CLASSIFIER in study.best_trial.distributions.keys():
            return self._VE.VISUALIZATION_CLASSIFIER
        else:
            raise AttributeError("Study must be either classification or regression.")

    @staticmethod
    def _make_folder(path):
        # make sure, the output folder for the plots exists; not that this will only work if the
        # last directory is missing
        if not os.path.exists(path):
            os.mkdir(path)


