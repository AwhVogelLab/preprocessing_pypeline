import numpy as np
import mne
import mne_bids
import os
from glob import glob
import pandas as pd
import json
import shutil
from pathlib import Path


FILTER_FREQS = (0, 80)  # low, high


class Preprocess:
    def __init__(
        self,
        data_dir: str,  # where you keep datasets
        root_dir: str,  # location of RAW data
        trial_start: float,  # start of trial (ms)
        trial_end: float,  # end of trial (ms)
        stim_conditions: list,  # valid condition labels
        timelock_ix: int,  # which position to timelock to (in event_code_dict)
        event_dict: dict,  # event name: code mapping
        event_code_dict: dict,  # event code: sequence mapping
        filter_freqs=FILTER_FREQS,  # frequencies for bandpass filtering
        srate: int = None,  # data sampling rate (Hz)
        no_et_spaces: bool = True,  # are there spaces in the eyetracking file?
        event_names: dict = None,  # names of each annotation
        baseline_time=None,  # times for baselining
        rejection_time=(None, None),  # times to do rejection
        drop_channels=None,  # channels to drop outright (not recommendedd)
        experiment_name=None,  # name of your experiment
    ):

        self.data_dir = data_dir
        self.root_dir = root_dir
        self.srate = srate
        self.trial_start_t = trial_start
        self.trial_end_t = trial_end
        self.event_dict = event_dict
        self.stim_conditions = stim_conditions
        self.event_code_dict = event_code_dict
        self.event_names = event_names if event_names is not None else event_dict
        self.event_names.update(  # these occasionally appear
            {
                "New Segment/": 99999,
                "New Segment/LostSamples: 2": 10001,
            }
        )
        self.timelock_ix = timelock_ix
        self.filter_freqs = filter_freqs
        self.no_et_spaces = no_et_spaces
        self.baseline_time = baseline_time
        rejection_time[0] = trial_start if rejection_time[0] is None else rejection_time[0]
        rejection_time[1] = trial_end if rejection_time[1] is None else rejection_time[1]
        self.rejection_time = rejection_time
        self.drop_channels = drop_channels
        self.experiment_name = experiment_name

    def rereference_to_average(self, data, reref_values):
        """
        re reference data to the average of the offline reference and the given data from another channel
        """
        assert data.shape == reref_values.shape
        return data - (0.5 * reref_values)

    def import_eeg(self, subject_number, overwrite=False):
        """
        function to import raw eeg data and convert it to a bids object

        Args:
            root_data_dir: parent directory of all subjects
            subject_number: the subject number (data should be in a folder with this name)
        Returns:
            bids object (saved from raw EEG)
            mne-python raw dataset
            events array (as a dataframe)


        """

        subject_dir = os.path.join(self.root_dir, subject_number)
        vhdr_file = sorted(glob("*.vhdr", root_dir=subject_dir))

        if len(vhdr_file) == 0:
            raise FileNotFoundError("No vhdr files in subject directory")
        elif len(vhdr_file) == 1:
            concatenated = False

            eegfile = os.path.join(subject_dir, vhdr_file[0])  # search for vhdr file
            eegdata = mne.io.read_raw_brainvision(
                eegfile, eog=["HEOG", "VEOG"], misc=["StimTrak"], preload=False
            )  # read into mne.raw structure
        elif len(vhdr_file) > 1:
            raws = []

            for file in vhdr_file:
                print(
                    "More than 1 vhdr file present in subject directory. They will be concatenated in alphabetical order"
                )

                eegfile = os.path.join(subject_dir, file)  # search for vhdr file
                raws.append(
                    mne.io.read_raw_brainvision(eegfile, eog=["HEOG", "VEOG"], misc=["StimTrak"], preload=False)
                )
            eegdata = mne.concatenate_raws(raws)

        events, event_dict = mne.events_from_annotations(eegdata)
        boundaries = {k: v for k, v in event_dict.items() if "New Segment" in k}
        self.event_names.update(boundaries)

        # drop artifactual extra conditions
        events_to_keep = np.isin(events[:, 2], list(self.event_names.values()))
        if any(~events_to_keep):
            uq_events, counts = np.unique(events[~events_to_keep, 2], return_counts=True)
            print(
                f"WARNING: Dropping {np.sum(counts)} events that are not in the event list:"
                + "\n".join([f"{uq_events[i]} ({counts[i]})" for i in range(len(uq_events))])
            )
        events = events[events_to_keep]

        eegdata.set_annotations(None)

        bids_path = mne_bids.BIDSPath(
            subject=subject_number,
            task=self.experiment_name,
            root=self.data_dir,
            datatype="eeg",
            suffix="eeg",
            extension=".vhdr",
        )
        mne_bids.write_raw_bids(
            eegdata,
            bids_path,
            overwrite=overwrite,
            events=events,
            event_id=self.event_names,
            verbose=False,
            allow_preload=True,
            format="BrainVision",
        )

        # update sidecar with base values
        bids_path.update(extension=".json")

        with open(Path(__file__).parent.joinpath("../base_bids_files/TEMPLATE_eeg.json")) as f:
            sidecar_base = json.load(f)

        mne_bids.update_sidecar_json(bids_path, sidecar_base)

        # events,_ = mne.events_from_annotations(eegdata)
        events = pd.read_csv(bids_path.copy().update(suffix="events", extension=".tsv").fpath, sep="\t")

        return eegdata, events

    def import_behavior(self, subject_number):
        """
        Imports behavioral data into a BIDS-compatible TSV
        """
        path = mne_bids.BIDSPath(
            subject=subject_number,
            task=self.experiment_name,
            root=self.data_dir,
            datatype="beh",
            suffix="beh",
            extension=".tsv",
            check="false",
        )
        for f in glob(os.path.join(self.root_dir, subject_number, "*.csv")):
            if "beh" in f:
                pd.read_csv(f).to_csv(path.fpath, sep="\t")

    def import_eyetracker(self, subject_number, keyword=None):
        """Loads in eyetracking data for a subject

        Args:
            subject_dir (str): directory where subject's data is found. Should contain ONE .asc file.
            It will be automatically converted if necessary
            keyword (str, default none): the keyword you use before sync codes


        Returns:
            eye: mne raw object containing eyetracking data
            et_events: events structure containing condition codes. These should match the EEG conditions
        """

        subject_dir = os.path.join(self.root_dir, subject_number)
        path = mne_bids.path.BIDSPath(
            subject=subject_number,
            task=self.experiment_name,
            root=self.data_dir,
            datatype="eyetracking",
            suffix="eyetracking",
            extension=".asc",
            check=False,
        )
        path.mkdir()

        # COPY MAIN ASC FILE (with spaces removed)

        asc_file = glob("*.asc", root_dir=subject_dir)
        if len(asc_file) == 0:
            raise FileNotFoundError("No .asc file. Are you sure you remembered to convert it?")
            # TODO: automatically convert edf files

        if len(asc_file) == 1:
            asc_file = os.path.join(subject_dir, asc_file[0])

            if self.no_et_spaces:
                shutil.copy2(asc_file, path.fpath)
            else:
                self.remove_eyetrack_spaces(asc_file, path.fpath)

            # load in eye tracker data
            eye = mne.io.read_raw_eyelink(path.fpath, create_annotations=["blinks", "messages"])

        else:  # more than one asc
            print("More than 1 asc file present in subject directory. They will be concatenated in alphabetical order")
            raws = []
            for ifile, file in enumerate(asc_file):
                file = os.path.join(subject_dir, file)

                ascpath = path.copy().update(split=ifile + 1)

                if self.no_et_spaces:
                    shutil.copy2(file, ascpath.fpath)
                else:
                    self.remove_eyetrack_spaces(file, ascpath.fpath)

                try:
                    raws.append(mne.io.read_raw_eyelink(ascpath.fpath, create_annotations=["blinks", "messages"]))
                except ValueError as e:
                    print(
                        f"Error reading {ascpath.fpath}. This may be due to a bug in mne if your eyetracking file"
                        + "contains dropouts where the eye is lost. To fix this, manually re-generate the as files"
                        + "with 'Block Flags Output' checked"
                    )
                    raise e
            eye = mne.concatenate_raws(raws)

        keyword = "" if keyword is None else keyword
        regexp = f"^{keyword}(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$"
        et_events, et_event_dict = mne.events_from_annotations(eye, regexp=regexp)

        # save sidecar
        path.update(extension=".json")
        shutil.copy(
            Path(__file__).parent.joinpath("../base_bids_files/TEMPLATE_eyetracking.json"),
            path.fpath,
        )
        print("WARNING: YOU WILL HAVE TO MODIFY THE SIDECAR FILE YOURSELF")  # option? automatically open this?

        # convert events to match the EEG events

        et_events_dict_convert = {}
        for k, v in et_event_dict.items():
            new_k = int(k.split(" ")[-1])
            et_events_dict_convert[v] = new_k
        et_events_converted = et_events.copy()
        for code in et_events_dict_convert.keys():
            et_events_converted[:, 2][et_events[:, 2] == code] = et_events_dict_convert[code]

        # save events as TSV
        path.update(suffix="events", extension=".tsv")

        eye_events = pd.DataFrame(columns=["onset", "duration", "trial_type", "value", "sample"])
        eye_events["sample"] = et_events_converted[:, 0]
        eye_events["value"] = et_events_converted[:, 2]
        eye_events["onset"] = eye_events["sample"] / 1000
        event_names_inv = {v: k for k, v in self.event_names.items()}
        get_events = lambda trl: event_names_inv[trl["value"]]
        eye_events["trial_type"] = eye_events.apply(get_events, axis=1)
        eye_events["duration"] = 0
        eye_events.to_csv(path.fpath, sep=str("\t"), index=False)

        # sidecar events
        path.update(extension=".json")
        shutil.copy(
            Path(__file__).parent.joinpath("../base_bids_files/TEMPLATE_events.json"),
            path.fpath,
        )
        print("WARNING: YOU WILL HAVE TO MODIFY THE SIDECAR FILE YOURSELF")  # option? automatically open this?

        return eye, eye_events

    def convert_bids_events(self, events):
        """
        Converts a BIDS events file to a mne events array
        of the form [sample,0,value]
        """
        return events[["sample", "duration", "value"]].to_numpy().astype(int)

    def make_eeg_epochs(self, eeg, eeg_events, eeg_trials_drop=None):
        """
        Function that handles epoching for EEG data. A replacement for make_and_sync_epochs when you don't want eyetracking

        Args:
            eeg: mne raw object containing EEG data
            eeg_events: events structure containing condition codes. These should match the eyetracking conditions
            eeg_trials_drop: trials to drop from EEG

        Returns:
            epochs: mne epochs object containing data
        """

        if eeg_trials_drop is None:
            eeg_trials_drop = []
        if self.srate is None:
            self.srate = eeg.info["sfreq"]

        # convert event dataframe to mne format (array of sample, duration, value)
        eeg_events = self.convert_bids_events(eeg_events)
        eeg_events = self.filter_events(eeg_events)

        # get EEG epochs object
        assert eeg.info["sfreq"] % self.srate == 0
        decim = eeg.info["sfreq"] / self.srate

        epochs = mne.Epochs(
            eeg,
            eeg_events,
            self.event_dict,
            tmin=self.trial_start_t,
            tmax=self.trial_end_t,
            on_missing="ignore",
            baseline=self.baseline_time,
            reject_tmin=self.rejection_time[0],
            reject_tmax=self.rejection_time[1],
            preload=True,
            decim=decim,
        ).drop(
            eeg_trials_drop
        )  # set up epochs object
        if self.drop_channels is not None:
            eeg_epochs = eeg_epochs.drop_channels(self.drop_channels)

        return epochs

    def make_and_sync_epochs(
        self,
        eeg,
        eeg_events,
        eye,
        eye_events,
        eeg_trials_drop=None,
        eye_trials_drop=None,
    ):
        """
        Function that does basic epoching
        converts EEG and eyetracking raw objects into epochs

        Args:
            eeg: mne raw object containing EEG data
            eeg_events: events structure containing condition codes. These should match the eyetracking conditions
            eye: mne raw object containing eyetracking data
            eye_events: events structure containing condition codes. These should match the EEG conditions
            eeg_trials_drop: trials to drop from EEG
            eye_trials_drop: trials to drop from eyetracking

        Returns:
            epochs: mne epochs object containing combined data

        """
        if eeg_trials_drop is None:
            eeg_trials_drop = []
        if eye_trials_drop is None:
            eye_trials_drop = []
        if self.srate is None:
            self.srate = eeg.info["sfreq"]

        # get our events list
        unmatched_codes = list(set(eeg_events["value"].unique()) ^ set(eye_events["value"].unique()))
        # convert event dataframe to mne format (array of sample, duration, value)
        eeg_events = self.convert_bids_events(eeg_events)
        eye_events = self.convert_bids_events(eye_events)
        eeg_events = eeg_events[~np.isin(eeg_events, unmatched_codes).any(axis=1)]
        eye_events = eye_events[~np.isin(eye_events, unmatched_codes).any(axis=1)]
        eeg_events = self.filter_events(eeg_events)
        eye_events = self.filter_events(eye_events)

        # get EEG epochs object
        assert eeg.info["sfreq"] % self.srate == 0
        decim = eeg.info["sfreq"] / self.srate

        eeg_epochs = mne.Epochs(
            eeg,
            eeg_events,
            self.event_dict,
            tmin=self.trial_start_t,
            tmax=self.trial_end_t,
            on_missing="ignore",
            baseline=self.baseline_time,
            reject_tmin=self.rejection_time[0],
            reject_tmax=self.rejection_time[1],
            preload=True,
            decim=decim,
        ).drop(
            eeg_trials_drop
        )  # set up epochs object
        if self.drop_channels is not None:
            eeg_epochs = eeg_epochs.drop_channels(self.drop_channels)

        assert eye.info["sfreq"] % self.srate == 0
        decim = eye.info["sfreq"] / self.srate
        eye_epochs = mne.Epochs(
            eye,
            eye_events,
            self.event_dict,
            tmin=self.trial_start_t,
            tmax=self.trial_end_t,
            on_missing="ignore",
            baseline=self.baseline_time,
            reject=None,
            flat=None,
            reject_by_annotation=False,
            preload=True,
            decim=decim,
        ).drop(eye_trials_drop)

        if "DIN" in eye_epochs.info["ch_names"]:
            eye_epochs.drop_channels(["DIN"])

        if len(eye_epochs) != len(
            eeg_epochs
        ):  # this happens if you abort recording mid trial. Trials should (normally) never be dropped
            dropped_eye_trials = [trial for trial, reason in enumerate(eye_epochs.drop_log) if len(reason) > 0]
            dropped_eeg_trials = [trial for trial, reason in enumerate(eeg_epochs.drop_log) if len(reason) > 0]
            print(
                f"WARNING: issue with trial count. EEG has {len(eeg_epochs)} trials, eyetracking has {len(eye_epochs)} trials\n.This is likely because you aborted the recording mid trial. If you did not do this, double check your event timings"
            )
            print(f"Dropping EEG trials: {dropped_eye_trials}")
            print(f"Dropping Eyetracking trials: {dropped_eeg_trials}")
            eeg_epochs.drop(dropped_eye_trials)
            eye_epochs.drop(dropped_eeg_trials)

        try:
            epochs = eeg_epochs.copy()
            epochs.add_channels([eye_epochs], force_update_info=True)
        except ValueError as e:
            print(f"EEG has {len(eeg_epochs.info.ch_names)} channels and {len(eeg_epochs)} trials")
            print(f"Eyetracking has {len(eye_epochs.info.ch_names)} channels and {len(eye_epochs)} trials")
            raise e

        # there is some code here that runs selections. What does this do??

        return epochs

    def remove_eyetrack_spaces(self, input_file, output_file):
        """Removes spaces and saccades from asc file"""
        with open(input_file, "r") as f:
            lines = f.readlines()
        newlines = np.array(lines)[
            [l != "\n" and "ESACC" not in l and "EFIX" not in l for l in lines]
        ]  # it breaks with empty lines or saccades, so delete these
        with open(output_file, "w") as f:
            f.writelines(newlines)

    def check_both_eyes(self, chan_labels, rej_chans):
        """
        If an artifact is found in eyetracking channels, ensures that it is found in both eyes
            chan_labels: list of all channel labels
            rej_chans: trials x channels matrix, boolean
        """
        if isinstance(chan_labels, list):
            chan_labels = np.array(chan_labels)
        if np.all(
            [eye_chan in chan_labels for eye_chan in ["xpos_right", "xpos_left", "ypos_right", "ypos_left"]]
        ):  # TODO: triple check this works
            x_chans = chan_labels[["xpos" in c for c in chan_labels]]
            y_chans = chan_labels[["ypos" in c for c in chan_labels]]

            rej_chans[:, np.isin(chan_labels, x_chans)] = rej_chans[:, np.isin(chan_labels, x_chans)].all(axis=1)[
                :, np.newaxis
            ]
            rej_chans[:, np.isin(chan_labels, y_chans)] = rej_chans[:, np.isin(chan_labels, y_chans)].all(axis=1)[
                :, np.newaxis
            ]
        return rej_chans

    def get_data_from_rej_period(self, epochs):
        """
        grab the data from the rejection period, which might be smaller than epoch length
        """
        return epochs.get_data(copy=True)[
            :,
            :,
            np.logical_and(
                epochs.times >= self.rejection_time[0],
                epochs.times <= self.rejection_time[1],
            ),
        ]

    def conv_ms_to_samples(self, dur, epochs):
        """
        convert a duration in ms to timepoints
        """
        return int(np.floor(dur * epochs.info["sfreq"] / 1000))  # convert ms to timepoints

    def artreject_nan(self, epochs):
        """
        Rejects trials that contain any nan values
        """
        eegdata = self.get_data_from_rej_period(epochs)
        return np.any(np.isnan(eegdata), 2)

    def artreject_slidingP2P(self, epochs, rejection_criteria, win=200, win_step=100):
        """
        Runs artifact rejection baased on a sliding window
            epochs: mne epochs object containing  data
            rejection_criteria - dict containing thresholds at which to reject (peak to peak)
            win: size of sliding window, int or "absolute"
            win_step: step size
        Returns:
            rej_chans: indicates which electrodes match automated rejection criteria
        """

        eegdata = self.get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())

        win = self.conv_ms_to_samples(win, epochs)
        win_step = self.conv_ms_to_samples(win_step, epochs)

        if isinstance(win, int):
            win_starts = np.arange(0, eegdata.shape[2] - win, win_step)
        elif win == "absolute":
            win_starts = [0]
            win = eegdata.shape[2]
        else:
            raise ValueError("win must be either an integer value or 'absolute'")

        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            threshold = rejection_criteria[chan_type]
            for st in win_starts:
                data_min = np.nanmin(eegdata[:, chans, st : st + win], axis=2)
                data_max = np.nanmax(eegdata[:, chans, st : st + win], axis=2)
                rej_chans[:, chans] = np.logical_or(rej_chans[:, chans], (data_max - data_min) > threshold)

        # if both eyes are recorded, then ONLY mark artifacts if they appear in both eyes
        rej_chans = self.check_both_eyes(epochs.ch_names, rej_chans)

        return rej_chans

    def artreject_value(self, epochs, rejection_criteria):
        """
        Runs artifact rejection baased on if a channel meets or exceeds a value from baseline (0)
            epochs: mne epochs object containing  data
            rejection_criteria - dict containing thresholds at which to reject

        Returns:
            rej_chans: indicates which electrodes match automated rejection criteria
        """

        eegdata = self.get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())
        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            threshold = rejection_criteria[chan_type]

            data_min = np.nanmin(eegdata[:, chans], axis=2)
            data_max = np.nanmax(eegdata[:, chans], axis=2)
            rej_chans[:, chans] = np.logical_or(data_max > threshold, data_min < -1 * threshold)

        # if both eyes are recorded, then ONLY mark artifacts if they appear in both eyes
        rej_chans = self.check_both_eyes(epochs.ch_names, rej_chans)

        return rej_chans

    def artreject_step(self, epochs, rejection_criteria, win=80, win_step=10):
        """
        Runs artifact rejection baased on a sliding window
            epochs: mne epochs object containing  data
            rejection_criteria - dict containing thresholds at which a step is too large
            win: size of sliding window, int
            win_step: step size
        Returns:
            rej_chans: indicates which electrodes match automated rejection criteria
        """

        eegdata = self.get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())

        win = self.conv_ms_to_samples(win, epochs)
        win_step = self.conv_ms_to_samples(win_step, epochs)

        win_starts = np.arange(0, eegdata.shape[2] - win, win_step)
        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            threshold = rejection_criteria[chan_type]
            for st in win_starts:

                first_half = np.nanmean(eegdata[:, chans, st : st + win // 2], axis=2)
                last_half = np.nanmean(eegdata[:, chans, st + win // 2 : st + win], axis=2)
                rej_chans[:, chans] = np.logical_or(rej_chans[:, chans], np.abs(first_half - last_half) > threshold)

        # if both eyes are recorded, then ONLY mark artifacts if they appear in both eyes
        rej_chans = self.check_both_eyes(epochs.ch_names, rej_chans)

        return rej_chans

    def artreject_linear(self, epochs, min_slope=75e-6, min_r2=0.3):
        """
        Rejects trials based on a linear regression fit

        Args:
            epochs: mne epochs object containing  data
            min_slope (int, optional): minimum slope to reject. Units are V/S
            min_r2 (float, optional): minimum r2 to reject at. Defaults to 0.3.
        """

        eegdata = self.get_data_from_rej_period(epochs)
        chans = np.array(epochs.info.get_channel_types()) == "eeg"  # TODO: make more flexible?

        # Note, using times from the epochs object means that slope will always be in V/S units
        xs = epochs.times[
            np.logical_and(
                epochs.times >= self.rejection_time[0],
                epochs.times <= self.rejection_time[1],
            )
        ]

        A = np.vstack([xs, np.ones(len(xs))]).T
        slopes = np.full((eegdata.shape[0:2]), 0, dtype=float)
        ssrs = np.full((eegdata.shape[0:2]), 0, dtype=float)

        for itrial in range(eegdata.shape[0]):
            (slopes[itrial, chans], _), ssrs[itrial, chans], _, _ = np.linalg.lstsq(
                A, eegdata[itrial, chans].T, rcond=None
            )

        r2s = 1 - ssrs / (eegdata.shape[2] * eegdata.var(axis=2))  # double check value for n
        rej_linear = np.logical_and(r2s > min_r2, slopes > min_slope)
        return rej_linear

    def artreject_flatline(self, epochs, rejection_criteria, flatline_duration=200):
        """
        Rejects channels with flatline behavior (more than [flatline_duration] ms of the same value)
        You should probably only run this on EEG channels...
        Args:
            epochs: mne epochs object containing  data
            rejection_criteria: dict containing difference thresholds for each window
            flatline_duration: length of time in ms that a channel must be flat to be rejected
        Returns:
            rej_chans: indicates which electrodes match automated rejection criteria
        """

        duration = self.conv_ms_to_samples(flatline_duration, epochs)

        def get_flatline(non_flats, duration=duration):
            """
            function that finds at least [duration] subsequent timepoints of the same value
            Not pretty, but fairly optimized
            Args:
                non_flats: boolean array where True indicates a non-flat moment (large enough change in value b/w adjacent timepoints)
                duration (optional): number of subsequent timepoints that need to be flat
            Returns:
                boolean indicating if there is a flatline of at least [duration] timepoints
            """
            return np.any(  # see if the gap is bigger than duration at any timepoint
                np.diff(  # see if the gap between indices is bigger than the duration
                    np.where(  # find the indices of non-flat moments
                        np.concatenate(
                            ([True], non_flats, [True])
                        )  # add True to the beginning and end of the array, to avoid unbounded runs of flats
                    )[0]
                )
                >= duration
            )

        eegdata = self.get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())
        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            if np.sum(
                chans
            ):  # some datasets might not have all channel types (e.g. EOGs), which causes apply_along_axis to fail
                threshold = rejection_criteria[chan_type]

                diff = np.diff(eegdata[:, chans], axis=2)
                non_flats = (diff < -threshold) | (diff > threshold)

                rej_chans[:, chans] = rej_chans[:, chans] | np.apply_along_axis(
                    get_flatline, 2, non_flats
                )  # apply the function to each trial

        rej_chans = self.check_both_eyes(epochs.ch_names, rej_chans)

        return rej_chans

    def filter_events(self, events):

        class NoCode(Exception):
            pass

        # iterates through the list of event lists, requiring it to match the sequence in required_event_order
        # only picks out the first stimulus
        new_events = []
        new_times = []
        for i in range(len(events)):
            for code, sequence in self.event_code_dict.items():
                try:
                    for j, ev in enumerate(sequence):
                        if events[i + j, 2] != ev:
                            raise NoCode
                except NoCode:
                    continue
                except IndexError:
                    break

                else:
                    new_events.append(code)
                    new_times.append(events[i + self.timelock_ix, 0])
        new_event_list = np.stack((new_times, np.zeros(len(new_times)), new_events), axis=1).astype(int)
        return new_event_list

    def deg2pix(self, eyeMoveThresh=1, distFromScreen=800, monitorWidth=532, screenResX=1920):
        """Converts degrees visual angle to a pixel value

        Args:
            eyeMoveThresh (int, optional): threshold (dva). Defaults to 1.
            distFromScreen (int, optional): distance from headrest to screen. Defaults to 900.
            monitorWidth (int, optional): Width of monitor. Defaults to 532.
            screenResX (int, optional): Monitor resolution. Defaults to 1920.

        Returns:
            pix: pixel value
        """

        pix_size_x = monitorWidth / screenResX
        mmfromfix = 2 * distFromScreen * np.tan(0.5 * np.deg2rad(eyeMoveThresh))
        pix = round(mmfromfix / pix_size_x)
        return pix

    def save_all_data(self, subject_number, epochs, rej_reasons):
        """
        Function to save all data to the subject's derivative directory

        Arguments:
            subject_number (str): the subject number
            epochs: mne epochs object
            rej_reasons: rejection reasons numpy array  (trials x channels)
        """
        # file processing

        path = mne_bids.path.BIDSPath(
            subject=subject_number,
            task=self.experiment_name,
            description="preprocessed",
            datatype="eeg",
            root=os.path.join(self.data_dir, "derivatives"),
        )
        og_path = mne_bids.path.BIDSPath(
            subject=subject_number,
            task=self.experiment_name,
            suffix="eeg",
            datatype="eeg",
            root=self.data_dir,
        )
        path.mkdir()

        # EVENTS

        events_final = pd.DataFrame(epochs.events, columns=["sample", "duration", "value"])
        event_dict_inv = {v: k for k, v in self.event_dict.items()}
        get_events = lambda trl: event_dict_inv[trl["value"]]
        events_final["trial_type"] = events_final.apply(get_events, axis=1)
        events_final["onset"] = events_final["sample"] / self.srate
        events_final = events_final[["onset", "duration", "trial_type", "value", "sample"]]

        path.update(suffix="events", extension=".tsv")
        events_final.to_csv(path.fpath, sep="\t", index=False)

        # COPY EVENTS SIDECAR
        path.update(suffix="events", extension=".json")
        shutil.copy(og_path.find_matching_sidecar("events.json"), path.fpath)

        # COPY ELECTRODES
        path.update(suffix="electrodes", extension=".tsv")
        shutil.copy(og_path.find_matching_sidecar("electrodes.tsv"), path.fpath)

        # COPY SIDECAR AND CHANGE TO EPOCHED

        sidecar = og_path.find_matching_sidecar("eeg.json")
        new_sidecar = path.update(suffix="eeg", extension=".json")
        with open(sidecar) as f:
            sidecar_data = json.load(f)
        new_keys = {
            "RecordingType": "epoched",
            "EpochLength": round(self.trial_end_t - self.trial_start_t, 3),
        }
        sidecar_data.update(new_keys)
        with open(new_sidecar.fpath, "w") as f:
            json.dump(sidecar_data, f, indent=4)

        ## BIDS NONCOMPLIANT FROM HERE ON. SUGGESTIONS WELCOME

        # SAVE EPOCHS AS FIF AND NPY

        path.update(suffix="eeg", extension=".fif", check=False)
        epochs.save(path.fpath, overwrite=True)
        path.update(suffix="eeg", extension=".npy", check=False)
        np.save(path.fpath, epochs.get_data())

        # ARTIFACTS
        path.update(suffix="artifacts", extension=".tsv", check=False)  # manually forcing us to allow an artifacts file
        pd.DataFrame(rej_reasons, columns=epochs.info["ch_names"]).to_csv(path.fpath, sep="\t", index=False)
