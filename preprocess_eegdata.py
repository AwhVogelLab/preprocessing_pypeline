import numpy as np
import mne
import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton
from matplotlib.text import Annotation


FILTER_FREQS = (0, 80)  # low, high

# FOR PLOTTING
SLIDER_STEP = 5  #


EYETRACK_SCALE = 1e-6  # what factor to downscale eyetracking by
CHAN_OFFSET = 0.00005


class Preprocess:
    def __init__(
        self,
        parent_dir: str,
        file_prefix: str,
        trial_start: float,
        trial_end: float,
        event_dict: dict,
        stim_conditions: list,
        timelock_ix: int,
        event_code_dict: dict,
        event_acceptable_window: int = 1000,
        filter_freqs=FILTER_FREQS,
        no_et_spaces: bool = True,
        baseline_time=None,
        rejection_time=(None, None),
        drop_channels=None,
    ):

        self.parent_dir = parent_dir
        self.file_prefix = file_prefix
        self.trial_start_t = trial_start
        self.trial_end_t = trial_end
        self.event_dict = event_dict
        self.stim_conditions = stim_conditions
        self.event_code_dict = event_code_dict
        self.event_acceptable_window = event_acceptable_window
        self.timelock_ix = timelock_ix
        self.filter_freqs = filter_freqs
        self.no_et_spaces = no_et_spaces
        self.baseline_time = baseline_time
        rejection_time[0] = trial_start if rejection_time[0] is None else rejection_time[0]
        rejection_time[1] = trial_end if rejection_time[1] is None else rejection_time[1]
        self.rejection_time = rejection_time
        self.drop_channels = drop_channels

    def rereference_to_average(self, data, reref_values):
        """
        re reference data to the average of the offline reference and the given data from another channel
        """
        assert data.shape == reref_values.shape
        return data - (0.5 * reref_values)

    def load_eeg(self, subject_dir, srate=None):
        """Load in eeg data for a subject, filters, and re-references it

        Args:
            subject_dir (str): directory where subject's data is found. Should contain a .vhdr file

        Returns:
            eegdata: mne raw structure
            events: mne events
            eeg_event_dict: dict returned by mne.events_from_annotations
        """
        vhdr_file = sorted(glob("*.vhdr", root_dir=subject_dir))

        if len(vhdr_file) == 0:
            raise FileNotFoundError("No vhdr files in subject directory")
        elif len(vhdr_file) == 1:

            eegfile = os.path.join(subject_dir, vhdr_file[0])  # search for vhdr file
            eegdata = mne.io.read_raw_brainvision(eegfile, eog=["HEOG", "VEOG"], misc=["StimTrak"], preload=True)  # read into mne.raw structure
        elif len(vhdr_file) > 1:
            raws = []

            for file in vhdr_file:
                print("More than 1 vhdr file present in subject directory. They will be concatenated in alphabetical order")

                eegfile = os.path.join(subject_dir, file)  # search for vhdr file
                raws.append(mne.io.read_raw_brainvision(eegfile, eog=["HEOG", "VEOG"], misc=["StimTrak"], preload=True))
                eegdata = mne.concatenate_raws(raws)
                print(eegdata)

        reref_index = mne.pick_channels(eegdata.ch_names, ["TP9"])

        eegdata.apply_function(self.rereference_to_average, picks=["eeg"], reref_values=np.squeeze(eegdata.get_data()[reref_index]))

        # eegdata.set_eeg_reference(['TP9'])
        eegdata.filter(*self.filter_freqs, n_jobs=-1)

        events, eeg_event_dict = mne.events_from_annotations(eegdata)  # extract events

        return eegdata, events, eeg_event_dict

    def remove_eyetrack_spaces(self, input_file, output_file):
        """Removes spaces and saccades from asc file"""
        with open(input_file, "r") as f:
            lines = f.readlines()
        newlines = np.array(lines)[[l != "\n" and "ESACC" not in l for l in lines]]  # it breaks with empty lines or saccades, so delete these
        with open(output_file, "w") as f:
            f.writelines(newlines)

    def load_eyetracking(self, subject_dir):
        """Loads in eyetracking data for a subject

        Args:
            subject_dir (str): directory where subject's data is found. Should contain ONE .asc file.
            It will be automatically converted if necessary


        Returns:
            eye: mne raw object containing eyetracking data
            et_events: events structure containing condition codes. These should match the EEG conditions
        """

        asc_file = glob("*.asc", root_dir=subject_dir)
        if len(asc_file) == 0:
            raise FileNotFoundError("You need to convert the edf file to asc first")
            # TODO: automatically convert edf files
        if len(asc_file) > 1:
            raise RuntimeError("More than 1 asc file present in subject directory")
        asc_file = os.path.join(subject_dir, asc_file[0])
        if self.no_et_spaces:
            et_file = asc_file
        else:
            self.remove_eyetrack_spaces(asc_file, os.path.join(subject_dir, "et_temp.asc"))
            et_file = os.path.join(subject_dir, "et_temp.asc")

        # load in eye tracker data
        eye = mne.io.read_raw_eyelink(et_file, create_annotations=["blinks", "messages"])

        # delete our temp file because it is no longer necessary
        if os.path.exists(os.path.join(subject_dir, "et_temp.asc")):
            os.remove(os.path.join(subject_dir, "et_temp.asc"))

        et_events, et_event_dict = mne.events_from_annotations(eye)

        et_events_dict_convert = {}
        for k, v in et_event_dict.items():
            try:
                new_k = int(k.split(" ")[-1])
                et_events_dict_convert[v] = new_k
            except ValueError:  # once got 'ELCL_PROC CENTROID (3)' as a key, which seems separate from the rest
                pass
        et_events_converted = et_events.copy()
        for code in et_events_dict_convert.keys():
            et_events_converted[:, 2][et_events[:, 2] == code] = et_events_dict_convert[code]  # make eyetracking events match eeg events

        return eye, et_events_converted, et_events_dict_convert

    def check_both_eyes(self, chan_labels, rej_chans):
        """
        If an artifact is found in eyetracking channels, ensures that it is found in both eyes
            chan_labels: list of all channel labels
            rej_chans: trials x channels matrix, boolean
        """
        if isinstance(chan_labels, list):
            chan_labels = np.array(chan_labels)
        if np.all([eye_chan in chan_labels for eye_chan in ["xpos_right", "xpos_left", "ypos_right", "ypos_left"]]):  # TODO: triple check this works
            # x_chans = chan_labels[[len(re.findall('xpos',c)) > 0 for c in chan_labels]]
            # y_chans = chan_labels[[len(re.findall('ypos',c)) > 0 for c in chan_labels]]
            x_chans = chan_labels[["xpos" in c for c in chan_labels]]
            y_chans = chan_labels[["ypos" in c for c in chan_labels]]

            rej_chans[:, np.isin(chan_labels, x_chans)] = rej_chans[:, np.isin(chan_labels, x_chans)].all(axis=1)[:, np.newaxis]
            rej_chans[:, np.isin(chan_labels, y_chans)] = rej_chans[:, np.isin(chan_labels, y_chans)].all(axis=1)[:, np.newaxis]
        return rej_chans

    def get_data_from_rej_period(self, epochs):
        """
        grab the data from the rejection period, which might be smaller than epoch length
        """
        return epochs.get_data(copy=True)[:, :, np.logical_and(epochs.times >= self.rejection_time[0], epochs.times <= self.rejection_time[1])]

    def conv_ms_to_samples(self, dur, epochs):
        """
        convert a duration in ms to timepoints
        """
        return int(np.floor(dur * epochs.info["sfreq"] / 1000))  # convert ms to timepoints

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
                data_min = eegdata[:, chans, st: st + win].min(axis=2)
                data_max = eegdata[:, chans, st: st + win].max(axis=2)
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

            data_min = eegdata[:, chans].min(axis=2)
            data_max = eegdata[:, chans].max(axis=2)
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

                first_half = eegdata[:, chans, st : st + win // 2].mean(axis=2)
                last_half = eegdata[:, chans, st + win // 2 : st + win].mean(axis=2)
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
        xs = epochs.times[np.logical_and(epochs.times >= self.rejection_time[0], epochs.times <= self.rejection_time[1])]

        A = np.vstack([xs, np.ones(len(xs))]).T
        slopes = np.full((eegdata.shape[0:2]), 0, dtype=float)
        ssrs = np.full((eegdata.shape[0:2]), 0, dtype=float)

        for itrial in range(eegdata.shape[0]):
            (slopes[itrial, chans], _), ssrs[itrial, chans], _, _ = np.linalg.lstsq(A, eegdata[itrial, chans].T, rcond=None)

        r2s = 1 - ssrs / (eegdata.shape[2] * eegdata.var(axis=2))  # double check value for n
        rej_linear = np.logical_and(r2s > min_r2, slopes > min_slope)
        return rej_linear

    def artreject_flatline(self, epochs, rejection_criteria, flatline_duration):
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
                        np.concatenate(([True], non_flats, [True]))  # add True to the beginning and end of the array, to avoid unbounded runs of flats
                    )[0]
                )
                >= duration
            )

        eegdata = self.get_data_from_rej_period(epochs)
        chan_types = np.array(epochs.info.get_channel_types())
        rej_chans = np.full((eegdata.shape[0:2]), False)

        for chan_type in rejection_criteria.keys():
            chans = chan_types == chan_type
            if np.sum(chans):  # some datasets might not have all channel types (e.g. EOGs), which causes apply_along_axis to fail
                threshold = rejection_criteria[chan_type]

                diff = np.diff(eegdata[:, chans], axis=2)
                non_flats = (diff < -threshold) | (diff > threshold)

                rej_chans[:, chans] = rej_chans[:, chans] | np.apply_along_axis(get_flatline, 2, non_flats)  # apply the function to each trial

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

    def subject_pipeline(self, sub):
        """Main pipeline that loads in data and performs artifact rejection

        Args:
            sub: subject id
        """
        print(f"Starting subject {sub}")
        subject_dir = os.path.join(self.parent_dir, sub)

        eeg, eeg_events, eeg_event_dict = self.load_eeg(subject_dir)
        eye, eye_events, et_event_dict = self.load_eyetracking(subject_dir)

        unmatched_codes = list(
            set(eeg_event_dict.values()) ^ set(et_event_dict.values())
        )  # delete any events that do not appear in both ET and EEG (usually a boundary)
        eeg_events = eeg_events[~np.isin(eeg_events, unmatched_codes).any(axis=1)]
        eye_events = eye_events[~np.isin(eye_events, unmatched_codes).any(axis=1)]

        eeg_events = self.filter_events(eeg_events)
        eye_events = self.filter_events(eye_events)

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
        )  # set up epochs object

        if self.drop_channels is not None:
            eeg_epochs = eeg_epochs.drop_channels(self.drop_channels)

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
        )  # set up epochs object

        code_dict = {v: k for k, v in self.event_dict.items()}  # invert the event dict
        eeg_epochs = eeg_epochs[[code_dict[c] for c in self.stim_conditions]].load_data()  # select out our relevant epochs

        eye_epochs = eye_epochs[[code_dict[c] for c in self.stim_conditions]].load_data()
        eye_epochs = eye_epochs.pick(np.setdiff1d(eye_epochs.ch_names, ["pupil_left", "pupil_right", "DIN"]))  # exclude irrelevant eye channels

        # rej_chans_eeg = self.reject_eeg(eeg_epochs)
        # rej_chans_eye = self.reject_eyetracking(eye_epochs)

        epochs = eeg_epochs.copy()
        epochs.add_channels([eye_epochs], force_update_info=True)  # concatenate eyetracking and eeg
        return epochs

        # rej_chans = np.concatenate((rej_chans_eeg,rej_chans_eye),1) # concatenate rejection index

        # save files
        # np.save(os.path.join(self.parent_dir, sub, f"{sub}_rej.npy"), rej_chans)
        # epochs.save(os.path.join(self.parent_dir, sub, f"{sub}_epo.fif"), overwrite=True)
        # np.save(os.path.join(subject_dir, f"{sub}_conditions"), eeg_events[:, 2])


class Epochs:
    def __init__(self, epochs, rej_manual, rej_chans):
        self.epochs = epochs
        self.rej_manual = rej_manual
        self.rej_chans = rej_chans


class Visualizer:
    def __init__(
        self,
        sub,
        parent_dir: str,
        srate: float,
        timelock: float,
        trial_start: float,
        trial_end: float,
        condition_dict: dict,
        rejection_time=(None, None),
        win_step: int = SLIDER_STEP,
        downscale={"eyegaze": EYETRACK_SCALE},
        chan_offset=CHAN_OFFSET,
        channels_drop=None,
        channels_ignore=None,
    ):
        self.sub = sub
        self.parent_dir = parent_dir
        self.condition_dict = condition_dict
        self.srate = srate
        self.timelock = timelock / srate
        self.trial_start = trial_start
        self.trial_end = trial_end
        self.win_step = win_step
        self.epoch_len = np.ceil((trial_end - trial_start) * srate)
        rejection_time[0] = trial_start if rejection_time[0] is None else rejection_time[0]
        rejection_time[1] = trial_end if rejection_time[1] is None else rejection_time[1]
        self.rejection_time = rejection_time
        self.downscale = downscale
        self.chan_offset = chan_offset

        self.epochs_obj = mne.read_epochs(os.path.join(parent_dir, sub, f"{sub}_epo.fif"))
        self.conditions = np.load(os.path.join(parent_dir, sub, f"{sub}_conditions.npy"))
        self.rej_chans = np.load(os.path.join(parent_dir, sub, f"{sub}_rej.npy"))
        self.rej_reasons = np.load(os.path.join(parent_dir, sub, f"{sub}_rej_reasons.npy"), allow_pickle=True)
        self.channels_ignore = channels_ignore

        if channels_drop is not None:
            channels_drop = [ch for ch in channels_drop if ch in self.epochs_obj.ch_names]
            if len(channels_drop) > 0:
                self.rej_chans = self.rej_chans[:, ~np.in1d(self.epochs_obj.ch_names, channels_drop)]
                self.rej_reasons = self.rej_reasons[:, ~np.in1d(self.epochs_obj.ch_names, channels_drop)]
                self.epochs_obj.drop_channels(channels_drop)
        if channels_ignore is not None:
            self.rej_chans[:, np.in1d(self.epochs_obj.ch_names, channels_ignore)] = False
            self.rej_reasons[:, np.in1d(self.epochs_obj.ch_names, channels_ignore)] = None

        self.rej_manual = self.rej_chans.any(1)

        self.info = self.epochs_obj.info
        self.chan_types = self.info.get_channel_types()

        self.chan_order = np.concatenate(
            (
                mne.pick_types(self.info, eeg=True),
                mne.pick_types(self.info, eog=True),
                mne.pick_types(self.info, eyetrack="eyegaze"),
                mne.pick_types(self.info, misc=True),
            )
        )

        self.chan_labels = np.array(self.epochs_obj.ch_names)[self.chan_order]
        self.rej_chans = self.rej_chans[:, self.chan_order]
        self.chan_types = np.array(self.chan_types)[self.chan_order]

        self.epochs_raw = self.epochs_obj.get_data(copy=True)
        self.epochs_pre = None  # initialized when we preprocess

        self.offset_dict = None
        self.xlim = (0, self.epoch_len * self.win_step)

        self.ylim = [None, None]

        self.pos = 0
        self.extra_chan_scale = 1

    def get_rejection_reason(self, trial):
        reasons = []
        for ch in np.where(self.rej_chans[trial])[1]:
            reasons.append(f"{self.chan_labels[ch]}: {self.rej_reasons[trial, ch]}")
        return reasons

    def preprocess_data_for_plot(self, downscale=None, chan_offset=None):

        if downscale is None:
            downscale = self.downscale
        if chan_offset is None:
            chan_offset = self.chan_offset

        self.offset_dict = {"eeg": 0, "eog": chan_offset, "eyegaze": chan_offset * 2, "misc": chan_offset * 5}  # must be in order and increasing

        epochs_raw = self.epochs_raw.copy()

        epochs_raw = epochs_raw[:, self.chan_order]

        epochs_raw *= self.extra_chan_scale

        self.epochs_pre = np.full(epochs_raw.shape, np.nan)

        self.ys = []
        for ichan in range(epochs_raw.shape[1]):
            extra_offset = self.offset_dict[self.chan_types[ichan]]
            downscale_factor = downscale[self.chan_types[ichan]]
            self.epochs_pre[:, ichan] = (epochs_raw[:, ichan] * downscale_factor) - chan_offset * ichan - extra_offset
            self.ys.append(-1 * chan_offset * ichan - extra_offset)

        self.ylim = (
            np.nanpercentile(self.epochs_pre[:, -1], 5) - chan_offset * 1.5,  # 5th percentile of lowest channel + 2 chans
            np.nanpercentile(self.epochs_pre[:, 0], 95) + chan_offset * 1.5,  # 95th percentile of highest + 2 chans
        )  

    def open_figure(self, color="white"):
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(f'EEG Viewer - Subject {self.sub} (press H for help)')

        self.plot_pos(0)
        axis_position = plt.axes([0.2, -0.1, 0.65, 0.03], facecolor=color)
        self.slider = Slider(axis_position, "Pos", 0, self.epochs_pre.shape[0], valstep=1)
        self.fig.canvas.mpl_connect("key_press_event", self.keypress_event)
        self.fig.canvas.mpl_connect("button_press_event", self.click_toggle)

        # make a new axis for the help window

        self.help_ax = plt.axes()

        self.help_ax.set_title('Keyboard Shortcuts',size=40)
        self.help_ax.text(0.5,1,"\n h: Hide and show this window \n" +
                "[ and ]: Change window size \n" +
                "+ and -: Change channel scale \n"+
                "r: Show rejection reasons \n"+
                "w: Save annotations \n",
                horizontalalignment='center',verticalalignment='top',transform=self.help_ax.transAxes,size=20)
        self.help_ax.set_axis_off()
        self.help_ax.set_visible(False)

    def plot_pos(self, pos):
        self.rej_reasons_on = False

        self.ax.plot(np.concatenate(self.epochs_pre[pos: pos + self.win_step], 1).T, color="#444444", linewidth=.75)

        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.vlines(np.arange(self.epoch_len, self.epoch_len * self.win_step, self.epoch_len), -1, 1, "#000000", linewidths=3)  # Divide Epochs
        self.ax.vlines(
            np.arange((self.rejection_time[0] - self.trial_start) * self.srate, self.epoch_len * self.win_step, self.epoch_len),
            -1,
            1,
            "#0000FF",
            linewidths=1.5,
        )  # baseline start
        self.ax.vlines(np.arange((-self.trial_start) * self.srate, self.epoch_len * self.win_step, self.epoch_len), -1, 1, "#FF00FF")  # Task start
        self.ax.vlines(
            np.arange((self.rejection_time[1] - self.trial_start) * self.srate, self.epoch_len * self.win_step, self.epoch_len),
            -1,
            1,
            "#0000FF",
            linewidths=1.5,
        )  # end of delay

        self.ax.set_yticks(self.ys, self.chan_labels)
        # self.ax.set_xticks(np.arange(0,self.epoch_len,0.1 *0 1000/self.srate),np.tile(np.arange(self.trial_start,self.trial_end,0.1),self.win_step))
        # self.ax.tick_params(bottom=False,labelbottom=False)
        self.ax.set_xticks(
            sorted(
                np.concatenate(
                    (
                        np.arange((self.rejection_time[0] - self.trial_start) * self.srate, self.epoch_len * self.win_step, self.epoch_len),
                        np.arange((self.rejection_time[1] - self.trial_start) * self.srate, self.epoch_len * self.win_step, self.epoch_len),
                    )
                )
            )
        )
        self.ax.set_xticklabels(([int(self.rejection_time[0] * 1000)] + [int(self.rejection_time[1] * 1000)]) * self.win_step)

        for i, epoch in enumerate(range(pos, pos + self.win_step)):
            # annotate with condition labels
            # self.ax.annotate(self.condition_dict[self.conditions[epoch]], (i * self.epoch_len, self.ylim[1] + 2 * CHAN_OFFSET), annotation_clip=False)
            # self.ax.annotate(f"Trial {epoch}", (i * self.epoch_len + self.epoch_len / 2, self.ylim[1] + 2 * CHAN_OFFSET), annotation_clip=False)
            self.ax.annotate(
                f"Trial {epoch}\n{self.condition_dict[self.conditions[epoch]]}",
                (i * self.epoch_len + self.epoch_len / 2, self.ylim[1] + 1.05 * CHAN_OFFSET),
                annotation_clip=False,
                ha="center",
            )
            if self.rej_manual[epoch]:
                self.ax.plot(
                    np.arange(i * self.epoch_len, (i + 1) * self.epoch_len + 1), self.epochs_pre[epoch, self.rej_chans[epoch]].T, color="#FF0000", linewidth=1
                )
                self.ax.fill_between([i * self.epoch_len, (i + 1) * self.epoch_len + 1], [self.ylim[0]], [self.ylim[1]], color="#edb74a", alpha=0.4, zorder=-10)
        # TODO: gray out ignored channels

    def rejection_reasons(self,force_show=False):
        '''
        function to show and hide rejection reasons
        '''

        # print(force_show)

        if not self.rej_reasons_on:

            for i in range(self.win_step):
                trial = self.slider.val + i

                for ch in np.where(self.rej_chans[trial])[0]:
                    self.ax.annotate(f'{self.chan_labels[ch]}: {self.rej_reasons[trial,ch]} (R)',(i * self.epoch_len,self.ys[ch]),backgroundcolor='white',annotation_clip=False)
            self.rej_reasons_on = True

        else:

            for child in self.ax.get_children():
                if isinstance(child,Annotation) and '(R)' in child.get_text():
                    child.remove()
            self.rej_reasons_on = False
        self.fig.canvas.draw_idle()        

    def update(self, force=False):
        pos = self.slider.val
        if pos < 0:
            self.slider.set_val(0)
            self.update()
        elif pos > self.epochs_pre.shape[0] - self.win_step:
            self.slider.set_val(self.epochs_pre.shape[0] - self.win_step)
            self.update()
        else:
            self.ax.clear()
            self.plot_pos(pos)
            if force:
                self.fig.canvas.draw_idle()
            else:
                self.fig.canvas.blit(self.ax.bbox)

    def keypress_event(self, ev):
        match ev.key:
            case "right":
                self.slider.set_val(self.slider.val + self.win_step)
                self.update()
            case "left":
                self.slider.set_val(self.slider.val - self.win_step)
                self.update()
            case ".":
                self.slider.set_val(self.slider.val + 1)
                self.update()
            case ",":
                self.slider.set_val(self.slider.val - 1)
                self.update()
            case "[":
                if self.win_step > 1:
                    self.win_step -= 1
                    self.xlim = (0, self.epoch_len * self.win_step)
                    self.update(force=True)
            case "]":
                self.win_step += 1
                self.xlim = (0, self.epoch_len * self.win_step)
                self.update(force=True)

            case "+":
                self.extra_chan_scale += 0.2
                self.chan_offset *= 2
                self.preprocess_data_for_plot()
                self.update(force=True)
            case "-":
                self.extra_chan_scale -= 0.2
                self.chan_offset /= 2
                self.preprocess_data_for_plot()
                self.update(force=True)
            case "w":
                self.save_annotations()

            case "r":
                self.rejection_reasons()
            case "h":
                self.help_ax.set_visible(not self.help_ax.get_visible())
                self.ax.set_visible(not self.ax.get_visible())
                self.fig.canvas.draw_idle()
            case _:
                print(f"key not recognized: {ev.key}. Press h for help.")

    def click_toggle(self, ev):

        if ev.button is MouseButton.LEFT and ev.xdata is not None:
            pos = self.slider.val

            epoch_index = np.arange(pos, pos + self.win_step)[int(ev.xdata // self.epoch_len)]
            self.rej_manual[epoch_index] = not self.rej_manual[epoch_index]
            self.update(force=True)

    def save_annotations(self):
        print(f'{np.sum(self.rej_manual)}/{len(self.rej_manual)} trials rejected. Saving annotations as ".../{self.sub}_rej_FINAL.npy"')
        np.save(os.path.join(self.parent_dir, self.sub, f"{self.sub}_rej_FINAL.npy"), self.rej_manual)
