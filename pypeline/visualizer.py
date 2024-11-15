import numpy as np
import mne
import mne_bids
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton
from matplotlib.text import Annotation
import pandas as pd


# FOR PLOTTING
SLIDER_STEP = 5  #


EYETRACK_SCALE = 1e-6  # what factor to downscale eyetracking by
CHAN_OFFSET = 0.00005


class Visualizer:
    def __init__(
        self,
        sub,
        parent_dir: str,
        srate: float,
        trial_start: float,
        trial_end: float,
        experiment_name: str,
        rejection_time=(None, None),
        win_step: int = SLIDER_STEP,
        downscale={"eyegaze": EYETRACK_SCALE},
        chan_offset=CHAN_OFFSET,
        channels_drop=None,
        channels_ignore=None,
        load_flags=True,
    ):
        self.sub = sub
        self.parent_dir = parent_dir
        self.experiment_name = experiment_name
        self.srate = srate
        self.trial_start = trial_start
        self.trial_end = trial_end
        self.win_step = win_step
        self.epoch_len = np.ceil((trial_end - trial_start) * srate)
        rejection_time[0] = trial_start if rejection_time[0] is None else rejection_time[0]
        rejection_time[1] = trial_end if rejection_time[1] is None else rejection_time[1]
        self.rejection_time = rejection_time
        self.downscale = downscale
        self.chan_offset = chan_offset

        self.data_path = mne_bids.path.BIDSPath(
            subject=sub,
            task=self.experiment_name,
            description="preprocessed",
            root=os.path.join(self.parent_dir, "derivatives"),
            datatype="eeg",
            check=False,
        )

        self.data_path.update(suffix="eeg", extension=".fif")  # load in preprocessed data
        self.epochs_obj = mne.read_epochs(self.data_path.fpath)

        self.data_path.update(suffix="events", extension=".tsv")  # load in events

        self.events = pd.read_csv(self.data_path.fpath, sep="\t")
        ## EEG Port codes
        all_port_codes = pd.read_csv(
            self.data_path.copy()
            .update(root=self.data_path.root.parent, description=None, suffix="events", extension="tsv")
            .fpath,
            sep="\t",
        )

        # re-index port code timings for each trial

        self.all_codes = []
        self.all_times = []

        for _, row in self.events.iterrows():
            trial_sample = row["sample"]
            trial_codes = all_port_codes[
                (all_port_codes["sample"] > trial_sample + self.trial_start * 1000)
                & (all_port_codes["sample"] < trial_sample + self.trial_end * 1000)
            ]
            # get codes which occured in a (trial_start, trial_end) sample around each trial's timelock event
            code_times = np.array(trial_codes["sample"])
            code_times -= trial_sample
            code_times -= int(self.trial_start * 1000)
            self.all_codes.append(trial_codes["value"].tolist())
            self.all_times.append(code_times.tolist())

        if len(self.all_codes) != len(self.events):
            raise RuntimeError(
                f"Error: could not find port codes for all trials. There are {len(self.events)} trials and {len(self)} sets of codes at corresponding times"
            )

        self.data_path.update(suffix="artifacts", extension=".tsv")  # load in artifacts and apply to channels
        rej = pd.read_csv(self.data_path.fpath, sep="\t", keep_default_na=False)
        self.rej_chans = rej.apply(np.vectorize(lambda x: len(x) > 0)).to_numpy()
        self.rej_reasons = rej.to_numpy()

        if channels_drop is not None:  # drop ignored channels
            channels_drop = [ch for ch in channels_drop if ch in self.epochs_obj.ch_names]
            if len(channels_drop) > 0:
                self.rej_chans = self.rej_chans[:, ~np.in1d(self.epochs_obj.ch_names, channels_drop)]
                self.rej_reasons = self.rej_reasons[:, ~np.in1d(self.epochs_obj.ch_names, channels_drop)]
                self.epochs_obj.drop_channels(channels_drop)

        if load_flags:  # manually load any previously saved rejection flags
            self.data_path.update(suffix="rejection_flags", extension=".npy")
            try:
                self.rej_manual = np.load(self.data_path.fpath)
                print("You have saved annotations already. Loading these.")
            except FileNotFoundError:
                print("No saved annotations found, resetting to default.")
                self.rej_manual = self.rej_chans.any(1)
        else:
            self.rej_manual = self.rej_chans.any(1)

        self.info = self.epochs_obj.info
        self.chan_types = np.array(self.info.get_channel_types())
        self.chan_labels = np.array(self.epochs_obj.ch_names)

        self.channels_ignore = channels_ignore  # make a mask for channels we ignore
        self.ignored_channels_mask = np.in1d(self.chan_labels, channels_ignore)

        if self.rej_chans.shape[1] != self.ignored_channels_mask.shape[0]:
            raise ValueError(
                f"There are {self.rej_chans.shape[1]} channels in the rejection labels,\
                             but {self.ignored_channels_mask.shape[0]} channels in the data. \
                             Please make sure that any channels without artifact labels are dropped"
            )

        if channels_ignore is not None:  # never reject ignored channels (eg EOG)
            self.rej_chans[:, self.ignored_channels_mask] = False
            self.rej_reasons[:, self.ignored_channels_mask] = None

        self.epochs_raw = self.epochs_obj.get_data(copy=True)
        self.epochs_pre = None  # initialized when we preprocess

        self.offset_dict = None
        self.xlim = (0, self.epoch_len * self.win_step)

        self.ylim = [None, None]
        self.stack = False

        self.pos = 0
        self.extra_chan_scale = 1

        self.rej_reasons_on = False
        self.port_codes_on = False

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

        self.offset_dict = {
            "eeg": 0,
            "eog": chan_offset,
            "eyegaze": chan_offset * 2,
            "misc": chan_offset * 5,
        }  # must be in order and increasing

        epochs_raw = self.epochs_raw.copy()

        epochs_raw *= self.extra_chan_scale

        self.epochs_pre = np.full(epochs_raw.shape, np.nan)

        self.ys = []
        for ichan in range(epochs_raw.shape[1]):
            extra_offset = self.offset_dict[self.chan_types[ichan]]
            downscale_factor = downscale[self.chan_types[ichan]]
            self.epochs_pre[:, ichan] = (epochs_raw[:, ichan] * downscale_factor) - chan_offset * ichan - extra_offset
            self.ys.append(-1 * chan_offset * ichan - extra_offset)
        self.ys = np.array(self.ys)

        self.ylim = (
            np.nanpercentile(self.epochs_pre[:, -1], 5)
            - chan_offset * 1.5,  # 5th percentile of lowest channel + 2 chans
            np.nanpercentile(self.epochs_pre[:, 0], 95) + chan_offset * 1.5,  # 95th percentile of highest + 2 chans
        )

        self.offset_dict_stacked = {
            "eeg": -self.ys[self.chan_types == "eeg"][-3],  # last 3rd EEG position
        }
        if np.sum(self.chan_types == "eog") > 0:
            self.offset_dict_stacked["eog"] = -self.ys[
                self.chan_types == "eog"
            ].mean()  # average of all EOG y positions
        if np.sum(self.chan_types == "eyegaze") > 0:
            self.offset_dict_stacked["eyegaze_x"] = -self.ys[self.chan_types == "eyegaze"][0]  # TOP of eye gaze
            self.offset_dict_stacked["eyegaze_y"] = -self.ys[self.chan_types == "eyegaze"][-1]  # BOTTOM of eye gaze
        if np.sum(self.chan_types == "misc") > 0:
            self.offset_dict_stacked["misc"] = -self.ys[self.chan_types == "misc"][-1]

        self.epochs_stacked = np.full(epochs_raw.shape, np.nan)
        self.ys_stacked = []
        for ichan in range(epochs_raw.shape[1]):

            if self.chan_types[ichan] == "eyegaze":
                if "x" in self.chan_labels[ichan]:
                    extra_offset = self.offset_dict_stacked["eyegaze_x"]
                elif "y" in self.chan_labels[ichan]:
                    extra_offset = self.offset_dict_stacked["eyegaze_y"]
                else:
                    raise ValueError('Eyegaze channels must be labeled with "x" or "y"')
            else:
                extra_offset = self.offset_dict_stacked[self.chan_types[ichan]]

            downscale_factor = downscale[self.chan_types[ichan]]
            self.epochs_stacked[:, ichan] = (epochs_raw[:, ichan] * downscale_factor) - extra_offset
            self.ys_stacked.append(-extra_offset)
        self.ys_stacked = np.array(self.ys_stacked)
        self.ylim_stacked = (
            np.nanpercentile(self.epochs_stacked[:, -1], 5)
            - chan_offset * 1.5,  # 5th percentile of lowest channel + 2 chans
            np.nanpercentile(self.epochs_stacked[:, 0], 95) + chan_offset * 1.5,  # 95th percentile of highest + 2 chans
        )

    def open_figure(self, color="white"):

        self.rej_reasons_on = False
        self.port_codes_on = False

        self.stack = False
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(f"EEG Viewer - Subject {self.sub} (press H for help)")

        self.plot_pos(0)
        axis_position = plt.axes([0.2, -0.1, 0.65, 0.03], facecolor=color)
        self.slider = Slider(axis_position, "Pos", 0, self.epochs_pre.shape[0], valstep=1)
        self.fig.canvas.mpl_connect("key_press_event", self.keypress_event)
        self.fig.canvas.mpl_connect("button_press_event", self.click_toggle)

        # make a new axis for the help window

        self.help_ax = plt.axes()

        self.help_ax.set_title("Keyboard Shortcuts", size=40)
        self.help_ax.text(
            0.5,
            1,
            "\n h: Hide and show this window \n"
            + "[ and ]: Change window size \n"
            + "+ and -: Change channel scale \n"
            + "r: Show rejection reasons \n"
            + "p: Show port codes \n"
            + "c: Stack channels \n"
            + "w: Save annotations \n",
            horizontalalignment="center",
            verticalalignment="top",
            transform=self.help_ax.transAxes,
            size=20,
        )
        self.help_ax.set_axis_off()
        self.help_ax.set_visible(False)

    def plot_channels(self, epochs, pos):
        """
        Helper function to plot channel data

        Args:
            epochs: numpy array of shape (trials (n epochs to plot), channels, timepoints)
            pos: int, the position of the slider
        """
        self.ax.plot(
            np.concatenate(epochs[pos : pos + self.win_step, ~self.ignored_channels_mask], 1).T,
            color="#000000",
            linewidth=0.75,
        )  # good channels

        self.ax.plot(
            np.concatenate(epochs[pos : pos + self.win_step, self.ignored_channels_mask], 1).T,
            color="#666666",
            linewidth=0.75,
        )  # ignored channels in gray

        for i, epoch in enumerate(range(pos, pos + self.win_step)):
            # annotate with condition labels

            self.ax.annotate(
                f"Trial {epoch}\n{self.events['trial_type'][epoch]}",
                (
                    i * self.epoch_len + self.epoch_len / 2,
                    self.ylim[1] + 1.05 * CHAN_OFFSET,
                ),
                annotation_clip=False,
                ha="center",
            )
            if self.rej_manual[epoch]:
                self.ax.plot(
                    np.arange(i * self.epoch_len, (i + 1) * self.epoch_len + 1),
                    epochs[epoch, self.rej_chans[epoch]].T,
                    color="#FF0000",
                    linewidth=1,
                )
                self.ax.fill_between(
                    [i * self.epoch_len, (i + 1) * self.epoch_len + 1],
                    [self.ylim[0]],
                    [self.ylim[1]],
                    color="#edb74a",
                    alpha=0.4,
                    zorder=-10,
                )

    def plot_helper_lines(self):
        """
        Helper function to plot all the vertical lines to denote trial start, etc
        (Aesthetics)
        """
        self.ax.vlines(
            np.arange(self.epoch_len, self.epoch_len * self.win_step, self.epoch_len),
            -1,
            1,
            "#000000",
            linewidths=3,
        )  # Divide Epochs
        self.ax.vlines(
            np.arange(
                (self.rejection_time[0] - self.trial_start) * self.srate,
                self.epoch_len * self.win_step,
                self.epoch_len,
            ),
            -1,
            1,
            "#0000FF",
            linewidths=1.5,
        )  # baseline start
        self.ax.vlines(
            np.arange(
                (-self.trial_start) * self.srate,
                self.epoch_len * self.win_step,
                self.epoch_len,
            ),
            -1,
            1,
            "#FF00FF",
        )  # Task start
        self.ax.vlines(
            np.arange(
                (self.rejection_time[1] - self.trial_start) * self.srate,
                self.epoch_len * self.win_step,
                self.epoch_len,
            ),
            -1,
            1,
            "#0000FF",
            linewidths=1.5,
        )  # end of delay

    def plot_pos(self, pos):
        """
        Primary plotting function:

        Args:
            pos: int, the position of the slider
        """

        self.plot_helper_lines()

        if self.stack:
            self.plot_channels(self.epochs_stacked, pos)
            self.ax.set_ylim(*self.ylim_stacked)
            self.ax.set_yticks([y * -1 for y in self.offset_dict_stacked.values()], self.offset_dict_stacked.keys())

        else:
            self.plot_channels(self.epochs_pre, pos)
            self.ax.set_ylim(*self.ylim)
            self.ax.set_yticks(self.ys, self.chan_labels)

        self.ax.set_xlim(*self.xlim)

        # self.ax.set_xticks(np.arange(0,self.epoch_len,0.1 *0 1000/self.srate),np.tile(np.arange(self.trial_start,self.trial_end,0.1),self.win_step))
        # self.ax.tick_params(bottom=False,labelbottom=False)
        # Plot time labels

        if self.rejection_time[0] == self.trial_start and self.rejection_time[1] == self.trial_end:
            pass
        elif self.rejection_time[0] == self.trial_start:
            self.ax.set_xticks(
                np.arange(
                    (self.rejection_time[1] - self.trial_start) * self.srate,
                    self.epoch_len * self.win_step,
                    self.epoch_len,
                )
            )
            self.ax.set_xticklabels([int(self.rejection_time[1] * 1000)] * self.win_step)

        elif self.rejection_time[1] == self.trial_end:
            self.ax.set_xticks(
                np.arange(
                    (self.rejection_time[0] - self.trial_start) * self.srate,
                    self.epoch_len * self.win_step,
                    self.epoch_len,
                )
            )
            self.ax.set_xticklabels([int(self.rejection_time[0] * 1000)] * self.win_step)

        else:
            self.ax.set_xticks(
                sorted(
                    np.concatenate(
                        (
                            np.arange(
                                (self.rejection_time[0] - self.trial_start) * self.srate,
                                self.epoch_len * self.win_step,
                                self.epoch_len,
                            ),
                            np.arange(
                                (self.rejection_time[1] - self.trial_start) * self.srate,
                                self.epoch_len * self.win_step,
                                self.epoch_len,
                            ),
                        )
                    )
                )
            )
            self.ax.set_xticklabels(
                [int(self.rejection_time[0] * self.srate), int(self.rejection_time[1] * 1000)] * self.win_step
            )

    def show_rejection_reasons(self):
        """
        function to show and hide rejection reasons
        """
        self.rej_annotations = []

        for i in range(self.win_step):
            trial = self.slider.val + i

            for ch in np.where(self.rej_chans[trial])[0]:
                an = self.ax.annotate(
                    f"{self.chan_labels[ch]}: {self.rej_reasons[trial,ch]} (R)",
                    (i * self.epoch_len, self.ys[ch]),
                    backgroundcolor="white",
                    annotation_clip=False,
                )
                self.rej_annotations.append(an)

    def show_port_codes(self):
        """
        Function to show port codes for trials on screen
        """

        self.code_annotations = []
        all_codes = []
        all_times = []
        for i in range(self.win_step):
            times = self.all_times[self.slider.val + i]
            all_times.extend([t + i * self.epoch_len for t in times])
            all_codes.extend(self.all_codes[self.slider.val + i])

        self.code_lines = self.ax.vlines(all_times, *self.ylim, color="g")

        for code, time in zip(all_codes, all_times):
            an = self.ax.annotate(code, (time, self.ylim[1] + 5e-6), ha="center", annotation_clip=False)
            self.code_annotations.append(an)

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
            if self.rej_reasons_on:
                self.show_rejection_reasons()
            if self.port_codes_on:
                self.show_port_codes()
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
                if self.rej_reasons_on:
                    for an in self.rej_annotations:
                        an.remove()
                    self.rej_reasons_on = False
                    self.fig.canvas.draw_idle()
                else:
                    self.show_rejection_reasons()
                    self.rej_reasons_on = True
                    self.fig.canvas.draw_idle()

            case "p":
                if self.port_codes_on:
                    for an in self.code_annotations:
                        an.remove()
                    self.code_lines.remove()
                    self.port_codes_on = False

                    self.fig.canvas.draw_idle()
                else:
                    self.port_codes_on = True
                    self.show_port_codes()
                    self.fig.canvas.draw_idle()

            case "c":
                self.stack = not self.stack
                self.update(force=True)
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
        self.data_path.update(suffix="rejection_flags", extension=".npy")
        print(
            f'{np.sum(self.rej_manual)}/{len(self.rej_manual)} trials rejected. Saving annotations as "{self.data_path.fpath}"'
        )
        np.save(self.data_path.fpath, self.rej_manual)
