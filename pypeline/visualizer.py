import numpy as np
import mne
import mne_bids
import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton
from matplotlib.text import Annotation
import pandas as pd
import re
from pathlib import Path


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

        self.data_path.update(suffix="eeg", extension=".fif")
        self.epochs_obj = mne.read_epochs(self.data_path.fpath)

        self.data_path.update(suffix="events", extension=".tsv")

        self.conditions = pd.read_csv(self.data_path.fpath, sep="\t")["trial_type"]

        self.data_path.update(suffix="artifacts", extension=".tsv")
        rej = pd.read_csv(self.data_path.fpath, sep="\t", keep_default_na=False)
        self.rej_chans = rej.apply(np.vectorize(lambda x: len(x) > 0)).to_numpy()
        self.rej_reasons = rej.to_numpy()

        # self.conditions = np.load(os.path.join(parent_dir, sub, f"{sub}_conditions.npy"))
        # self.rej_chans = np.load(os.path.join(parent_dir, sub, f"{sub}_rej.npy"))
        # self.rej_reasons = np.load(os.path.join(parent_dir, sub, f"{sub}_rej_reasons.npy"), allow_pickle=True)

        if channels_drop is not None:
            channels_drop = [ch for ch in channels_drop if ch in self.epochs_obj.ch_names]
            if len(channels_drop) > 0:
                self.rej_chans = self.rej_chans[:, ~np.in1d(self.epochs_obj.ch_names, channels_drop)]
                self.rej_reasons = self.rej_reasons[:, ~np.in1d(self.epochs_obj.ch_names, channels_drop)]
                self.epochs_obj.drop_channels(channels_drop)

        self.channels_ignore = channels_ignore
        self.ignored_channels_mask = np.in1d(self.epochs_obj.ch_names, channels_ignore)

        if self.rej_chans.shape[1] != self.ignored_channels_mask.shape[0]:
            raise ValueError(
                f"There are {self.rej_chans.shape[1]} channels in the rejection labels,\
                             but {self.ignored_channels_mask.shape[0]} channels in the data. \
                             Please make sure that any channels without artifact labels are dropped"
            )

        if channels_ignore is not None:
            self.rej_chans[:, self.ignored_channels_mask] = False
            self.rej_reasons[:, self.ignored_channels_mask] = None

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
        self.stack = False

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

        self.offset_dict = {
            "eeg": 0,
            "eog": chan_offset,
            "eyegaze": chan_offset * 2,
            "misc": chan_offset * 5,
        }  # must be in order and increasing

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
        self.ys = np.array(self.ys)

        self.ylim = (
            np.nanpercentile(self.epochs_pre[:, -1], 5)
            - chan_offset * 1.5,  # 5th percentile of lowest channel + 2 chans
            np.nanpercentile(self.epochs_pre[:, 0], 95) + chan_offset * 1.5,  # 95th percentile of highest + 2 chans
        )

        self.offset_dict_stacked = {
            "eeg": -self.ys[self.chan_types == "eeg"][-3],  # last 3rd EEG position
            "eog": -self.ys[self.chan_types == "eog"].mean(),  # average of all EOG y positions
            "eyegaze_x": -self.ys[self.chan_types == "eyegaze"][0],  # TOP of eye gaze
            "eyegaze_y": -self.ys[self.chan_types == "eyegaze"][-1],  # BOTTOM of eye gaze
        }

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
                f"Trial {epoch}\n{self.conditions[epoch]}",
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
        self.rej_reasons_on = False

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

    def rejection_reasons(self, force_show=False):
        """
        function to show and hide rejection reasons
        """

        # print(force_show)

        if not self.rej_reasons_on:

            for i in range(self.win_step):
                trial = self.slider.val + i

                for ch in np.where(self.rej_chans[trial])[0]:
                    self.ax.annotate(
                        f"{self.chan_labels[ch]}: {self.rej_reasons[trial,ch]} (R)",
                        (i * self.epoch_len, self.ys[ch]),
                        backgroundcolor="white",
                        annotation_clip=False,
                    )
            self.rej_reasons_on = True

        else:

            for child in self.ax.get_children():
                if isinstance(child, Annotation) and "(R)" in child.get_text():
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
