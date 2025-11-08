import time
import numpy as np

import os
import sys
import shutil

from datetime import datetime

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.math import floatMatrix

# Use TensorFlow summary writer (tf.summary) for TensorBoard logging only. Import is optional.
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except Exception:
    tf = None
    _TF_AVAILABLE = False


class FlightLogger:
    CREATED: bool = False

    def reset(self, unique_name: str, run_id: int, map: floatMatrix = None) -> None:
        # Close any existing loggers
        self.exit()
        # Re-initialize
        self.__init__(unique_name, run_id, map)

    def deleteLastFlightLog(self) -> None:
        # remove directory and all contents
        try:
            shutil.rmtree(self.origin)
        except Exception:
            print(f"Failed to delete flight log directory {self.origin}")

    def __init__(self, unique_name: str, run_id: int, map: floatMatrix = None) -> None:
        origin = f"flightData/{unique_name}/run_{run_id}"
        os.makedirs(origin, exist_ok=True)
        self.origin = origin
        self.map_file = os.path.join(origin, "environment_map.npy")

        # TensorBoard writer (TensorFlow-only)
        self.tb_writer = None
        try:
            if not _TF_AVAILABLE:
                raise RuntimeError('TensorFlow not available')
            self.tb_logdir = os.path.join(origin, "tensorboard")
            os.makedirs(self.tb_logdir, exist_ok=True)
            # create a TF summary writer
            self.tf_writer = tf.summary.create_file_writer(self.tb_logdir)

            if not self.CREATED:
                self.general_tf_writer = tf.summary.create_file_writer(f"flightData/{unique_name}/general_tensorboard")
                self.CREATED = True
                print(f"Created general TensorBoard log at flightData/{unique_name}/general_tensorboard")

            # Wrapper to provide add_text/add_scalar/add_histogram and flush/close
            class _TFWrapper:
                def __init__(self, tf_mod, writer):
                    self._tf = tf_mod
                    self._writer = writer

                def add_text(self, tag, text, step=None):
                    try:
                        with self._writer.as_default():
                            # tf.summary.text expects a tensor or string and requires step
                            self._tf.summary.text(tag, self._tf.convert_to_tensor([text]), step=step)
                            self._writer.flush()
                    except Exception:
                        pass

                def add_scalar(self, tag, value, step=None):
                    try:
                        with self._writer.as_default():
                            self._tf.summary.scalar(tag, value, step=step)
                            self._writer.flush()
                    except Exception:
                        pass

                def add_histogram(self, tag, values, step=None):
                    try:
                        with self._writer.as_default():
                            self._tf.summary.histogram(tag, values, step=step)
                            self._writer.flush()
                    except Exception:
                        pass

                def flush(self):
                    try:
                        self._writer.flush()
                    except Exception:
                        pass

                def close(self):
                    try:
                        # TF writer doesn't have a close method in some versions; flush is enough
                        self._writer.flush()
                    except Exception:
                        pass

            self.tb_writer = _TFWrapper(tf, self.tf_writer)
            self.general_tb_writer = _TFWrapper(tf, self.general_tf_writer)
            # write initial info about the run (no CSV/TXT files; map and TB path)
            try:
                self.tb_writer.add_text('flight/info', (
                    f"Run folder: {self.origin}\n"
                    f"TensorBoard dir: {self.tb_logdir}\n"
                    f"Map File: {self.map_file}\n"
                ), step=0)
            except Exception:
                pass
        except Exception:
            self.tb_writer = None

        if map is not None and not os.path.exists(self.map_file):
            np.save(self.map_file, map)

        # counters used for tensorboard step indices
        self.iFlightLog = 0
        self.iLossLog = 0
        self.iCommandsLog = 0

    def addTestResults(self, testStep: int, successRate: float, failuresByBounce: int, failuresByTimeout: int, averageReward: float = 0.0, averageDiscountedSum: float = 0.0) -> None:
        if self.general_tb_writer is not None:
            try:
                step = int(self.iFlightLog)
                if hasattr(self.general_tb_writer, 'add_scalar'):
                    self.general_tb_writer.add_scalar('test/success_rate', float(successRate), testStep)
                    self.general_tb_writer.add_scalar('test/failures_by_bounce', float(failuresByBounce), testStep)
                    self.general_tb_writer.add_scalar('test/failures_by_timeout', float(failuresByTimeout), testStep)
                    self.general_tb_writer.add_scalar('test/average_reward', float(averageReward), testStep)
                    self.general_tb_writer.add_scalar('test/average_discounted_sum', float(averageDiscountedSum), testStep)
            except Exception:
                pass

    def addReward(self, timeFloat: float, reward: float, averageReward: float = 0.0, averageCriticLoss: float = 0.0, averageActorLoss: float = 0.0, averageTemperatureLoss: float = 0.0, averageTemperature: float = 0.0, n_bounced: int = 0, discountedSum: float = 0.0, actionSmallerOddForPID: float = 0.5) -> None:
        if self.general_tb_writer is not None:
            try:
                step = int(self.iFlightLog)
                if hasattr(self.general_tb_writer, 'add_scalar'):
                    self.general_tb_writer.add_scalar('reward/step_reward', float(reward), timeFloat)
                    self.general_tb_writer.add_scalar('reward/average_reward', float(averageReward), timeFloat)
                    self.general_tb_writer.add_scalar('loss/average_critic_loss', float(averageCriticLoss), timeFloat)
                    self.general_tb_writer.add_scalar('loss/average_actor_loss', float(averageActorLoss), timeFloat)
                    self.general_tb_writer.add_scalar('loss/average_temperature_loss', float(averageTemperatureLoss), timeFloat)
                    self.general_tb_writer.add_scalar('loss/average_temperature', float(averageTemperature), timeFloat)
                    self.general_tb_writer.add_scalar('loss/number_bounced', float(n_bounced), timeFloat)
                    self.general_tb_writer.add_scalar('reward/discounted_sum', float(discountedSum), timeFloat)
                    self.general_tb_writer.add_scalar('settings/action_smaller_odd_for_pid', float(actionSmallerOddForPID), timeFloat)
            except Exception:
                pass

    def flightLog(self, timeFloat: float, position: floatMatrix, velocity: floatMatrix, orientation: floatMatrix, angular_velocity: floatMatrix, target_position: floatMatrix, target_velocity: floatMatrix):
        # increment counter (used as tensorboard step)
        self.iFlightLog += 1

        # Also log to TensorBoard when available. We'll log scalars for each component
        # and histograms for vector/matrix data for occasional inspection.
        if self.tb_writer is not None:
            try:
                step = int(self.iFlightLog)
                # time is a scalar
                if hasattr(self.tb_writer, 'add_scalar'):
                    self.tb_writer.add_scalar('flight/time', float(timeFloat), step)

                def _log_array(tag_base: str, arr: floatMatrix):
                    try:
                        a = np.asarray(arr)
                        # If it's a scalar or 1D, log components as scalars
                        if a.ndim == 0 or a.size == 1:
                            self.tb_writer.add_scalar(f'{tag_base}', float(a.ravel()[0]), step)
                        elif a.ndim == 1 or a.ndim == 2 and a.shape[0] == 1:
                            # log each element as tag_base/0, tag_base/1 ...
                            for i, v in enumerate(a.ravel()):
                                self.tb_writer.add_scalar(f'{tag_base}/{i}', float(v), step)
                        else:
                            # For larger arrays, add a histogram (may be ignored if unsupported)
                            if hasattr(self.tb_writer, 'add_histogram'):
                                try:
                                    self.tb_writer.add_histogram(tag_base, a, step)
                                except Exception:
                                    # fallback: add a scalar for mean
                                    self.tb_writer.add_scalar(f'{tag_base}/mean', float(np.mean(a)), step)
                    except Exception:
                        pass

                _log_array('flight/position', position)
                _log_array('flight/velocity', velocity)
                _log_array('flight/orientation', orientation)
                _log_array('flight/angular_velocity', angular_velocity)
                _log_array('flight/target_position', target_position)
                _log_array('flight/target_velocity', target_velocity)
            except Exception:
                # Don't let TB logging break main flow
                pass

    def lossLog(self, timeFloat: float, actorLoss: float, criticLoss1: float, criticLoss2: float, temperatureLoss: float, temperature: float = 0.0):
        # increment counter (used as tensorboard step)
        self.iLossLog += 1

        # Log losses to TensorBoard
        if self.tb_writer is not None:
            try:
                step = int(self.iFlightLog)
                if hasattr(self.tb_writer, 'add_scalar'):
                    self.tb_writer.add_scalar('loss/time', float(timeFloat), step)
                    self.tb_writer.add_scalar('loss/actor_loss', float(actorLoss), step)
                    self.tb_writer.add_scalar('loss/critic_loss1', float(criticLoss1), step)
                    self.tb_writer.add_scalar('loss/critic_loss2', float(criticLoss2), step)
                    self.tb_writer.add_scalar('loss/temperature_loss', float(temperatureLoss), step)
                    self.tb_writer.add_scalar('loss/temperature', float(temperature), step)
            except Exception:
                pass

    def commandLog(self, timeFloat: float, X: float, Y: float, Z: float, L: float, M: float, N: float, action1: float = 0.0, action2: float = 0.0, action3: float = 0.0):
        # increment counter (used as tensorboard step)
        self.iCommandsLog += 1

        # Log commands to TensorBoard
        if self.tb_writer is not None:
            try:
                step = int(self.iCommandsLog)
                if hasattr(self.tb_writer, 'add_scalar'):
                    self.tb_writer.add_scalar('commands/time', float(timeFloat), step)
                    self.tb_writer.add_scalar('commands/X', float(X), step)
                    self.tb_writer.add_scalar('commands/Y', float(Y), step)
                    self.tb_writer.add_scalar('commands/Z', float(Z), step)
                    self.tb_writer.add_scalar('commands/L', float(L), step)
                    self.tb_writer.add_scalar('commands/M', float(M), step)
                    self.tb_writer.add_scalar('commands/N', float(N), step)
            except Exception:
                pass

    def writeInfo(self, info: str):
        print(info)
        # write info to tensorboard as text (append)
        if self.tb_writer is not None and hasattr(self.tb_writer, 'add_text'):
            try:
                t = time.time()
                self.tb_writer.add_text('flight/info_append', f"{time.ctime(t)}: {info}", int(t))
            except Exception:
                pass

    def exit(self):
        # close tensorboard writer if present
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:
                pass