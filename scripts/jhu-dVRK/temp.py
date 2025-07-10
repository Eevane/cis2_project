import csv
import os

### put it in __init__
self.start_time = time.monotonic()
self.recording_enabled = False
self.record_size = 0

self.output_csv_path = f"/home/pshao7/dvrk_python_devel/{self.start_time:.6f}-MTMR-Mul-Test-joint_data.csv"
self.csv_file = open(self.output_csv_path, "a", newline='')
self.csv_writer = csv.writer(self.csv_file)
self.header_written = os.path.getsize(self.output_csv_path) > 0



### put it in run_following
'''For recording'''
current_time = time.monotonic()
print(f"recording enabled: {self.recording_enabled}")
if not self.recording_enabled and float(current_time - self.start_time) >= 10.0:
    print("Start recording joint data")
    self.recording_enabled = True

if self.recording_enabled and self.record_size >= 1500:
    print("Auto stopping: 20 minutes reached.")
    # time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    # time.strftime("%Y-%m-%d %H:%M:%S", self.start_time)
    print(f"start_time: {self.start_time}")
    print(f"end_time: {current_time}")
    self.recording_enabled = False
    self.running = False

if self.recording_enabled:
    self.record_size += 1
    print("Recording data.")

    timestamp = time.time()
    master1_js = self.master1.measured_js()
    master2_js = self.master2.measured_js()
    puppet_js  = self.puppet.measured_js()


    master1_q = list(master1_js.Position[:6])
    master1_dq = list(master1_js.Velocity[:6])
    master1_torque = list(master1_js.Effort[:6])

    master2_q = list(master2_js.Position[:6])
    master2_dq = list(master2_js.Velocity[:6])
    master2_torque = list(master2_js.Effort[:6])

    puppet_q = list(puppet_js.Position)
    puppet_dq = list(puppet_js.Velocity)
    puppet_torque = list(puppet_js.Effort)

    row = [timestamp] + master1_q + master1_dq + master1_torque + master2_q + master2_dq + master2_torque + puppet_q + puppet_dq + puppet_torque

    if not self.header_written:
        headers = ['timestamp'] + \
                [f'master1_q{i}' for i in range(6)] + [f'master1_dq{i}' for i in range(6)] + [f'master1_tau{i}' for i in range(6)] + \
                [f'master2_q{i}' for i in range(6)] + [f'master2_dq{i}' for i in range(6)] + [f'master2_tau{i}' for i in range(6)] + \
                [f'puppet_q{i}' for i in range(6)]  + [f'puppet_dq{i}' for i in range(6)]  + [f'puppet_tau{i}' for i in range(6)]
        self.csv_writer.writerow(headers)
        self.header_written = True

    self.csv_writer.writerow(row)