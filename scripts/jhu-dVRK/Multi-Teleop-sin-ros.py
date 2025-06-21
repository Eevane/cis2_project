#!/usr/bin/env python

# Author: Junxiang Wang
# Date: 2024-04-12

# (C) Copyright 2024-2025 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

""" Multilateral teleoperation single console - ROS2 version """
# modified by Xiangyi Le

import argparse
import crtk
from enum import Enum
import geometry_msgs.msg
import math
import numpy
import PyKDL
import std_msgs.msg
import sys
import time

class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    def __init__(self, ral, mtm1, mtm2, puppet, clutch_topic, run_period, align_mtm, operator_present_topic = "", alpha = 0.5, beta = 0.5):
        print('Initialzing dvrk_teleoperation for {}, {} and {}'.format(mtm1.name, mtm2.name, puppet.name))
        self.ral = ral
        self.run_period = run_period

        self.master1 = mtm1
        self.master2 = mtm2
        self.puppet = puppet

        # dominance factor
        self.alpha = alpha
        self.beta = beta

        self.scale = 0.2
        self.velocity_scale = 0.2

        self.gripper_max = 60 * math.pi / 180
        self.gripper_zero = 0.0 # Set to e.g. 20 degrees if gripper cannot close past zero
        self.jaw_min = -20 * math.pi / 180
        self.jaw_max = 80 * math.pi / 180
        self.jaw_rate = 2 * math.pi

        self.can_align_mtm = align_mtm

        # slowly eliminate alignment offset if we can align mtm,
        # otherwise maintain fixed initial alignment offset
        self.align_rate = 0.25 * math.pi if self.can_align_mtm else 0.0

        # don't require alignment before beginning teleop if mtm wrist can't be actuated
        self.operator_orientation_tolerance = 5 * math.pi / 180 if self.can_align_mtm else math.pi
        self.operator_gripper_threshold = 5 * math.pi / 180
        self.operator_roll_threshold = 3 * math.pi / 180

        self.gripper_to_jaw_scale = self.jaw_max / (self.gripper_max - self.gripper_zero)
        self.gripper_to_jaw_offset = -self.gripper_zero * self.gripper_to_jaw_scale

        self.operator_is_active = False
        if operator_present_topic:
            self.operator_is_present = False
            self.operator_button = crtk.joystick_button(ral, operator_present_topic)
            self.operator_button.set_callback(self.on_operator_present)
        else:
            self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
        self.clutch_button = crtk.joystick_button(ral, clutch_topic)
        self.clutch_button.set_callback(self.on_clutch)

        # for plotting -- don't need now since ROS has plotjuggler
        self.a = 0
        self.time_data = []
        self.y_data_l = []
        self.y_data_l_expected = []
        self.m1_force = []
        self.m2_force = []
        self.puppet_force = []

    # average rotation by quaternion
    def average_rotation(self, rotation1, rotation2, alpha=0.5):
        # transfrom into quaternion
        quat1 = numpy.array(rotation1.GetQuaternion())   # return a tuple
        quat2 = numpy.array(rotation2.GetQuaternion())

        # average and norm
        mean_quat = alpha * quat1 + (1-alpha) * quat2
        mean_quat /= numpy.linalg.norm(mean_quat)

        # # transform into rotation matrix
        # angle = 2 * numpy.arccos(mean_quat[3])
        # s = numpy.sqrt(1 - mean_quat[3] ** 2)

        # if s < 1e-8:
        #     axis = numpy.array([1.0, 0.0, 0.0])
        # else:
        #     axis = numpy.array([mean_quat[0], mean_quat[1], mean_quat[2]]) / s
        



        return PyKDL.Rotation.Quaternion(mean_quat[0], mean_quat[1], mean_quat[2], mean_quat[3])

    # callback for operator pedal/button
    def on_operator_present(self, present):
        self.operator_is_present = present
        if not present:
            self.operator_is_active = False

    # callback for clutch pedal/button
    def on_clutch(self, clutch_pressed):
        self.clutch_pressed = clutch_pressed

    # compute relative orientation of mtm1 and psm
    def alignment_offset_master1(self):
        return self.master1.measured_cp()[0].M.Inverse() * self.puppet.setpoint_cp()[0].M
    
    # compute relative orientation of mtm2 and psm
    def alignment_offset_master2(self):
        return self.master2.measured_cp()[0].M.Inverse() * self.puppet.setpoint_cp()[0].M

    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        self.master1_cartesian_initial = self.master1.measured_cp()[0]
        self.master2_cartesian_initial = self.master2.measured_cp()[0]
        self.puppet_cartesian_initial = self.puppet.setpoint_cp()[0]

        self.alignment_offset_initial_master1 = self.alignment_offset_master1()
        self.alignment_offset_initial_master2 = self.alignment_offset_master2()
        self.master1_offset_angle, self.master1_offset_axis = self.alignment_offset_initial_master1.GetRotAngle()
        self.master2_offset_angle, self.master2_offset_axis = self.alignment_offset_initial_master2.GetRotAngle()

    def gripper_to_jaw(self, gripper_angle):
        jaw_angle = self.gripper_to_jaw_scale * gripper_angle + self.gripper_to_jaw_offset

        # make sure we don't set goal past joint limits
        return max(jaw_angle, self.jaw_min)

    def jaw_to_gripper(self, jaw_angle):
        return (jaw_angle - self.gripper_to_jaw_offset) / self.gripper_to_jaw_scale

    def check_arm_state(self):
        if not self.puppet.is_homed():
            print(f'ERROR: {self.ral.node_name()}: puppet ({self.puppet.name}) is not homed anymore')
            self.running = False
        if not self.master1.is_homed():
            print(f'ERROR: {self.ral.node_name()}: master ({self.master1.name}) is not homed anymore')
            self.running = False
        if not self.master2.is_homed():
            print(f'ERROR: {self.ral.node_name()}: master ({self.master2.name}) is not homed anymore')
            self.running = False

    def enter_aligning(self):
        self.current_state = teleoperation.State.ALIGNING
        self.last_align = None
        self.last_operator_prompt = time.perf_counter()

        self.master1.use_gravity_compensation(True)
        self.master2.use_gravity_compensation(True)
        self.puppet.hold()

        # reset operator activity data in case operator is inactive
        self.operator_roll_min = math.pi * 100
        self.operator_roll_max = -math.pi * 100
        self.operator_gripper_min = math.pi * 100
        self.operator_gripper_max = -math.pi * 100

    def transition_aligning(self):
        if self.operator_is_active and self.clutch_pressed:
            self.enter_clutched()
            return

        master1_orientation_error, _ = self.alignment_offset_master1().GetRotAngle()
        master2_orientation_error, _ = self.alignment_offset_master2().GetRotAngle()
        aligned = master1_orientation_error <= self.operator_orientation_tolerance and master2_orientation_error <= self.operator_orientation_tolerance
        if aligned and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        master1_orientation_error, _ = self.alignment_offset_master1().GetRotAngle()
        master2_orientation_error, _ = self.alignment_offset_master2().GetRotAngle()

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        if self.operator_is_present:
            master1_gripper = self.master1.gripper.measured_js()[0][0]
            master2_gripper = self.master2.gripper.measured_js()[0][0]
            
            master1_gripper_range = max(master1_gripper, self.operator_gripper_max) - min(master1_gripper, self.operator_gripper_min)
            master2_gripper_range = max(master2_gripper, self.operator_gripper_max) - min(master2_gripper, self.operator_gripper_min)
            if master1_gripper_range >= self.operator_gripper_threshold or master2_gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active = True

            # determine amount of roll around z axis by rotation of y-axis
            master1_rotation, master2_rotation, puppet_rotation = self.master1.measured_cp()[0].M, self.master2.measured_cp()[0].M, self.puppet.setpoint_cp()[0].M
            master1_y_axis = PyKDL.Vector(master1_rotation[0,1], master1_rotation[1,1], master1_rotation[2,1])
            master2_y_axis = PyKDL.Vector(master2_rotation[0,1], master2_rotation[1,1], master2_rotation[2,1])
            puppet_y_axis = PyKDL.Vector(puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1])
            roll_1 = math.acos(PyKDL.dot(puppet_y_axis, master1_y_axis))
            roll_2 = math.acos(PyKDL.dot(puppet_y_axis, master2_y_axis))

            master1_roll_range = max(roll_1, self.operator_roll_max) - min(roll_1, self.operator_roll_min)
            master2_roll_range = max(roll_2, self.operator_roll_max) - min(roll_2, self.operator_roll_min)
            if master1_roll_range >= self.operator_roll_threshold or master2_roll_range >= self.operator_roll_threshold:
                self.operator_is_active = True

        # periodically send move_cp to MTM to align with PSM
        master1_aligned = master1_orientation_error <= self.operator_orientation_tolerance
        master2_aligned = master2_orientation_error <= self.operator_orientation_tolerance
        now = time.perf_counter()
        if not self.last_align or now - self.last_align > 4.0:
            master1_move_cp = PyKDL.Frame(self.puppet.setpoint_cp()[0].M, self.master1.setpoint_cp()[0].p)
            master2_move_cp = PyKDL.Frame(self.puppet.setpoint_cp()[0].M, self.master2.setpoint_cp()[0].p)
            self.master1.move_cp(master1_move_cp)
            self.master2.move_cp(master2_move_cp)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if self.operator_is_present and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not master1_aligned:
                print(f'Unable to align master ({self.master1.name}), angle error is {master1_orientation_error * 180 / math.pi} (deg)')
            if not master2_aligned:
                print(f'Unable to align master ({self.master2.name}), angle error is {master2_orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master ({self.master1.name}) or ({self.master2.name}) gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED

        # let MTM position move freely, but lock orientation
        wrench = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.master1.body.servo_cf(wrench)
        self.master2.body.servo_cf(wrench)
        self.master1.lock_orientation(self.master1.measured_cp()[0].M)
        self.master2.lock_orientation(self.master2.measured_cp()[0].M)

        self.puppet.hold()

    def transition_clutched(self):
        if not self.clutch_pressed or not self.operator_is_present:
            self.enter_aligning()

    def run_clutched(self):
        # let arm move freely
        wrench = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.master1.body.servo_cf(wrench)
        self.master2.body.servo_cf(wrench)
        self.master1.lock_orientation(self.master1.measured_cp()[0].M)
        self.master2.lock_orientation(self.master2.measured_cp()[0].M)

        self.puppet.hold()

        #pass

    def enter_following(self):
        self.current_state = teleoperation.State.FOLLOWING
        # update MTM/PSM origins position
        self.update_initial_state()

        # set up gripper ghost to rate-limit jaw speed
        jaw_setpoint = self.puppet.jaw.setpoint_js()[0]
        if len(jaw_setpoint) != 1:
            print(f'{self.ral.node_name()}: unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet.name})')
            self.running = False
        self.gripper_ghost = self.jaw_to_gripper(jaw_setpoint[0])

        self.master1.use_gravity_compensation(True)
        self.master2.use_gravity_compensation(True)

    def transition_following(self):
        if not self.operator_is_present:
            self.enter_aligning()
        elif self.clutch_pressed:
            self.enter_clutched()

    def run_following(self):               
        """
        Forward Process
        """
        # Force channel
        # master1
        master1_measured_cf = self.master1.body.measured_cf()[0]   # (6,) numpy array
        master1_measured_cf[0:3] *= -1.0
        master1_measured_cf[3:6] *= 0   # turn off torque

        # master2
        master2_measured_cf = self.master2.body.measured_cf()[0]   # (6,) numpy array
        master2_measured_cf[0:3] *= -1.0
        master2_measured_cf[3:6] *= 0   # turn off torque

        # puppet
        puppet_measured_cf = self.puppet.body.measured_cf()[0]
        puppet_measured_cf[0:3] *= -1.0
        puppet_measured_cf[3:6] *= 0

        # force input
        #### add gamma?
        force_goal = 0.2 * (self.beta * master1_measured_cf + (1 - self.beta) * master2_measured_cf + puppet_measured_cf)
        force_goal = force_goal.tolist()


        # Position channel
        master1_measured_cp = self.master1.measured_cp()[0]       # return PyKDL.Frame
        master2_measured_cp = self.master2.measured_cp()[0]       # return PyKDL.Frame

        # set translation of psm
        master1_translation = master1_measured_cp.p - self.master1_cartesian_initial.p   # PyKDL.Vector
        master1_translation *= self.scale
        master2_translation = master2_measured_cp.p - self.master2_cartesian_initial.p   # PyKDL.Vector
        master2_translation *= self.scale
        master_total_translation = (master1_translation + master2_translation) / 2.0
        puppet_position = master_total_translation + self.puppet_cartesian_initial.p

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.master1_offset_angle += math.copysign(min(abs(self.master1_offset_angle), max_delta), -self.master1_offset_angle)
        self.master2_offset_angle += math.copysign(min(abs(self.master2_offset_angle), max_delta), -self.master2_offset_angle)
        
        master1_alignment_offset = PyKDL.Rotation.Rot(self.master1_offset_axis, self.master1_offset_angle)
        master1_rotation_alignment = master1_measured_cp.M * master1_alignment_offset

        master2_alignment_offset = PyKDL.Rotation.Rot(self.master2_offset_axis, self.master2_offset_angle)
        master2_rotation_alignment = master2_measured_cp.M * master2_alignment_offset

        # average rotation
        puppet_rotation = self.average_rotation(master1_rotation_alignment, master2_rotation_alignment)

        # set cartesian goal of psm
        puppet_cartesian_goal = PyKDL.Frame(puppet_rotation, puppet_position)


        # Velocity channel
        master1_measured_cv = self.master1.measured_cv()[0]   # (6,) numpy array
        master1_measured_cv[0:3] *= self.velocity_scale      # scale the linear velocity
        master1_measured_cv[3:6] *= 0.2      # scale down the angular velocity by 0.2

        master2_measured_cv = self.master2.measured_cv()[0]   # master2
        master2_measured_cv[0:3] *= self.velocity_scale     
        master2_measured_cv[3:6] *= 0.2     

        # average velocity
        puppet_velocity_goal = (master1_measured_cv + master2_measured_cv) / 2.0
        puppet_velocity_goal = puppet_velocity_goal.tolist()


        # Move
        self.puppet.servo_cs(puppet_cartesian_goal, puppet_velocity_goal, force_goal)


        ### Jaw/gripper teleop
        master1_gripper_measured_js = self.master1.gripper.measured_js()
        master2_gripper_measured_js = self.master2.gripper.measured_js()
        current_master1_gripper = master1_gripper_measured_js[0][0]
        current_master2_gripper = master2_gripper_measured_js[0][0]

        master1_ghost_lag = current_master1_gripper - self.gripper_ghost
        master2_ghost_lag = current_master2_gripper - self.gripper_ghost
        # average gripper lag
        ghost_lag = (master1_ghost_lag + master2_ghost_lag) / 2.0

        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_ghost += math.copysign(min(abs(ghost_lag), max_delta), ghost_lag)
        self.puppet.jaw.servo_jp(numpy.array([self.gripper_to_jaw(self.gripper_ghost)]))

        print("run_following is finished.")



        """
        Backward Process
        """
        # Position channel
        puppet_measured_cp = self.puppet.measured_cp()[0]       # return PyKDL.Frame
        puppet_translation = puppet_measured_cp.p - self.puppet_cartesian_initial.p     ##### should it update puppet initial cartesian after forward process???
        puppet_translation /= self.scale

        # set translation of mtm1
        master1_position = puppet_translation + self.master1_cartesian_initial.p
        # set translation of mtm2
        master2_position = puppet_translation + self.master2_cartesian_initial.p

        # set rotation of mtm1
        master1_rotation = puppet_measured_cp.M * master1_alignment_offset.Inverse()
        # set rotation of mtm2
        master2_rotation = puppet_measured_cp.M * master2_alignment_offset.Inverse()

        # set cartesian goal of mtm1 and mtm2
        master1_cartesian_goal = PyKDL.Frame(master1_rotation, master1_position)
        master2_cartesian_goal = PyKDL.Frame(master2_rotation, master2_position)


        # Velocity channel
        puppet_measured_cv = self.puppet.measured_cv()[0]   # (6,) numpy array
        puppet_measured_cv[0:3] /= self.velocity_scale      # scale the linear velocity
        puppet_measured_cv[3:6] *= 0.2      # scale down the angular velocity by 0.2
        master1_velocity_goal = puppet_measured_cv.tolist()
        master2_velocity_goal = puppet_measured_cv.tolist()


        # Move
        self.master1.servo_cs(master1_cartesian_goal, master1_velocity_goal, force_goal)
        self.master2.servo_cs(master2_cartesian_goal, master2_velocity_goal, force_goal)


        # """
        # plot
        # """
        # self.y_data_l.append([puppet_measured_cp.p.x(), puppet_measured_cp.p.y(), puppet_measured_cp.p.z()])
        # self.y_data_l_expected.append([puppet_position.p.x(), puppet_position.p.y(), puppet_position.p.z()])

        # self.m1_force.append(master1_measured_cf)
        # self.m2_force.append(master2_measured_cf)
        # self.puppet_force.append(puppet_measured_cf)
        # self.a += 1


    def home(self):
        print("Homing arms...")
        timeout = 10.0 # seconds
        if not self.puppet.enable(timeout) or not self.puppet.home(timeout):
            print('    ! failed to home {} within {} seconds'.format(self.puppet.name, timeout))
            return False

        if not self.master1.enable(timeout) or not self.master1.home(timeout):
            print('    ! failed to home {} within {} seconds'.format(self.master1.name, timeout))
            return False
        
        if not self.master2.enable(timeout) or not self.master2.home(timeout):
            print('    ! failed to home {} within {} seconds'.format(self.master2.name, timeout))
            return False

        print("    Homing is complete")
        return True


    def run(self):
        homed_successfully = self.home()
        if not homed_successfully:
            print("home not success")
            return
        
        # puppet_initial_position = numpy.array([0, 0, 0.13, 0, 0, 0])
        # self.puppet.move_jp(puppet_initial_position)

        teleop_rate = self.ral.create_rate(int(1/self.run_period))
        print("Running teleop at {} Hz".format(int(1/self.run_period)))

        self.enter_aligning()
        self.running = True

        while not self.ral.is_shutdown():
            # check if teleop state should transition
            if self.current_state == teleoperation.State.ALIGNING:
                self.transition_aligning()
                print("transition_aligning is finished.")
            elif self.current_state == teleoperation.State.CLUTCHED:
                self.transition_clutched()
                print("transition_clutched is finished.")
            elif self.current_state == teleoperation.State.FOLLOWING:
                self.transition_following()
                print("transition_following is finished.")
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            self.check_arm_state()
            
           
            if not self.running:
                break

            # run teleop state handler
            if self.current_state == teleoperation.State.ALIGNING:
                self.run_aligning()
                print("run_aligning is finished.")
            elif self.current_state == teleoperation.State.CLUTCHED:
                self.run_clutched()
                print("run_clutched is finished.")
            elif self.current_state == teleoperation.State.FOLLOWING:
                self.run_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            teleop_rate.sleep()

        # # save data
        # numpy.savetxt('multi_array.txt', self.y_data_l, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        # numpy.savetxt('multi_array_exp.txt', self.y_data_l_expected, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        # numpy.savetxt('multi_m1_force.txt', self.m1_force, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        # numpy.savetxt('multi_m2_force.txt', self.m2_force, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')
        # numpy.savetxt('multi_puppet_force.txt', self.puppet_force, fmt='%f', delimiter=' ', header='Column1 Column2 Column3', comments='')

        print(f"Program finished!")

class MTM:
            
    class ServoMeasCF:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_servo_cf()
            self.utils.add_measured_cf()

    class Gripper:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_measured_js()

    def __init__(self, ral, arm_name, timeout):
        self.name = arm_name
        self.ral = ral.create_child(arm_name)
        self.utils = crtk.utils(self, self.ral, timeout)

        self.utils.add_operating_state()
        self.utils.add_measured_cp()
        self.utils.add_measured_cv()
        self.utils.add_setpoint_cp()
        self.utils.add_move_cp()
        self.utils.add_servo_cs()

        self.gripper = self.Gripper(self.ral.create_child('gripper'), timeout)
        self.body = self.ServoMeasCF(self.ral.create_child('body'), timeout)

        # non-CRTK topics
        self.lock_orientation_pub = self.ral.publisher('lock_orientation',
                                                        geometry_msgs.msg.Quaternion,
                                                        latch = True, queue_size = 1)
        self.unlock_orientation_pub = self.ral.publisher('unlock_orientation',
                                                         std_msgs.msg.Empty,
                                                         latch = True, queue_size = 1)
        self.use_gravity_compensation_pub = self.ral.publisher('use_gravity_compensation',
                                                                std_msgs.msg.Bool,
                                                                latch = True, queue_size = 1)

    def lock_orientation(self, orientation):
        """orientation should be a PyKDL.Rotation object"""
        q = geometry_msgs.msg.Quaternion()
        q.x, q.y, q.z, q.w = orientation.GetQuaternion()
        self.lock_orientation_pub.publish(q)

    def unlock_orientation(self):
        self.unlock_orientation_pub.publish(std_msgs.msg.Empty())

    def use_gravity_compensation(self, gravity_compensation):
        """Turn on/off gravity compensation (only applies to Cartesian effort mode)"""
        msg = std_msgs.msg.Bool(data=gravity_compensation)
        self.use_gravity_compensation_pub.publish(msg)

class PSM:
    class MeasureCF:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_measured_cf()
    class MeasuredCP:
        def __init__(self,ral,timeout):
            self.utils = crtk.utils(self,ral,timeout)
            self.utils.add_measured_cp()
    class Jaw:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_setpoint_js()
            self.utils.add_servo_jp()

    def __init__(self, ral, arm_name, timeout):
        self.name = arm_name
        self.ral = ral.create_child(arm_name)
        self.utils = crtk.utils(self, self.ral, timeout)

        self.utils.add_operating_state()
        self.utils.add_setpoint_cp()
        self.utils.add_servo_cp()
        self.utils.add_servo_cs()
        self.utils.add_hold()
        self.utils.add_measured_cv()
        self.utils.add_move_jp()
        self.utils.add_measured_cp()

        self.body = self.MeasureCF(self.ral.create_child('body'), timeout)
        self.local = self.MeasuredCP(self.ral.create_child('local'),timeout)
        self.jaw = self.Jaw(self.ral.create_child('jaw'), timeout)

    

if __name__ == '__main__':
    # extract ros arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mtm', type = str, required = True, nargs=2,
                        choices = ['MTML', 'MTMR'],
                        help = 'Must type in two MTM arm names corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-p', '--psm', type = str, required = True,
                        choices = ['PSM1', 'PSM2', 'PSM3'],
                        help = 'PSM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-c', '--clutch', type = str, default='/console_1/clutch',
                        help = 'ROS topic corresponding to clutch button/pedal input')
    parser.add_argument('-o', '--operator', type = str, default='/console_1/operator_present', const=None, nargs='?',
                        help = 'ROS topic corresponding to operator present button/pedal/sensor input - use "-o" without an argument to disable')
    parser.add_argument('-n', '--no-mtm-alignment', action='store_true',
                        help="don't align mtm (useful for using haptic devices as MTM which don't have wrist actuation)")
    parser.add_argument('-i', '--interval', type=float, default=0.0025,
                        help = 'time interval/period to run at - should be as long as console\'s period to prevent timeouts')
    args = parser.parse_args(argv)

    ral = crtk.ral('dvrk_python_teleoperation')
    mtm1 = MTM(ral, args.mtm[0], timeout=20*args.interval)
    mtm2 = MTM(ral, args.mtm[1], timeout=20*args.interval)
    psm = PSM(ral, args.psm, timeout=20*args.interval)
    application = teleoperation(ral, mtm1, mtm2, psm, args.clutch, args.interval,
                                not args.no_mtm_alignment, operator_present_topic = args.operator, alpha = 0.5, beta = 0.5)
     
    ral.spin_and_execute(application.run)
