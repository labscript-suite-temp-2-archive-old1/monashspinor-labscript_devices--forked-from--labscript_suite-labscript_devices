#####################################################################
#                                                                   #
# /MOGLabsXRF421.py                                                 #
#                                                                   #
# Copyright 2013, Monash University                                 #
#                                                                   #
# This file is part of the module labscript_devices, in the         #
# labscript suite (see http://labscriptsuite.org), and is           #
# licensed under the Simplified BSD License. See the license.txt    #
# file in the root of the project for the full license.             #
#                                                                   #
#####################################################################
from labscript_devices import runviewer_parser, labscript_device, BLACS_tab, BLACS_worker

from labscript import IntermediateDevice, DDS, StaticDDS, Device, config, LabscriptError, set_passed_properties
from labscript_utils.unitconversions import MOGLabsDDSFreqConversion, MOGLabsDDSAmpConversion

import numpy as np
import labscript_utils.h5_lock, h5py
import labscript_utils.properties

        
@labscript_device
class MOGLabsXRF421(IntermediateDevice):
    description = 'MOGLabs-XRF421 DDS outputs'
    allowed_children = [DDS]
    clock_limit = 1e6

    @set_passed_properties(
        property_names = {'connection_table_properties': ['update_mode']}
        )
    def __init__(self, name, parent_device, 
                 addr, port=None, **kwargs):

        IntermediateDevice.__init__(self, name, parent_device, **kwargs)
        if addr.startswith('COM') or addr == 'USB':
            if port is not None: addr = 'COM%d'%port
            addr = addr.split(' ', 1)[0]
            addr = '%s,%i' % (addr, 115200)
            self.BLACS_connection = addr
            self.is_usb = True
        else:
            if not ':' in addr:
                if port is None: port=7802
                addr = '%s:%d' % (addr, port)
            self.BLACS_connection = addr
            self.is_usb = False
        
    def add_device(self, device):
        Device.add_device(self, device)
        device.frequency.default_value = 80e6   # TODO: Is this used anywhere?
        if isinstance(device, DDS):
            # Check that the user has not specified another digital line as the gate for this DDS, that doesn't make sense.
            # Then instantiate a DigitalQuantity to keep track of gating.
            if device.gate is None:
                device.gate = DigitalQuantity(device.name + '_gate', device, 'gate')
            else:
                raise LabscriptError('You cannot specify a digital gate ' +
                                     'for a DDS connected to %s. '% (self.name) + 
                                     'The digital gate is always internal to the XRF421.')
            
    def get_default_unit_conversion_classes(self, device):
        """Child devices call this during their __init__ (with themselves
        as the argument) to check if there are certain unit calibration
        classes that they should apply to their outputs, if the user has
        not otherwise specified a calibration class"""
        if device.connection in ['channel 0', 'channel 1']:
            # Default calibration classes for the non-static Freq, Amp, Phase channels:
            return MOGLabsDDSFreqConversion, MOGLabsDDSAmpConversion, None
        else:
            return None, None, None
        
    def quantise_freq(self, data, device):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # Ensure that frequencies are within bounds:
        if np.any(data > 400e6)  or np.any(data < 20e6):
            raise LabscriptError('%s %s '% (device.description, device.name) +
                              'can only have frequencies between 20MHz and 400MHz, ' + 
                              'the limit imposed by %s.'%self.name)
        # It's faster to add 0.5 then typecast than to round to integers first:
        scale_factor = 10   # round to 0.1 Hz
        data = np.array((scale_factor*data)+0.5,dtype=np.uint32)
        return data, scale_factor
        
    def quantise_phase(self, data, device):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # ensure that phase wraps around:
        data %= 360
        data /= 360.
        # It's faster to add 0.5 then typecast than to round to integers first:
        scale_factor = 2**16
        data = np.array((scale_factor*data)-0.5,dtype=np.uint32)
        return data, scale_factor
        
    def quantise_amp(self, data, device):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # ensure that amplitudes are within bounds:
        if np.any(data > 1)  or np.any(data < 0):
            raise LabscriptError('%s %s '%(device.description, device.name) +
                              'can only have amplitudes between 0 and 1 (fractional amplitude), ' + 
                              'the limit imposed by %s.'% self.name)
        scale_factor = 2**14
        data = np.array((scale_factor)-0.5, dtype=np.uint16)
        return data, scale_factor
        
    def generate_code(self, hdf5_file):
        DDSs = {}
        for output in self.child_devices:
            # Check that the instructions will fit into RAM:
            if isinstance(output, DDS) and len(output.frequency.raw_output) > 8191 - 2: # -2 to include space for dummy instructions
                raise LabscriptError('%s can only support 8191 instructions. '%self.name +
                                     'Please decrease the sample rates of devices on the same clock, ' + 
                                     'or connect %s to a different pseudoclock.'%self.name)
            try:
                prefix, channel = output.connection.split()
                channel = int(channel)
            except:
                raise LabscriptError('%s %s has invalid connection string: \'%s\'. ' % (output.description, output.name, str(output.connection)) + 
                                     'Format must be \'channel n\' with n from 0 to 1.')
            DDSs[channel] = output
        for connection in DDSs:
            if connection in range(4):
                # Dynamic DDS
                dds = DDSs[connection]   
                dds.frequency.raw_output, dds.frequency.scale_factor = self.quantise_freq(dds.frequency.raw_output, dds)
                dds.phase.raw_output, dds.phase.scale_factor = self.quantise_phase(dds.phase.raw_output, dds)
                dds.amplitude.raw_output, dds.amplitude.scale_factor = self.quantise_amp(dds.amplitude.raw_output, dds)
            else:
                raise LabscriptError('%s %s has invalid connection string: \'%s\'. '%(dds.description, dds.name, str(dds.connection)) + 
                                     'Format must be \'channel n\' with n from 0 to 1.')
                                
        dtypes = [('freq%d'%i,np.uint32) for i in range(2)] + \
                 [('phase%d'%i,np.uint16) for i in range(2)] + \
                 [('amp%d'%i,np.uint16) for i in range(2)]
                          
        clockline = self.parent_clock_line
        pseudoclock = clockline.parent_device
        times = pseudoclock.times[clockline]
       
        out_table = np.zeros(len(times), dtype=dtypes)
        out_table['freq0'].fill(1)
        out_table['freq1'].fill(1)
        
        for connection in range(2):
            if not connection in DDSs:
                continue
            dds = DDSs[connection]
            # The last two instructions are left blank, for BLACS
            # to fill in at program time.
            out_table['freq%d'%connection][:] = dds.frequency.raw_output
            out_table['amp%d'%connection][:] = dds.amplitude.raw_output
            out_table['phase%d'%connection][:] = dds.phase.raw_output

        grp = self.init_device_group(hdf5_file)
        grp.create_dataset('TABLE_DATA',compression=config.compression,data=out_table) 
        grp.create_dataset('STATIC_DATA',compression=config.compression,data=static_table) 
        self.set_property('frequency_scale_factor', 10, location='device_properties')
        self.set_property('amplitude_scale_factor', 2**14, location='device_properties')
        self.set_property('phase_scale_factor', 2**16, location='device_properties')



import time

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED  

from blacs.device_base_class import DeviceTab

@BLACS_tab
class MOGLabsXRF421Tab(DeviceTab):
    def initialise_GUI(self):        
        # Capabilities
        self.base_units =    {'freq': 'Hz',    'amp': 'Arb',   'phase': 'Degrees'}
        self.base_min =      {'freq': 20.0e6,  'amp': 0,       'phase': 0}
        self.base_max =      {'freq': 170.0e6, 'amp': 1,       'phase': 360}
        self.base_step =     {'freq': 1.0e6,   'amp': 0.001,   'phase': 1}
        self.base_decimals = {'freq': 1,       'amp': 4,       'phase': 3} # TODO: find out what the phase precision is!
        self.num_DDS = 2
        
        # Create DDS Output objects
        dds_prop = {}
        for i in range(self.num_DDS): # 2 is the number of DDS outputs on this device
            dds_prop['channel %d' % i] = {}
            for subchnl in ['freq', 'amp', 'phase']:
                dds_prop['channel %d' % i][subchnl] = {'base_unit': self.base_units[subchnl],
                                                       'min': self.base_min[subchnl],
                                                       'max': self.base_max[subchnl],
                                                       'step': self.base_step[subchnl],
                                                       'decimals': self.base_decimals[subchnl]
                                                      }
            dds_prop['channel %d' % i]['gate'] = {}

        # Create the output objects    
        self.create_dds_outputs(dds_prop)        
        # Create widgets for output objects
        dds_widgets, ao_widgets, do_widgets = self.auto_create_widgets()
        # and auto place the widgets in the UI
        self.auto_place_widgets(("DDS Outputs", dds_widgets))
        
        # Get connection details
        connection_object = self.settings['connection_table'].find_by_name(self.device_name)
        # Store the COM port / IP address to be used
        blacs_connection =  str(connection_object.BLACS_connection)
        if ',' in blacs_connection:
            self.is_usb = True
            self.com_port, baud_rate = blacs_connection.split(',')
            self.baud_rate = int(baud_rate)
            self.ip, self.port = None, None
        elif ':' in blacs_connection:
            self.is_usb = False
            self.ip, port = blacs_connection.split(':')
            self.port = int(port)
            self.com_port, self.baud_rate = None, None
        
        # self.update_mode = connection_object.properties.get('update_mode', 'synchronous')
        
        # Create and set the primary worker
        self.create_worker("main_worker", MOGLabsXRF421Worker, {'com_port': self.com_port,
                                                                'baud_rate': self.baud_rate,
                                                                'ip': self.ip,
                                                                'port': self.port,
                                                                'is_usb': self.is_usb})
        self.primary_worker = "main_worker"

        # Set the capabilities of this device
        self.supports_remote_value_check(True)
        self.supports_smart_programming(True) 

@BLACS_worker        
class MOGLabsXRF421Worker(Worker):
    def init(self, timeout=1, check=True, debug=False):
        global serial; import serial
        global socket; import socket
        global h5py; import labscript_utils.h5_lock, h5py
        self.smart_cache = {'TABLE_DATA': ''}
        self.reconnect(timeout, check)
        for i in [1, 2]:
            self.cmd('MODE,%i,TSB' % i)             # set into table mode
            # self.cmd('TABLE,REARM,%i,ON' % i)       # enable table rearm
            # self.cmd('TABLE,RESTART,%i,ON' % i)     # enable table restart

    #================ GENERIC COMMUNICATIONS METHODS ================
    def reconnect(self, timeout=1, check=True):
        "Reestablish connection with unit"
        if hasattr(self, 'dev'): self.dev.close()
        if self.is_usb:
            self.dev = serial.Serial(self.com_port, self.baud_rate, bytesize=8, parity='N', stopbits=1, timeout=timeout, writeTimeout=0)
        else:
            self.dev = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dev.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.dev.settimeout(timeout)
            self.dev.connect((self.ip, self.port))
        # check the connection?
        if check:
            try:
                self.info = self.ask('info')
                self.serial = self.ask('get,serial')
            except Exception as E:
                print '!', E
                raise Exception('Device did not respond to query.')

    def cmd(self, cmd):
        "Send the specified command, and check the response is OK"
        self.flush()
        self.send(cmd)
        resp = self.recv()
        if resp.startswith('OK'):
            return resp
        else:
            raise Exception(resp)
        
    def ask(self, cmd):
        "Send followed by receive"
        # check if there's any response waiting on the line
        self.flush()
        self.send(cmd)
        resp = self.recv().strip()
        if resp.startswith('ERR:'):
            raise Exception(resp[4:].strip())
        return resp
        
    def ask_dict(self, cmd):
        "Send a request which returns a dictionary response"
        resp = self.ask(cmd)
        # might start with "OK"
        if resp.startswith('OK'): resp = resp[3:].strip()
        # expect a colon in there
        if not ':' in resp: raise Exception('Response to "%s" not a dictionary'%cmd)
        # response could be comma-delimited (new) or newline-delimited (old)
        vals = collections.OrderedDict()
        for entry in returnsp.split(',' if ',' in resp else '\n'):
            name, val = entry.split(':')
            vals[name.strip()] = val.strip()
        return vals
        
    def ask_bin(self, cmd):
        "Send a request which returns a binary response"
        self.send(cmd)
        head = self.recv_raw(4)
        print repr(head)
        # is it an error message?
        if head == 'ERR:': raise RuntimeError(head+self.recv())
        datalen = struct.unpack('<L',head)[0]
        data = self.recv_raw(datalen)
        if len(data) != datalen: raise RuntimeError('Binary response block has incorrect length')
        return data
    
    def send(self,cmd):
        "Send command, appending newline if not present"
        if not cmd.endswith('\r\n'): cmd += '\r\n'
        self.send_raw(cmd)
    
    def has_data(self,timeout=0):
        if self.is_usb:
            if self.dev.inWaiting(): return True
            if timeout == 0: return False
            time.sleep(timeout)
            return self.dev.inWaiting > 0
        else:
            return len(select.select([self.dev],[],[],timeout)[0]) > 0
        
    def flush(self,buffer=256):
        while self.has_data():
            dat = self.recv(buffer)
            if self._DEBUG: print 'FLUSHED', repr(dat)
    
    def recv(self,buffer=256):
        "A somewhat robust multi-packet receive call"
        if self.is_usb:
            data = self.dev.readline(buffer)
            if len(data):
                t0 = self.dev.timeout
                self.dev.timeout = 0 if data.endswith('\r\n') else 0.1
                while True:
                    segment = self.dev.readline(buffer)
                    if len(segment) == 0: break
                    data += segment
                self.dev.timeout = t0
            if len(data) == 0: raise RuntimeError('timed out')
        else:
            data = self.dev.recv(buffer)
            timeout = 0 if data.endswith('\r\n') else 0.1
            while self.has_data(timeout):
                try:
                    segment = self.dev.recv(buffer)
                except IOError:
                    if len(data): break
                    raise
                data += segment
        if self._DEBUG: print '<<',len(data),repr(data)
        return data
    
    def send_raw(self,cmd):
        "Send, without appending newline"
        if self._DEBUG and len(cmd) < 256: print '>>',repr(cmd)
        if self.is_usb:
            return self.dev.write(cmd)
        else:
            return self.dev.send(cmd)
    
    def recv_raw(self,size):
        "Receive exactly 'size' bytes"
        buffer = ''
        while size > 0:
            if self.is_usb:
                chunk = self.dev.read(size)
            else:
                chunk = self.dev.recv(size)
            buffer += chunk
            size -= len(chunk)
        if self._DEBUG:
            print '<< RECV_RAW got', len(buffer)
            print '<<', repr(buffer)
        return buffer
        
    def set_timeout(self,val):
        if self.is_usb:
            old = self.dev.timeout
            self.dev.timeout = val
            return old
        else:
            old = self.dev.gettimeout()
            self.dev.settimeout(val)
            return old

    def set_get(self,name,val):
        "Set specified name and then query it"
        self.cmd('set,'+name+','+str(val)+'\n')
        actualval = self.ask('get,'+name)
        if self._DEBUG: print 'SET',name,'=',repr(val),repr(actualval)
        return actualval

    def close(self):
        if hasattr(self, 'dev'): self.dev.close()

    #================ SPECIFIC COMMUNICATIONS METHODS ================
    def parse_response(response, return_raw=False):
        units = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6,
                 'dBm': 1, 'deg': 1}
        val, unit, raw = response.split()[-3:]
        val = float(val) # * units[unit]
        raw = int(raw.replace('(', '').replace(')', ''), 0)
        if return_raw:
            return raw, val
        else:
            return val

    def get_val(self, key, channel):
        response = self.ask('%s,%i' % (key, channel))
        return parse_response(response)

    def set_val(self, key, channel, val):
        response = self.ask('%s,%i,%f' % (key, channel, val))
        if not response.startswith('OK'):
            raise Exception('Failed to set %s of channel %i to %s' % (key, channel, val))
        return parse_response(response)

    def get_status(self, channel):
        response = self.ask('STATUS,%i', channel)
        status = dict([[y.strip() for y in x.split(':')] for x in response.split(',')])
        # status['AMP'] = status.pop('POW')
        return status

    def program_static(self, channel, key, value):
        if key in ['FREQ', 'POW', 'PHASE']:
            self.set_val(channel, key, value)
        elif key == 'SIG':
            if value in ['OFF', 'off', 0, False]:
                response = self.ask('OFF,%i,SIG')
            elif value in ['ON', 'on', 1, True]:
                response = self.ask('ON,%i,SIG')
        elif key == 'AMP':
            if value in ['OFF', 'off', 0, False]:
                response = self.ask('OFF,%i,POW')
            elif value in ['ON', 'on', 1, True]:
                response = self.ask('ON,%i,POW')
        else:
            raise TypeError('Quantity name must be one of FREQ, POW, PHASE, SIG, or AMP.')
        if not response.startswith('OK'):
            raise Exception('Error: Failed to disable/enable signal/amp on channel %i' % channel)

    #================ METHODS REQUIRED BY BLACS ======================
    def check_remote_values(self):
        try:
            results = {}
            for i in [1, 2]:
                vals = {key: get_val(key, i) for key in ['FREQ', 'POW', 'PHASE']}
                status = self.get_status(i)
                vals['AMP'] = status['POW']
                vals['SIG'] = status['SIG']
                results['channel %d' % i] = vals
        except socket.timeout:
            raise Exception('Failed to check remote values. Timed out.')
        return results
        
    def program_manual(self, front_panel_values):
        for i in [1, 2]:
            # Get a dictionary of front panel values for this channel
            vals = front_panel_values['channel %d' % (i-1)]
            # Check the length of the programmed table
            table_length = dev.ask('TABLE,LENGTH,%i' % i)
            # If fewer than two instructions, create an empty table with two entries
            if table_length < 2:
                dev.cmd('TABLE,CLEAR,%i' % i)
                dev.cmd('TABLE,LENGTH,%i,2' % i)
            # Create two short instructions with the new front_panel_values, wait on the second
            dev.cmd('TABLE,ENTRY,%i,1,%f,%f,%f,1' % (i, vals['freq'], vals['amp'], vals['phase']))
            dev.cmd('TABLE,ENTRY,%i,2,%f,%f,%f,1,TRIG' % (i, vals['freq'], vals['amp'], vals['phase']))
            # Stop the table and restart it to load the new values
            dev.ask('TABLE,STOP,%i' % i)
            dev.ask('TABLE,START,%i' % i)
            vals['gate']
        return self.check_remote_values()
     
    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        # Store the initial values in case we have to abort and restore them:
        self.initial_values = initial_values
        # Store the final values to for use during transition_to_static:
        self.final_values = {}
        table_data = None
        with h5py.File(h5file) as hdf5_file:
            group = hdf5_file['/devices/' + device_name]
            # Now program the buffered outputs:
            if 'TABLE_DATA' in group:
                table_data = group['TABLE_DATA'][:]
        
        # Now program the buffered outputs:
        if table_data is not None:
            data = table_data
            oldtable = self.smart_cache['TABLE_DATA']
            for i, line in enumerate(data):
                for ch in [1, 2]:
                    old_vals = {}
                    new_vals = {}
                    for key in ['freq', 'amp', 'phase', 'dds_en']:
                        new_vals[key] = line['%s%d' % (key, ch)]
                        old_vals[key] = oldtable[i-1]['%s%d' % (key, ch)]
                    if fresh or i >= len(oldtable) or new_vals != old_vals:
                        ix = i+2    # First two instructions are for static updates
                        ix += 1     # mogrf indexing begins at 1
                        inst = 'TABLE,ENTRY,%i,%i,%f,%f,%f,%f' % (ch, ix, vals['freq'], vals['amp'], vals['phase'])
                        self.connection.write('t%d %04x %08x,%04x,%04x,ff\r\n'%(ch, i, line['freq%d'%ch],line['phase%d'%ddsno],line['amp%d'%ddsno]))
                        self.connection.readline()

                self.logger.debug('Time spent on line %s: %s'%(i,tt))
            # Store the table for future smart programming comparisons:
            try:
                self.smart_cache['TABLE_DATA'][:len(data)] = data
                self.logger.debug('Stored new table as subset of old table')
            except: # new table is longer than old table
                self.smart_cache['TABLE_DATA'] = data
                self.logger.debug('New table is longer than old table and has replaced it.')
                
            # Get the final values of table mode so that the GUI can
            # reflect them after the run:
            self.final_values['channel 0'] = {}
            self.final_values['channel 1'] = {}
            self.final_values['channel 0']['freq'] = data[-1]['freq0']/10.0
            self.final_values['channel 1']['freq'] = data[-1]['freq1']/10.0
            self.final_values['channel 0']['amp'] = data[-1]['amp0']/1023.0
            self.final_values['channel 1']['amp'] = data[-1]['amp1']/1023.0
            self.final_values['channel 0']['phase'] = data[-1]['phase0']*360/16384.0
            self.final_values['channel 1']['phase'] = data[-1]['phase1']*360/16384.0
            
            # Transition to table mode:
            self.connection.write('m t\r\n')
            self.connection.readline()
            if self.update_mode == 'synchronous':
                # Transition to hardware synchronous updates:
                self.connection.write('I e\r\n')
                self.connection.readline()
                # We are now waiting for a rising edge to trigger the output
                # of the second table pair (first of the experiment)
            elif self.update_mode == 'asynchronous':
                # Output will now be updated on falling edges.
                pass
            else:
                raise ValueError('invalid update mode %s'%str(self.update_mode))
                
            
        return self.final_values
    
    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)
        
    def abort_buffered(self):
        # TODO: untested
        return self.transition_to_manual(True)
    
    def transition_to_manual(self,abort = False):
        self.connection.write('m 0\r\n')
        if self.connection.readline() != "OK\r\n":
            raise Exception('Error: Failed to execute command: "m 0"')
        self.connection.write('I a\r\n')
        if self.connection.readline() != "OK\r\n":
            raise Exception('Error: Failed to execute command: "I a"')
        if abort:
            # If we're aborting the run, then we need to reset DDSs 2 and 3 to their initial values.
            # 0 and 1 will already be in their initial values. We also need to invalidate the smart
            # programming cache for them.
            values = self.initial_values
            DDSs = [2,3]
            self.smart_cache['STATIC_DATA'] = None
        else:
            # If we're not aborting the run, then we need to set DDSs 0 and 1 to their final values.
            # 2 and 3 will already be in their final values.
            values = self.final_values
            DDSs = [0,1]
            
        # only program the channels that we need to
        for ddsnumber in DDSs:
            channel_values = values['channel %d'%ddsnumber]
            for subchnl in ['freq','amp','phase']:            
                self.program_static(ddsnumber,subchnl,channel_values[subchnl])
            
        # return True to indicate we successfully transitioned back to manual mode
        return True
                     
    def shutdown(self):
        self.connection.close()
